import argparse
import itertools
import json
import os
import pickle
from typing import Dict, List, Sequence

import numpy as np


DEFAULT_PARAM_GRID = {
    "sxy_gaussian": [3],
    "compat_gaussian": [3],
    "sxy_bilateral": [40, 80],
    "srgb_bilateral": [3, 5],
    "compat_bilateral": [4],
    "iterations": [5, 10],
}

DEFAULT_NOAUG_CHECKPOINT = (
    "/mnt/research/v.gomezgilyaspik/students/smishra/"
    "S2-iceberg-areas/runs/s2_v2_noaug/best_model.pth"
)

DEFAULT_V2_NORM_SOURCE = (
    "/mnt/research/v.gomezgilyaspik/students/smishra/"
    "S2-iceberg-areas/S2UnetPlusPlus/train_validate_test_v2/X_train.pkl"
)


def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def default_parent_crf_dir() -> str:
    # The scripts live in test-crf/, and the original pickles sit one level up.
    return os.path.abspath(os.path.join(script_dir(), ".."))


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_pickle(path: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_pickle(path: str, obj) -> None:
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def squeeze_mask_channel(labels: np.ndarray) -> np.ndarray:
    # The training data stores masks as (N,1,H,W), but the CRF code wants (N,H,W).
    labels = np.asarray(labels)
    if labels.ndim == 4 and labels.shape[1] == 1:
        return labels[:, 0]
    return labels


def one_hot_from_labels(labels: np.ndarray, n_classes: int) -> np.ndarray:
    # This is handy for debugging because it makes perfectly confident probabilities from the labels.
    flat = np.eye(n_classes, dtype=np.float32)[labels]
    return np.moveaxis(flat, -1, 1)


def synthetic_probs_from_labels(
    labels: np.ndarray,
    n_classes: int = 3,
    true_prob: float = 0.90,
) -> np.ndarray:
    # This gives us "realistic enough" softmax maps for testing the CRF pipeline without a checkpoint.
    if not (0.0 < true_prob < 1.0):
        raise ValueError("true_prob must be between 0 and 1")
    off_prob = (1.0 - true_prob) / (n_classes - 1)
    probs = np.full((labels.shape[0], n_classes, labels.shape[1], labels.shape[2]), off_prob, dtype=np.float32)
    for cls in range(n_classes):
        probs[:, cls][labels == cls] = true_prob
    return probs


def uniform_probs(n_samples: int, height: int, width: int, n_classes: int = 3) -> np.ndarray:
    return np.full((n_samples, n_classes, height, width), 1.0 / n_classes, dtype=np.float32)


def compute_channel_stats(chips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # This is optional and only used if we ever need to reproduce a normalized training run.
    if chips.ndim != 4:
        raise ValueError(f"Expected chips shape (N,C,H,W), got {chips.shape}")
    mean = chips.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = chips.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def normalize_chips(chips: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(1, -1, 1, 1)
    return ((chips.astype(np.float32) - mean) / std).astype(np.float32)


def compute_iou(pred: np.ndarray, target: np.ndarray, classes: Sequence[int] = (1,)) -> float:
    # We ignore the ocean class here and focus on the classes we care about comparing.
    scores: List[float] = []
    for cls in classes:
        pred_mask = pred == cls
        target_mask = target == cls
        union = np.logical_or(pred_mask, target_mask).sum()
        if union == 0:
            continue
        inter = np.logical_and(pred_mask, target_mask).sum()
        scores.append(float(inter) / float(union))
    return float(np.mean(scores)) if scores else 0.0


def compute_class_area_bias(pred: np.ndarray, target: np.ndarray, cls: int = 1) -> float:
    # This tells us whether CRF changes total iceberg area, not just boundary overlap.
    pred_area = float((pred == cls).sum())
    target_area = float((target == cls).sum())
    if target_area == 0:
        return 0.0 if pred_area == 0 else float("inf")
    return 100.0 * (pred_area - target_area) / target_area


def scale_chip_to_uint8(chip: np.ndarray) -> np.ndarray:
    # The bilateral term expects image-like values, so we stretch each band into uint8 just for CRF.
    if chip.ndim != 3:
        raise ValueError(f"Expected chip shape (C,H,W), got {chip.shape}")
    channels: List[np.ndarray] = []
    for band in chip:
        lo = float(np.nanpercentile(band, 2))
        hi = float(np.nanpercentile(band, 98))
        scaled = np.clip((band - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        channels.append((scaled * 255.0).astype(np.uint8))
    return np.stack(channels, axis=-1)


def try_import_densecrf():
    # Some systems have the original package, others have the maintained fork.
    try:
        import pydensecrf.densecrf as dcrf  # type: ignore
        from pydensecrf.utils import unary_from_softmax  # type: ignore
        return dcrf, unary_from_softmax
    except ImportError:
        try:
            import pydensecrf2.densecrf as dcrf  # type: ignore
            from pydensecrf2.utils import unary_from_softmax  # type: ignore
            return dcrf, unary_from_softmax
        except ImportError as exc:
            raise ImportError(
                "DenseCRF dependency not found. Install `pydensecrf` or `pydensecrf2` to run Phase 2."
            ) from exc


def apply_densecrf(prob: np.ndarray, chip: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    dcrf, unary_from_softmax = try_import_densecrf()
    n_classes, height, width = prob.shape

    # The chip only affects the pairwise appearance term; the class probabilities drive the unary term.
    image_uint8 = scale_chip_to_uint8(chip)
    unary = unary_from_softmax(np.ascontiguousarray(prob))

    dense = dcrf.DenseCRF2D(width, height, n_classes)
    dense.setUnaryEnergy(unary)
    # This term only cares about nearby pixels agreeing in space.
    dense.addPairwiseGaussian(
        sxy=float(params["sxy_gaussian"]),
        compat=float(params["compat_gaussian"]),
    )
    # This term also looks at band values, so sharp image changes can preserve boundaries.
    dense.addPairwiseBilateral(
        sxy=float(params["sxy_bilateral"]),
        srgb=float(params["srgb_bilateral"]),
        rgbim=np.ascontiguousarray(image_uint8),
        compat=float(params["compat_bilateral"]),
    )

    # After a few mean-field updates, we convert the refined distribution back to hard labels.
    q = dense.inference(int(params["iterations"]))
    q = np.asarray(q, dtype=np.float32).reshape((n_classes, height, width))
    return np.argmax(q, axis=0).astype(np.uint8)


def parse_int_list(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def build_param_grid(args: argparse.Namespace) -> List[Dict[str, int]]:
    if getattr(args, "param_json", None):
        with open(args.param_json, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            grid = loaded
        elif isinstance(loaded, list):
            return loaded
        else:
            raise ValueError("param_json must contain either a dict grid or a list of param sets")
    else:
        # The CLI takes comma-separated values so a tiny grid search is easy to launch from the shell.
        grid = {
            "sxy_gaussian": parse_int_list(args.sxy_gaussian),
            "compat_gaussian": parse_int_list(args.compat_gaussian),
            "sxy_bilateral": parse_int_list(args.sxy_bilateral),
            "srgb_bilateral": parse_int_list(args.srgb_bilateral),
            "compat_bilateral": parse_int_list(args.compat_bilateral),
            "iterations": parse_int_list(args.iterations),
        }

    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    param_sets = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    max_param_sets = getattr(args, "max_param_sets", None)
    if max_param_sets:
        param_sets = param_sets[:max_param_sets]
    return param_sets


def summarize_array(name: str, arr: np.ndarray) -> str:
    return f"{name}: shape={arr.shape}, dtype={arr.dtype}"
