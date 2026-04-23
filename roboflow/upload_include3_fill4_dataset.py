from __future__ import annotations

import argparse
import glob
from pathlib import Path

from roboflow import Roboflow


API_KEY = "4jupREKAZ5yv5g93WVQX"
WORKSPACE_ID = "lulus-workspace-pe1hc"
PROJECT_ID = "icebergseg"
BATCH_NAME = "restart_2026_03_25"

DATASET_DIR = Path(
    "/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/"
    "S2-iceberg-areas/roboflow_manual_upload_include3_fill4_iceberg"
)

ANNOTATION_FILE = DATASET_DIR / "_annotations.coco.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload the include-3 fill-4 iceberg dataset to Roboflow.")
    parser.add_argument(
        "--project-id",
        default=PROJECT_ID,
        help=f"Roboflow project id to upload into. Default: {PROJECT_ID}.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help=f"Dataset folder containing PNGs and _annotations.coco.json. Default: {DATASET_DIR}.",
    )
    parser.add_argument(
        "--batch-name",
        default=BATCH_NAME,
        help=f"Batch name to assign in Roboflow. Default: {BATCH_NAME}.",
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=1,
        help="1-based image index to start or resume from. Default: 1.",
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=3,
        help="Number of retries for each image+annotation upload. Default: 3.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    annotation_file = dataset_dir / "_annotations.coco.json"

    if API_KEY == "YOUR_PRIVATE_API_KEY":
        raise SystemExit("Set API_KEY in roboflow/upload_include3_fill4_dataset.py first.")
    if WORKSPACE_ID == "YOUR_WORKSPACE_SLUG":
        raise SystemExit("Set WORKSPACE_ID in roboflow/upload_include3_fill4_dataset.py first.")
    if not annotation_file.exists():
        raise SystemExit(f"Missing annotation file: {annotation_file}")

    image_paths = sorted(glob.glob(str(dataset_dir / "*.png")))
    if not image_paths:
        raise SystemExit(f"No PNG images found in {dataset_dir}")
    if args.start_at < 1 or args.start_at > len(image_paths):
        raise SystemExit(f"--start-at must be between 1 and {len(image_paths)}")

    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE_ID).project(args.project_id)

    print(
        f"Uploading {len(image_paths)} images from {dataset_dir} to {WORKSPACE_ID}/{args.project_id} "
        f"starting at index {args.start_at} with {args.num_retries} retries..."
    )
    for idx, image_path in enumerate(image_paths, start=1):
        if idx < args.start_at:
            continue
        try:
            result = project.single_upload(
                image_path=image_path,
                annotation_path=str(annotation_file),
                batch_name=args.batch_name,
                num_retry_uploads=args.num_retries,
            )
        except Exception as exc:
            raise SystemExit(
                f"Upload failed at [{idx}/{len(image_paths)}] {Path(image_path).name}: {exc}"
            ) from exc
        print(f"[{idx}/{len(image_paths)}] {Path(image_path).name}: {result}")


if __name__ == "__main__":
    main()
