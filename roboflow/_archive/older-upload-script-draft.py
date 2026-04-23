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
    parser.add_argument("--start-at", type=int, default=1, help="1-based image index to start or resume from.")
    parser.add_argument("--num-retries", type=int, default=3, help="Retries for each image+annotation upload.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if API_KEY == "YOUR_PRIVATE_API_KEY":
        raise SystemExit("Set API_KEY first.")
    if not ANNOTATION_FILE.exists():
        raise SystemExit(f"Missing annotation file: {ANNOTATION_FILE}")

    image_paths = sorted(glob.glob(str(DATASET_DIR / "*.png")))
    if not image_paths:
        raise SystemExit(f"No PNG images found in {DATASET_DIR}")
    if args.start_at < 1 or args.start_at > len(image_paths):
        raise SystemExit(f"--start-at must be between 1 and {len(image_paths)}")

    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)

    print(
        f"Uploading {len(image_paths)} images to {WORKSPACE_ID}/{PROJECT_ID} "
        f"starting at index {args.start_at} with {args.num_retries} retries..."
    )

    for idx, image_path in enumerate(image_paths, start=1):
        if idx < args.start_at:
            continue
        try:
            result = project.single_upload(
                image_path=image_path,
                annotation_path=str(ANNOTATION_FILE),
                batch_name=BATCH_NAME,
                num_retry_uploads=args.num_retries,
            )
        except Exception as exc:
            raise SystemExit(
                f"Upload failed at [{idx}/{len(image_paths)}] {Path(image_path).name}: {exc}"
            ) from exc
        print(f"[{idx}/{len(image_paths)}] {Path(image_path).name}: {result}")


if __name__ == "__main__":
    main()
