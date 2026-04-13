"""
Frame extraction script for LMVD video dataset.

Extracts frames from videos for training the image classifier.

Usage:
    python extract_frames.py [options]

Options:
    --fps: Frames per second to extract (default: 1)
    --max-frames: Maximum frames per video (default: 100)
    --resize: Resize frames to this size (default: 256)
"""

import argparse
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    fps: float = 1.0,
    max_frames: int = 100,
    resize: int = 256
) -> int:
    """
    Extract frames from a single video.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        fps: Target frames per second
        max_frames: Maximum number of frames to extract
        resize: Resize frames to this size (square)

    Returns:
        Number of frames extracted
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return 0

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0 or total_frames <= 0:
        logger.warning(f"Invalid video properties: {video_path}")
        cap.release()
        return 0

    # Calculate frame interval
    frame_interval = int(video_fps / fps)
    if frame_interval < 1:
        frame_interval = 1

    # Create output directory
    video_name = video_path.stem
    video_output_dir = output_dir / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0 and extracted_count < max_frames:
            # Resize frame
            if resize:
                # Center crop to square, then resize
                h, w = frame.shape[:2]
                min_dim = min(h, w)
                start_x = (w - min_dim) // 2
                start_y = (h - min_dim) // 2
                frame = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]
                frame = cv2.resize(frame, (resize, resize))

            # Save frame
            frame_path = video_output_dir / f"frame_{extracted_count:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            extracted_count += 1

        frame_count += 1

    cap.release()
    return extracted_count


def process_labeled_structure(
    data_dir: Path,
    output_dir: Path,
    fps: float,
    max_frames: int,
    resize: int
) -> dict:
    """Process videos organized by label folders."""
    stats = {"depressed": 0, "control": 0, "total_frames": 0}

    for label in ["depressed", "control"]:
        label_dir = data_dir / "videos" / label
        if not label_dir.exists():
            logger.warning(f"Label directory not found: {label_dir}")
            continue

        output_label_dir = output_dir / label
        output_label_dir.mkdir(parents=True, exist_ok=True)

        videos = list(label_dir.glob("*.mp4")) + list(label_dir.glob("*.avi"))
        logger.info(f"Processing {len(videos)} {label} videos...")

        for video_path in tqdm(videos, desc=f"Extracting {label}"):
            frames = extract_frames_from_video(
                video_path,
                output_label_dir,
                fps=fps,
                max_frames=max_frames,
                resize=resize
            )
            stats[label] += 1
            stats["total_frames"] += frames

    return stats


def process_with_labels_file(
    data_dir: Path,
    output_dir: Path,
    labels_file: Path,
    fps: float,
    max_frames: int,
    resize: int
) -> dict:
    """Process videos using a labels file."""
    import pandas as pd

    # Load labels
    if labels_file.suffix == '.csv':
        labels_df = pd.read_csv(labels_file)
    else:
        labels_df = pd.read_json(labels_file)

    # Detect column names
    video_col = None
    label_col = None

    for col in labels_df.columns:
        if 'video' in col.lower() or 'id' in col.lower() or 'file' in col.lower():
            video_col = col
        if 'label' in col.lower() or 'class' in col.lower():
            label_col = col

    if not video_col or not label_col:
        raise ValueError(f"Could not detect video and label columns in {labels_file}")

    logger.info(f"Using columns: video={video_col}, label={label_col}")

    stats = {"depressed": 0, "control": 0, "total_frames": 0}

    # Find video directory
    video_dirs = [
        data_dir / "videos",
        data_dir / "raw_videos",
        data_dir
    ]
    video_dir = None
    for vd in video_dirs:
        if vd.exists():
            video_dir = vd
            break

    if not video_dir:
        raise ValueError("Could not find video directory")

    logger.info(f"Looking for videos in: {video_dir}")

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing"):
        video_id = str(row[video_col])
        label = str(row[label_col]).lower()

        # Normalize label
        if 'depress' in label or label == '1':
            label = 'depressed'
        else:
            label = 'control'

        # Find video file
        video_path = None
        for ext in ['.mp4', '.avi', '']:
            candidate = video_dir / f"{video_id}{ext}"
            if candidate.exists():
                video_path = candidate
                break

        if not video_path:
            logger.warning(f"Video not found: {video_id}")
            continue

        # Extract frames
        output_label_dir = output_dir / label
        output_label_dir.mkdir(parents=True, exist_ok=True)

        frames = extract_frames_from_video(
            video_path,
            output_label_dir,
            fps=fps,
            max_frames=max_frames,
            resize=resize
        )

        stats[label] += 1
        stats["total_frames"] += frames

    return stats


def create_splits(output_dir: Path, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Create train/val/test splits from extracted frames."""
    import random

    splits = {"train": [], "val": [], "test": []}

    for label in ["depressed", "control"]:
        label_dir = output_dir / label
        if not label_dir.exists():
            continue

        # Get all video directories
        video_dirs = [d for d in label_dir.iterdir() if d.is_dir()]
        random.shuffle(video_dirs)

        # Split
        n = len(video_dirs)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        for i, vdir in enumerate(video_dirs):
            if i < train_end:
                split = "train"
            elif i < val_end:
                split = "val"
            else:
                split = "test"

            # Get frame paths
            frames = list(vdir.glob("*.jpg"))
            for frame in frames:
                splits[split].append({
                    "path": str(frame.relative_to(output_dir)),
                    "label": label,
                    "video": vdir.name
                })

    # Save splits
    for split_name, items in splits.items():
        split_file = output_dir / f"{split_name}.json"
        with open(split_file, 'w') as f:
            json.dump(items, f, indent=2)
        logger.info(f"Saved {split_name} split: {len(items)} frames")

    # Save combined metadata
    metadata = {
        "train_frames": len(splits["train"]),
        "val_frames": len(splits["val"]),
        "test_frames": len(splits["test"]),
        "total_frames": sum(len(s) for s in splits.values())
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return splits


def main():
    parser = argparse.ArgumentParser(description='Extract frames from LMVD videos')
    parser.add_argument('--fps', type=float, default=1.0, help='Frames per second')
    parser.add_argument('--max-frames', type=int, default=100, help='Max frames per video')
    parser.add_argument('--resize', type=int, default=256, help='Resize frames to this size')
    parser.add_argument('--data-dir', type=str, default=None, help='Data directory')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "data" / "lmvd"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "data" / "lmvd" / "frames"

    print("=" * 60)
    print("Frame Extraction from LMVD Videos")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"FPS: {args.fps}")
    print(f"Max frames per video: {args.max_frames}")
    print(f"Resize: {args.resize}x{args.resize}")

    # Check for labels file
    labels_file = None
    for f in data_dir.glob("labels*.csv"):
        labels_file = f
        break
    for f in data_dir.glob("labels*.json"):
        labels_file = f
        break

    # Process videos
    if labels_file and labels_file.name != "labels_template.csv":
        logger.info(f"Using labels file: {labels_file}")
        stats = process_with_labels_file(
            data_dir, output_dir, labels_file,
            args.fps, args.max_frames, args.resize
        )
    else:
        logger.info("Using folder-based label structure")
        stats = process_labeled_structure(
            data_dir, output_dir,
            args.fps, args.max_frames, args.resize
        )

    print("\n" + "=" * 60)
    print("Extraction Summary")
    print("=" * 60)
    print(f"Depressed videos: {stats['depressed']}")
    print(f"Control videos: {stats['control']}")
    print(f"Total frames extracted: {stats['total_frames']}")

    # Create splits
    if stats['total_frames'] > 0:
        print("\nCreating train/val/test splits...")
        create_splits(output_dir)

        print("\n" + "=" * 60)
        print("Frame extraction complete!")
        print("=" * 60)
        print(f"\nNext step: python train_image_model.py")
    else:
        print("\nNo frames extracted. Please check your video files.")


if __name__ == "__main__":
    main()
