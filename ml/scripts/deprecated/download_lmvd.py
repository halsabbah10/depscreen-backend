"""
Script to download the LMVD (Large-scale Multimodal Vlog Dataset) for depression detection.

LMVD Dataset Information:
- Paper: "A Large-Scale Multimodal Vlog Dataset for Depression Detection in the Wild"
- GitHub: https://github.com/LMVD/LMVD
- Dataset hosting: Figshare

This script provides instructions and helpers for downloading the dataset.

Usage:
    python download_lmvd.py
"""

from pathlib import Path


def print_download_instructions():
    """Print instructions for downloading the LMVD dataset."""

    data_dir = Path(__file__).parent.parent / "data" / "lmvd"

    print("=" * 70)
    print("LMVD Dataset Download Instructions")
    print("=" * 70)

    print(f"""
The LMVD (Large-scale Multimodal Vlog Dataset) is a dataset for depression
detection from vlog videos. Due to its size and licensing, it requires
manual download.

DOWNLOAD STEPS:
==============

1. Visit the official LMVD GitHub repository:
   https://github.com/LMVD/LMVD

2. Follow the dataset access instructions in the README.
   You may need to:
   - Fill out a data use agreement
   - Request access via email
   - Download from Figshare or provided links

3. The dataset typically includes:
   - Video files (MP4 format)
   - Label files (CSV/JSON with depression/control labels)
   - Metadata (video IDs, participant info)

4. Download and extract to:
   {data_dir}

   Expected structure:
   {data_dir}/
   ├── videos/
   │   ├── depressed/
   │   │   ├── video_001.mp4
   │   │   └── ...
   │   └── control/
   │       ├── video_101.mp4
   │       └── ...
   ├── labels.csv
   └── metadata.json

ALTERNATIVE STRUCTURE:
=====================
If the dataset uses a different structure (e.g., all videos in one folder
with a separate labels file), that's fine too. The preprocessing script
can handle various formats.

DATASET SIZE:
============
- The full dataset is several GB in size
- Consider starting with a subset for initial testing
- Ensure you have sufficient disk space

AFTER DOWNLOAD:
==============
Run the frame extraction script:
    python extract_frames.py

This will:
- Extract frames from each video at 1 FPS
- Organize frames by label (depressed/control)
- Create a processed dataset ready for training
""")

    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "videos" / "depressed").mkdir(parents=True, exist_ok=True)
    (data_dir / "videos" / "control").mkdir(parents=True, exist_ok=True)

    print(f"""
DIRECTORIES CREATED:
===================
Created directory structure at: {data_dir}

Place your downloaded videos in the appropriate folders:
- Depressed subjects: {data_dir}/videos/depressed/
- Control subjects: {data_dir}/videos/control/

If you have a labels file instead, place it at:
{data_dir}/labels.csv

The labels file should have columns like:
- video_id (or filename)
- label (depressed/control or 1/0)
""")

    print("=" * 70)


def verify_dataset():
    """Check if dataset files exist."""
    data_dir = Path(__file__).parent.parent / "data" / "lmvd"

    print("\nVerifying dataset...")

    # Check for videos
    video_dirs = [data_dir / "videos" / "depressed", data_dir / "videos" / "control"]

    video_count = 0
    for vdir in video_dirs:
        if vdir.exists():
            videos = list(vdir.glob("*.mp4")) + list(vdir.glob("*.avi"))
            video_count += len(videos)
            print(f"  {vdir.name}/: {len(videos)} videos")

    # Check for labels file
    label_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.json"))
    if label_files:
        print(f"  Label files found: {[f.name for f in label_files]}")

    if video_count > 0:
        print(f"\nTotal videos found: {video_count}")
        print("\nDataset appears to be downloaded!")
        print("Next step: python extract_frames.py")
        return True
    else:
        print("\nNo videos found yet. Please download the dataset.")
        return False


def create_sample_labels():
    """Create a sample labels file template."""
    data_dir = Path(__file__).parent.parent / "data" / "lmvd"
    labels_path = data_dir / "labels_template.csv"

    template = """video_id,label,split
video_001,depressed,train
video_002,depressed,train
video_003,control,train
video_004,control,val
video_005,depressed,test
"""

    with open(labels_path, "w") as f:
        f.write(template)

    print(f"\nCreated labels template at: {labels_path}")
    print("Edit this file to match your actual video files and labels.")


if __name__ == "__main__":
    print_download_instructions()

    # Create template
    create_sample_labels()

    # Verify
    verify_dataset()
