"""
Script to download the Suicide-Watch dataset from Kaggle.

Prerequisites:
1. Install kaggle: pip install kaggle
2. Set up Kaggle API credentials:
   - Go to kaggle.com/account
   - Create new API token (downloads kaggle.json)
   - Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\ (Windows)
   - chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)

Usage:
    python download_kaggle.py
"""

from pathlib import Path


def download_suicide_watch_dataset():
    """Download the Suicide-Watch v13 dataset from Kaggle."""

    # Target directory
    data_dir = Path(__file__).parent.parent / "data" / "suicide_watch"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset to: {data_dir}")

    try:
        # Import kaggle (must be installed)
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi

        # Initialize API
        api = KaggleApi()
        api.authenticate()

        # Dataset info
        dataset = "nikhileswarkomati/suicide-watch"

        print(f"Downloading {dataset}...")

        # Download dataset
        api.dataset_download_files(dataset, path=str(data_dir), unzip=True)

        print("Download complete!")

        # List downloaded files
        files = list(data_dir.glob("*.csv"))
        print("\nDownloaded files:")
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.2f} MB)")

        return True

    except ImportError:
        print("\nError: kaggle package not installed.")
        print("Install with: pip install kaggle")
        print("\nAlternatively, download manually:")
        print("1. Go to: https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/versions/13")
        print("2. Click 'Download' button")
        print(f"3. Extract to: {data_dir}")
        return False

    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nManual download instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/versions/13")
        print("2. Click 'Download' button")
        print(f"3. Extract to: {data_dir}")
        return False


def verify_dataset():
    """Verify the dataset is properly downloaded."""
    data_dir = Path(__file__).parent.parent / "data" / "suicide_watch"

    expected_files = ["Suicide_Detection.csv"]  # Main file name may vary
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in dataset directory.")
        return False

    print("\nDataset verification:")
    for f in csv_files:
        print(f"  Found: {f.name}")

        # Quick peek at structure
        try:
            import pandas as pd

            df = pd.read_csv(f, nrows=5)
            print(f"    Columns: {list(df.columns)}")
            print(f"    Sample rows: {len(df)}")
        except Exception as e:
            print(f"    Could not read: {e}")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Suicide-Watch Dataset Downloader")
    print("=" * 60)

    success = download_suicide_watch_dataset()

    if success:
        verify_dataset()

    print("\n" + "=" * 60)
    if success:
        print("Dataset ready for preprocessing!")
        print("Next step: python preprocess_text.py")
    else:
        print("Please download manually and try again.")
    print("=" * 60)
