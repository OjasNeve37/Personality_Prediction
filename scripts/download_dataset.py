import kagglehub
import shutil
import os
from pathlib import Path

def download_mbti_dataset():
    """Download MBTI dataset using kagglehub"""
    print("Downloading MBTI dataset from Kaggle...")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("datasnaek/mbti-type")
        print(f"✓ Dataset downloaded to: {path}")
        
        # Copy to your project's data/raw folder
        target_dir = Path("data/raw")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Find and copy the CSV file
        source_path = Path(path)
        csv_files = list(source_path.glob("*.csv"))
        
        if not csv_files:
            print("⚠ Warning: No CSV files found in downloaded dataset")
            print(f"Contents of {source_path}:")
            for item in source_path.iterdir():
                print(f"  - {item.name}")
            return None
        
        for file in csv_files:
            target_file = target_dir / file.name
            shutil.copy2(file, target_file)
            print(f"✓ Copied {file.name} to {target_file}")
        
        print("\n✓ Dataset ready for use!")
        return str(target_dir)
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nManual download instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/datasnaek/mbti-type")
        print("2. Download the dataset")
        print("3. Extract mbti_1.csv to: data/raw/")
        return None

if __name__ == "__main__":
    download_mbti_dataset()
