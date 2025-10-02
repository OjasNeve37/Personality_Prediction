# Personality Prediction

This project uses machine learning to predict personality traits from resumes and text.

## Large Files

Some files are too large for GitHub and are not included in this repository. You'll need to download them separately:

### Data Files
- `data/processed/processed_data.csv` (148.61 MB)
- `data/raw/mbti_1.csv` (59.94 MB)

### Model Files
- `models/saved_models/final_model.pt` (1.25 GB)
- `models/saved_models/best_model.pt`

You can download these files from [Google Drive/Hugging Face Hub] (add your preferred sharing link).

## Setup Instructions

1. Clone this repository
2. Download the large files from the link above
3. Place the files in their respective directories:
   ```
   data/
   ├── processed/
   │   └── processed_data.csv
   ├── raw/
   │   └── mbti_1.csv
   └── models/
       └── saved_models/
           ├── final_model.pt
           └── best_model.pt
   ```
4. Create a virtual environment and install dependencies:
   ```bash
   python -m venv cv_env
   source cv_env/bin/activate  # On Windows: .\cv_env\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

[Add usage instructions here]