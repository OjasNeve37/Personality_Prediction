"""
Complete pipeline test
"""
import os
import sys
from scripts.data_preprocessing import DataPreprocessor
from scripts.train_model import BERTPersonalityClassifier
from transformers import BertTokenizer
from config import Config

def test_pipeline():
    print("="*70)
    print("TESTING COMPLETE PIPELINE")
    print("="*70)
    
    # Test 1: Configuration
    print("\n✓ Test 1: Configuration")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Model: {Config.MODEL_NAME}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    
    # Test 2: Data Preprocessing
    print("\n✓ Test 2: Data Preprocessing")
    preprocessor = DataPreprocessor()
    test_text = "This is a test sentence with special characters!!!"
    cleaned = preprocessor.clean_text(test_text)
    print(f"  Original: {test_text}")
    print(f"  Cleaned: {cleaned}")
    
    # Test 3: Tokenizer
    print("\n✓ Test 3: Tokenizer")
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    tokens = tokenizer.tokenize("Hello world")
    print(f"  Tokens: {tokens}")
    
    # Test 4: Model Architecture
    print("\n✓ Test 4: Model Architecture")
    model = BERTPersonalityClassifier(num_traits=5)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test 5: Check if processed data exists
    print("\n✓ Test 5: Data Files")
    if os.path.exists(Config.PROCESSED_DATA_PATH):
        print(f"  Processed data found: {Config.PROCESSED_DATA_PATH}")
    else:
        print(f"  ⚠ Processed data not found. Run data_preprocessing.py first")
    
    # Test 6: Check raw data
    print("\n✓ Test 6: Raw Dataset")
    if os.path.exists(Config.DATASET_PATH):
        print(f"  Raw dataset found: {Config.DATASET_PATH}")
    else:
        print(f"  ❌ Raw dataset not found at: {Config.DATASET_PATH}")
        print(f"  Run: python scripts/download_dataset.py")
    
    # Test 7: Check if trained model exists
    print("\n✓ Test 7: Trained Model")
    model_path = f"{Config.MODEL_DIR}/best_model.pt"
    if os.path.exists(model_path):
        print(f"  Trained model found: {model_path}")
    else:
        print(f"  ⚠ Trained model not found. Run train_model.py first")
    
    print("\n" + "="*70)
    print("PIPELINE TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    test_pipeline()
