import os
import torch

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
    
    # Dataset
    DATASET_PATH = os.path.join(RAW_DATA_DIR, 'mbti_1.csv')
    PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'processed_data.csv')
    
    # Model Parameters
    MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 256  # REDUCED from 512 to save memory
    BATCH_SIZE = 4    # REDUCED from 16 to 4
    GRADIENT_ACCUMULATION_STEPS = 4  # NEW: Simulates batch_size of 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 2
    NUM_TRAITS = 5
    
    # Training
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    RANDOM_SEED = 42
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Big Five Traits
    TRAITS = ['openness', 'conscientiousness', 'extraversion', 
              'agreeableness', 'neuroticism']
    
    # Resume parsing
    RESUME_FORMATS = ['.pdf', '.docx', '.doc']
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)

# Create directories
Config.create_directories()
