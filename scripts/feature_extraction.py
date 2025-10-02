import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
sys.path.append('..')
from config import Config

class PersonalityDataset(Dataset):
    """Custom dataset for personality prediction"""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(labels)
        }

def create_data_loaders(df, tokenizer, batch_size=16):
    """Create train, validation, and test data loaders"""
    from sklearn.model_selection import train_test_split
    
    # Prepare features and labels
    texts = df['processed_text'].values
    labels = df[Config.TRAITS].values
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, random_state=Config.RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=Config.RANDOM_SEED
    )
    
    # Create datasets
    train_dataset = PersonalityDataset(X_train, y_train, tokenizer, Config.MAX_LENGTH)
    val_dataset = PersonalityDataset(X_val, y_val, tokenizer, Config.MAX_LENGTH)
    test_dataset = PersonalityDataset(X_test, y_test, tokenizer, Config.MAX_LENGTH)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    
    print("Loading processed data...")
    df = pd.read_csv(Config.PROCESSED_DATA_PATH)
    
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        df, tokenizer, Config.BATCH_SIZE
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
