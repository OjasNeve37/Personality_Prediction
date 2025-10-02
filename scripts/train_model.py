import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, get_scheduler
from torch.optim import AdamW  # Import AdamW from torch.optim instead of transformers
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os
import gc

# Fix import paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from scripts.feature_extraction import create_data_loaders

class BERTPersonalityClassifier(nn.Module):
    """BERT-based personality trait predictor"""
    def __init__(self, num_traits=5, dropout=0.3):
        super(BERTPersonalityClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(Config.MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_traits)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        logits = torch.sigmoid(self.classifier(output))
        
        return logits

class PersonalityTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
        
        # Learning rate scheduler
        total_steps = len(train_loader) * Config.NUM_EPOCHS
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self):
        """Train for one epoch with gradient accumulation"""
        self.model.train()
        total_loss = 0
        
        # Clear GPU cache before training
        torch.cuda.empty_cache()
        gc.collect()
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        # Initialize gradient accumulation
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            
            # Calculate loss and normalize by accumulation steps
            loss = self.criterion(outputs, labels) / Config.GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass
            loss.backward()
            
            # Update weights only after accumulation steps
            if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Clear cache periodically
                if (batch_idx + 1) % (Config.GRADIENT_ACCUMULATION_STEPS * 10) == 0:
                    torch.cuda.empty_cache()
            
            total_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
            progress_bar.set_postfix({
                'loss': loss.item() * Config.GRADIENT_ACCUMULATION_STEPS,
                'gpu_mem': f'{torch.cuda.memory_allocated()/1024**3:.2f}GB'
            })
            
            # Detach to free memory
            del input_ids, attention_mask, labels, outputs, loss
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        # Clear cache before validation
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Free memory
                del input_ids, attention_mask, labels, outputs, loss
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self, num_epochs):
        """Full training loop"""
        print(f"\nTraining on {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Batch size: {Config.BATCH_SIZE}")
        print(f"Gradient accumulation steps: {Config.GRADIENT_ACCUMULATION_STEPS}")
        print(f"Effective batch size: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Print GPU memory usage
            if torch.cuda.is_available():
                print(f"\nGPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pt')
                print(f"✓ Best model saved (Val Loss: {val_loss:.4f})")
        
        # Save final model
        self.save_model('final_model.pt')
        
        # Plot training history
        self.plot_training_history()
    
    def save_model(self, filename):
        """Save model checkpoint"""
        filepath = f"{Config.MODEL_DIR}/{filename}"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, filepath)
    
    def plot_training_history(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', marker='o')
        plt.plot(self.val_losses, label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{Config.MODEL_DIR}/training_history.png")
        print(f"\nTraining history plot saved to {Config.MODEL_DIR}/training_history.png")

def main():
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Load data
    print("Loading processed data...")
    df = pd.read_csv(Config.PROCESSED_DATA_PATH)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        df, tokenizer, Config.BATCH_SIZE
    )
    
    # Initialize model
    print("Initializing model...")
    model = BERTPersonalityClassifier(num_traits=Config.NUM_TRAITS)
    
    # Initialize trainer
    trainer = PersonalityTrainer(model, train_loader, val_loader, Config.DEVICE)
    
    # Train model
    trainer.train(Config.NUM_EPOCHS)
    
    print("\n✓ Training complete!")

if __name__ == "__main__":
    main()
