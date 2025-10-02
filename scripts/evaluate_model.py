import torch
import torch.nn as nn
from transformers import BertTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Fix import paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from scripts.train_model import BERTPersonalityClassifier
from scripts.feature_extraction import create_data_loaders

# Rest of your code...


def load_model(model_path, device):
    """Load trained model"""
    model = BERTPersonalityClassifier(num_traits=Config.NUM_TRAITS)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask)
            
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    predictions = np.array(all_predictions)
    actuals = np.array(all_labels)
    
    return predictions, actuals

def calculate_metrics(predictions, actuals):
    """Calculate evaluation metrics"""
    results = {}
    
    for i, trait in enumerate(Config.TRAITS):
        mse = mean_squared_error(actuals[:, i], predictions[:, i])
        mae = mean_absolute_error(actuals[:, i], predictions[:, i])
        r2 = r2_score(actuals[:, i], predictions[:, i])
        
        results[trait] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': np.sqrt(mse),
            'R2': r2
        }
    
    return results

def plot_predictions(predictions, actuals, trait_idx=0):
    """Plot predictions vs actual values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals[:, trait_idx], predictions[:, trait_idx], alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions vs Actuals: {Config.TRAITS[trait_idx].title()}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{Config.MODEL_DIR}/predictions_{Config.TRAITS[trait_idx]}.png")
    print(f"Plot saved: predictions_{Config.TRAITS[trait_idx]}.png")

def main():
    # Load tokenizer and data
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    df = pd.read_csv(Config.PROCESSED_DATA_PATH)
    
    # Create data loaders
    _, _, test_loader = create_data_loaders(df, tokenizer, Config.BATCH_SIZE)
    
    # Load trained model
    model_path = f"{Config.MODEL_DIR}/best_model.pt"
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, Config.DEVICE)
    
    # Evaluate
    print("Evaluating model...")
    predictions, actuals = evaluate_model(model, test_loader, Config.DEVICE)
    
    # Calculate metrics
    results = calculate_metrics(predictions, actuals)
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    for trait, metrics in results.items():
        print(f"\n{trait.upper()}:")
        print(f"  MSE:  {metrics['MSE']:.4f}")
        print(f"  MAE:  {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  R²:   {metrics['R2']:.4f}")
    
    # Plot for each trait
    for i in range(Config.NUM_TRAITS):
        plot_predictions(predictions, actuals, i)
    
    print("\n✓ Evaluation complete!")

if __name__ == "__main__":
    main()
