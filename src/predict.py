"""
Prediction script for test data without labels
"""
import os
import yaml
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer

from src.dataset import get_test_loader
from src.model import get_model


def predict(config_path='configs/config.yaml', model_path=None):
    """
    Generate predictions for test data
    
    Args:
        config_path: Path to configuration file
        model_path: Path to trained model (if None, uses config default)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if model_path is None:
        model_path = config['model_save_path']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get prediction config
    temperature = config.get('prediction', {}).get('temperature', 1.0)
    positive_threshold = config.get('prediction', {}).get('positive_threshold', 0.5)
    
    if temperature != 1.0:
        print(f"Using temperature scaling: T={temperature}")
    if positive_threshold != 0.5:
        print(f"Using positive threshold adjustment: threshold={positive_threshold}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config['model']['text_model'])
    
    # Load test data
    print("Loading test data...")
    test_loader = get_test_loader(config, tokenizer)
    print(f"Test batches: {len(test_loader)}")
    
    # Load model
    print("Loading model...")
    model = get_model(config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Generate predictions
    print("Generating predictions...")
    all_guids = []
    all_predictions = []
    
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask, image)
            
            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
            
            # Get probabilities
            probs = F.softmax(logits, dim=1)
            
            # Apply threshold adjustment for positive class
            if positive_threshold != 0.5:
                adjusted_probs = probs.clone()
                scale_factor = 0.5 / positive_threshold
                adjusted_probs[:, 2] = adjusted_probs[:, 2] * scale_factor
                adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=1, keepdim=True)
                predictions = torch.argmax(adjusted_probs, dim=1).cpu().numpy()
            else:
                predictions = torch.argmax(probs, dim=1).cpu().numpy()
            
            # Collect results
            all_guids.extend(batch['guid'])
            all_predictions.extend([label_map[p] for p in predictions])
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'guid': all_guids,
        'tag': all_predictions
    })
    
    # Save predictions
    output_path = config['predictions_file']
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")
    
    # Print distribution
    print("\nPrediction distribution:")
    print(results_df['tag'].value_counts())
    
    return results_df


if __name__ == '__main__':
    predict()
