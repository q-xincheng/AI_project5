"""
Training script for multimodal sentiment classification
"""
import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import get_data_loaders
from src.model import get_model


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, data_loader, optimizer, criterion, device, gradient_clip=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(data_loader, desc='Training')
    for batch in progress_bar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, image)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro')
    
    return avg_loss, accuracy, f1


def evaluate(model, data_loader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask, image)
            loss = criterion(logits, labels)
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro')
    
    return avg_loss, accuracy, f1, all_predictions, all_labels


def plot_confusion_matrix(labels, predictions, output_path):
    """Plot and save confusion matrix"""
    label_names = ['negative', 'neutral', 'positive']
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train(config_path='configs/config.yaml'):
    """Main training function"""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config['model']['text_model'])
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(config, tokenizer)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = get_model(config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    # 类别顺序：negative=0, neutral=1, positive=2
    class_weights = torch.tensor([1.0, 2.5, 1.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    print("\nStarting training...")
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device,
            config['training']['gradient_clip']
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_predictions, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'config': config
            }, config['model_save_path'])
            print(f"✓ Saved best model (F1: {val_f1:.4f})")
            patience_counter = 0
            
            # Save confusion matrix for best model
            plot_confusion_matrix(
                val_labels, val_predictions,
                os.path.join(config['output_dir'], 'confusion_matrix.png')
            )
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_f1'], label='Train')
    plt.plot(history['val_f1'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'training_history.png'))
    plt.close()
    
    print(f"\n✓ Training completed!")
    print(f"Best validation F1: {best_val_f1:.4f}")
    
    # Final validation with best model
    print("\nFinal evaluation on validation set:")
    checkpoint = torch.load(config['model_save_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_loss, val_acc, val_f1, val_predictions, val_labels = evaluate(
        model, val_loader, criterion, device
    )
    
    print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        val_labels, val_predictions,
        target_names=['negative', 'neutral', 'positive']
    ))
    
    # Save results
    with open(os.path.join(config['output_dir'], 'results.txt'), 'w') as f:
        f.write(f"Experiment Mode: {config.get('experiment_mode', 'multimodal')}\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"Validation F1 Score: {val_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(
            val_labels, val_predictions,
            target_names=['negative', 'neutral', 'positive']
        ))


if __name__ == '__main__':
    train()
