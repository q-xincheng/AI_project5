"""
Training script for multimodal sentiment classification
"""
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import get_data_loaders
from src.model import get_model
from src.loss import get_criterion
from src.utils import apply_prediction_adjustments


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


def evaluate(model, data_loader, criterion, device, config=None, save_neutral_errors=False):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_guids = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            labels = batch['label'].to(device)
            guids = batch.get('guid', [])
            
            # Forward pass
            logits = model(input_ids, attention_mask, image)
            loss = criterion(logits, labels)
            
            # Apply prediction adjustments (temperature scaling, threshold adjustment)
            predictions, probs = apply_prediction_adjustments(logits, config)
            predictions = predictions.cpu().numpy()
            
            # Statistics
            total_loss += loss.item()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            all_guids.extend(guids)
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro')
    
    # Calculate per-class recall
    recall_per_class = recall_score(all_labels, all_predictions, average=None, labels=[0, 1, 2])
    negative_recall = recall_per_class[0]
    neutral_recall = recall_per_class[1]
    positive_recall = recall_per_class[2]
    
    # Track neutral errors if requested
    neutral_errors = []
    if save_neutral_errors:
        for i, (true_label, pred_label, guid) in enumerate(zip(all_labels, all_predictions, all_guids)):
            if true_label == 1 and pred_label != 1:  # Neutral misclassified
                pred_class_name = ['negative', 'neutral', 'positive'][pred_label]
                neutral_errors.append({
                    'guid': guid,
                    'true_label': 'neutral',
                    'predicted_label': pred_class_name,
                    'probabilities': all_probs[i]
                })
    
    return avg_loss, accuracy, f1, all_predictions, all_labels, {
        'negative_recall': negative_recall,
        'neutral_recall': neutral_recall,
        'positive_recall': positive_recall,
        'neutral_errors': neutral_errors
    }


def plot_confusion_matrix(labels, predictions, output_path, recall_per_class=None):
    """Plot and save confusion matrix with recall annotations"""
    label_names = ['negative', 'neutral', 'positive']
    cm = confusion_matrix(labels, predictions)
    
    # Calculate recall per class if not provided
    if recall_per_class is None:
        recall_per_class = recall_score(labels, predictions, average=None, labels=[0, 1, 2])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Add recall annotations
    title = 'Confusion Matrix\n'
    title += f'Recall - Negative: {recall_per_class[0]:.3f}, '
    title += f'Neutral: {recall_per_class[1]:.3f}, '
    title += f'Positive: {recall_per_class[2]:.3f}'
    plt.title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_neutral_errors(neutral_errors, output_path):
    """Save neutral classification errors to file"""
    if not neutral_errors:
        return
    
    with open(output_path, 'w') as f:
        f.write("Neutral Classification Errors\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total neutral samples misclassified: {len(neutral_errors)}\n\n")
        
        # Group by predicted class
        by_predicted = {'negative': [], 'positive': []}
        for error in neutral_errors:
            if error['predicted_label'] in by_predicted:
                by_predicted[error['predicted_label']].append(error)
        
        for pred_class in ['negative', 'positive']:
            errors = by_predicted[pred_class]
            f.write(f"\nNeutral misclassified as {pred_class.upper()}: {len(errors)} samples\n")
            f.write("-" * 80 + "\n")
            for error in errors[:50]:  # Limit to first 50 per class
                probs = error['probabilities']
                f.write(f"GUID: {error['guid']}\n")
                f.write(f"  Probabilities: neg={probs[0]:.4f}, neu={probs[1]:.4f}, pos={probs[2]:.4f}\n")
                f.write("\n")
            
            if len(errors) > 50:
                f.write(f"... and {len(errors) - 50} more\n")


def save_detailed_metrics(metrics_dict, output_path):
    """Save detailed metrics to file"""
    with open(output_path, 'w') as f:
        f.write("Detailed Training Metrics\n")
        f.write("=" * 80 + "\n\n")
        
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value}\n")
            elif isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")


def train_single_run(config, seed=None, fold_idx=None, k_folds=None):
    """
    Single training run (used by both regular training and multi-seed/k-fold training)
    
    Args:
        config: Configuration dictionary
        seed: Optional seed override
        fold_idx: Optional fold index for k-fold CV
        k_folds: Optional number of folds for k-fold CV
    
    Returns:
        Dictionary with training results
    """
    # Set seed
    if seed is not None:
        set_seed(seed)
    else:
        set_seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['model']['text_model'])
    
    # Load data
    train_loader, val_loader = get_data_loaders(config, tokenizer, fold_idx=fold_idx, k_folds=k_folds)
    
    # Create model
    model = get_model(config)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = get_criterion(config, device)
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
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'neg_recall': [], 'neu_recall': [], 'pos_recall': []
    }
    
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device,
            config['training']['gradient_clip']
        )
        
        # Validate
        save_errors = config.get('monitoring', {}).get('save_neutral_errors', False)
        val_loss, val_acc, val_f1, val_predictions, val_labels, extra_metrics = evaluate(
            model, val_loader, criterion, device, config=config, save_neutral_errors=save_errors
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['neg_recall'].append(extra_metrics['negative_recall'])
        history['neu_recall'].append(extra_metrics['neutral_recall'])
        history['pos_recall'].append(extra_metrics['positive_recall'])
        
        if config.get('monitoring', {}).get('verbose_metrics', True):
            print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro-F1: {val_f1:.4f}")
            print(f"Recall - Neg: {extra_metrics['negative_recall']:.4f}, "
                  f"Neu: {extra_metrics['neutral_recall']:.4f}, "
                  f"Pos: {extra_metrics['positive_recall']:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_predictions = val_predictions
            best_labels = val_labels
            best_metrics = extra_metrics
            
            # Save model checkpoint
            model_save_path = config['model_save_path']
            if fold_idx is not None:
                # For k-fold, save with fold index
                base, ext = os.path.splitext(model_save_path)
                model_save_path = f"{base}_fold{fold_idx}{ext}"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'config': config
            }, model_save_path)
            
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            if config.get('monitoring', {}).get('verbose_metrics', True):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Return results
    return {
        'best_val_f1': best_val_f1,
        'best_val_acc': accuracy_score(best_labels, best_predictions),
        'best_epoch': best_epoch,
        'history': history,
        'predictions': best_predictions,
        'labels': best_labels,
        'metrics': best_metrics
    }


def train(config_path='configs/config.yaml'):
    """Main training function"""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Count parameters (just for info)
    print("\nLoading tokenizer and model for parameter count...")
    tokenizer = BertTokenizer.from_pretrained(config['model']['text_model'])
    model = get_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    del model  # Free memory
    
    # Check if k-fold cross-validation is enabled
    k_folds = config.get('stability', {}).get('k_fold')
    multi_seed = config.get('stability', {}).get('multi_seed')
    
    if k_folds is not None and k_folds > 1:
        # K-fold cross-validation
        print(f"\n{'='*80}")
        print(f"Running {k_folds}-Fold Stratified Cross-Validation")
        print(f"{'='*80}\n")
        
        fold_results = []
        for fold_idx in range(k_folds):
            print(f"\n{'='*80}")
            print(f"Training Fold {fold_idx + 1}/{k_folds}")
            print(f"{'='*80}\n")
            
            result = train_single_run(config, fold_idx=fold_idx, k_folds=k_folds)
            fold_results.append(result)
            
            print(f"\nFold {fold_idx + 1} Results:")
            print(f"  Best Val F1: {result['best_val_f1']:.4f}")
            print(f"  Best Val Acc: {result['best_val_acc']:.4f}")
            print(f"  Negative Recall: {result['metrics']['negative_recall']:.4f}")
            print(f"  Neutral Recall: {result['metrics']['neutral_recall']:.4f}")
            print(f"  Positive Recall: {result['metrics']['positive_recall']:.4f}")
        
        # Calculate average metrics
        avg_f1 = np.mean([r['best_val_f1'] for r in fold_results])
        avg_acc = np.mean([r['best_val_acc'] for r in fold_results])
        avg_neg_recall = np.mean([r['metrics']['negative_recall'] for r in fold_results])
        avg_neu_recall = np.mean([r['metrics']['neutral_recall'] for r in fold_results])
        avg_pos_recall = np.mean([r['metrics']['positive_recall'] for r in fold_results])
        
        print(f"\n{'='*80}")
        print("K-Fold Cross-Validation Summary")
        print(f"{'='*80}")
        print(f"Average Validation F1: {avg_f1:.4f} ± {np.std([r['best_val_f1'] for r in fold_results]):.4f}")
        print(f"Average Validation Acc: {avg_acc:.4f} ± {np.std([r['best_val_acc'] for r in fold_results]):.4f}")
        print(f"Average Negative Recall: {avg_neg_recall:.4f}")
        print(f"Average Neutral Recall: {avg_neu_recall:.4f}")
        print(f"Average Positive Recall: {avg_pos_recall:.4f}")
        
        # Save summary
        with open(os.path.join(config['output_dir'], 'kfold_results.txt'), 'w') as f:
            f.write("K-Fold Cross-Validation Results\n")
            f.write("=" * 80 + "\n\n")
            for i, result in enumerate(fold_results):
                f.write(f"Fold {i + 1}:\n")
                f.write(f"  Val F1: {result['best_val_f1']:.4f}\n")
                f.write(f"  Val Acc: {result['best_val_acc']:.4f}\n")
                f.write(f"  Negative Recall: {result['metrics']['negative_recall']:.4f}\n")
                f.write(f"  Neutral Recall: {result['metrics']['neutral_recall']:.4f}\n")
                f.write(f"  Positive Recall: {result['metrics']['positive_recall']:.4f}\n\n")
            
            f.write("\nAverage Metrics:\n")
            f.write(f"  Val F1: {avg_f1:.4f} ± {np.std([r['best_val_f1'] for r in fold_results]):.4f}\n")
            f.write(f"  Val Acc: {avg_acc:.4f} ± {np.std([r['best_val_acc'] for r in fold_results]):.4f}\n")
            f.write(f"  Negative Recall: {avg_neg_recall:.4f}\n")
            f.write(f"  Neutral Recall: {avg_neu_recall:.4f}\n")
            f.write(f"  Positive Recall: {avg_pos_recall:.4f}\n")
        
        return
    
    elif multi_seed is not None and isinstance(multi_seed, list) and len(multi_seed) > 1:
        # Multi-seed training
        print(f"\n{'='*80}")
        print(f"Running Multi-Seed Training with seeds: {multi_seed}")
        print(f"{'='*80}\n")
        
        seed_results = []
        for seed in multi_seed:
            print(f"\n{'='*80}")
            print(f"Training with seed {seed}")
            print(f"{'='*80}\n")
            
            # Update config with current seed
            config['seed'] = seed
            result = train_single_run(config, seed=seed)
            seed_results.append(result)
            
            print(f"\nSeed {seed} Results:")
            print(f"  Best Val F1: {result['best_val_f1']:.4f}")
            print(f"  Best Val Acc: {result['best_val_acc']:.4f}")
            print(f"  Negative Recall: {result['metrics']['negative_recall']:.4f}")
            print(f"  Neutral Recall: {result['metrics']['neutral_recall']:.4f}")
            print(f"  Positive Recall: {result['metrics']['positive_recall']:.4f}")
        
        # Calculate average metrics
        avg_f1 = np.mean([r['best_val_f1'] for r in seed_results])
        avg_acc = np.mean([r['best_val_acc'] for r in seed_results])
        avg_neg_recall = np.mean([r['metrics']['negative_recall'] for r in seed_results])
        avg_neu_recall = np.mean([r['metrics']['neutral_recall'] for r in seed_results])
        avg_pos_recall = np.mean([r['metrics']['positive_recall'] for r in seed_results])
        
        print(f"\n{'='*80}")
        print("Multi-Seed Training Summary")
        print(f"{'='*80}")
        print(f"Average Validation F1: {avg_f1:.4f} ± {np.std([r['best_val_f1'] for r in seed_results]):.4f}")
        print(f"Average Validation Acc: {avg_acc:.4f} ± {np.std([r['best_val_acc'] for r in seed_results]):.4f}")
        print(f"Average Negative Recall: {avg_neg_recall:.4f}")
        print(f"Average Neutral Recall: {avg_neu_recall:.4f}")
        print(f"Average Positive Recall: {avg_pos_recall:.4f}")
        
        # Save summary
        with open(os.path.join(config['output_dir'], 'multiseed_results.txt'), 'w') as f:
            f.write("Multi-Seed Training Results\n")
            f.write("=" * 80 + "\n\n")
            for seed, result in zip(multi_seed, seed_results):
                f.write(f"Seed {seed}:\n")
                f.write(f"  Val F1: {result['best_val_f1']:.4f}\n")
                f.write(f"  Val Acc: {result['best_val_acc']:.4f}\n")
                f.write(f"  Negative Recall: {result['metrics']['negative_recall']:.4f}\n")
                f.write(f"  Neutral Recall: {result['metrics']['neutral_recall']:.4f}\n")
                f.write(f"  Positive Recall: {result['metrics']['positive_recall']:.4f}\n\n")
            
            f.write("\nAverage Metrics:\n")
            f.write(f"  Val F1: {avg_f1:.4f} ± {np.std([r['best_val_f1'] for r in seed_results]):.4f}\n")
            f.write(f"  Val Acc: {avg_acc:.4f} ± {np.std([r['best_val_acc'] for r in seed_results]):.4f}\n")
            f.write(f"  Negative Recall: {avg_neg_recall:.4f}\n")
            f.write(f"  Neutral Recall: {avg_neu_recall:.4f}\n")
            f.write(f"  Positive Recall: {avg_pos_recall:.4f}\n")
        
        return
    
    else:
        # Regular single training run
        print(f"\n{'='*80}")
        print("Starting Training")
        print(f"{'='*80}\n")
        
        result = train_single_run(config)
        
        # Plot training history
        history = result['history']
        plt.figure(figsize=(15, 10))
        
        # Loss
        plt.subplot(2, 3, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss')
        plt.grid(True, alpha=0.3)
        
        # Accuracy
        plt.subplot(2, 3, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy')
        plt.grid(True, alpha=0.3)
        
        # F1 Score
        plt.subplot(2, 3, 3)
        plt.plot(history['train_f1'], label='Train')
        plt.plot(history['val_f1'], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Macro F1 Score')
        plt.legend()
        plt.title('Macro F1 Score')
        plt.grid(True, alpha=0.3)
        
        # Per-class recall
        plt.subplot(2, 3, 4)
        plt.plot(history['neg_recall'], label='Negative', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.title('Negative Recall')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        plt.plot(history['neu_recall'], label='Neutral', marker='s', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.title('Neutral Recall')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 6)
        plt.plot(history['pos_recall'], label='Positive', marker='^', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.title('Positive Recall')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config['output_dir'], 'training_history.png'), dpi=150)
        plt.close()
        
        # Plot confusion matrix
        recall_per_class = [
            result['metrics']['negative_recall'],
            result['metrics']['neutral_recall'],
            result['metrics']['positive_recall']
        ]
        plot_confusion_matrix(
            result['labels'], result['predictions'],
            os.path.join(config['output_dir'], 'confusion_matrix.png'),
            recall_per_class=recall_per_class
        )
        
        # Save neutral errors
        if config.get('monitoring', {}).get('save_neutral_errors', False):
            neutral_errors = result['metrics'].get('neutral_errors', [])
            if neutral_errors:
                save_neutral_errors(
                    neutral_errors,
                    config.get('neutral_errors_file', 'outputs/neutral_errors.txt')
                )
        
        # Print final results
        print(f"\n{'='*80}")
        print("Training Completed!")
        print(f"{'='*80}")
        print(f"Best validation Macro-F1: {result['best_val_f1']:.4f}")
        print(f"Best validation Accuracy: {result['best_val_acc']:.4f}")
        print(f"Negative Recall: {result['metrics']['negative_recall']:.4f}")
        print(f"Neutral Recall: {result['metrics']['neutral_recall']:.4f}")
        print(f"Positive Recall: {result['metrics']['positive_recall']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(
            result['labels'], result['predictions'],
            target_names=['negative', 'neutral', 'positive'],
            digits=4
        ))
        
        # Save detailed results
        with open(os.path.join(config['output_dir'], 'results.txt'), 'w') as f:
            f.write(f"Experiment Mode: {config.get('experiment_mode', 'multimodal')}\n")
            f.write(f"Validation Accuracy: {result['best_val_acc']:.4f}\n")
            f.write(f"Validation Macro-F1 Score: {result['best_val_f1']:.4f}\n")
            f.write(f"Negative Recall: {result['metrics']['negative_recall']:.4f}\n")
            f.write(f"Neutral Recall: {result['metrics']['neutral_recall']:.4f}\n")
            f.write(f"Positive Recall: {result['metrics']['positive_recall']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(
                result['labels'], result['predictions'],
                target_names=['negative', 'neutral', 'positive'],
                digits=4
            ))
        
        # Save detailed metrics
        if config.get('monitoring', {}).get('verbose_metrics', True):
            metrics_dict = {
                'best_epoch': result['best_epoch'],
                'best_val_f1': result['best_val_f1'],
                'best_val_acc': result['best_val_acc'],
                'negative_recall': result['metrics']['negative_recall'],
                'neutral_recall': result['metrics']['neutral_recall'],
                'positive_recall': result['metrics']['positive_recall'],
                'total_neutral_errors': len(result['metrics'].get('neutral_errors', []))
            }
            save_detailed_metrics(
                metrics_dict,
                config.get('metrics_file', 'outputs/detailed_metrics.txt')
            )


if __name__ == '__main__':
    train()
