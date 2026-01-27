"""
Data loading utilities for multimodal sentiment classification
"""
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from sklearn.model_selection import StratifiedKFold
from collections import Counter


class MultimodalDataset(Dataset):
    """Dataset for multimodal (text + image) sentiment classification"""
    
    def __init__(self, data_file, data_dir, tokenizer, max_length=128, 
                 image_size=224, mode='train', transform=None):
        """
        Args:
            data_file: Path to CSV file with guid and tag columns
            data_dir: Directory containing image and text files
            tokenizer: BERT tokenizer for text processing
            max_length: Maximum sequence length for text
            image_size: Size to resize images
            mode: 'train' or 'test' (test has null labels)
            transform: Optional image transformation
        """
        self.data = pd.read_csv(data_file)
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        
        # Label mapping
        self.label_map = {
            'positive': 2,
            'neutral': 1,
            'negative': 0,
            'null': -1  # For test data
        }
        
        # Image transformations
        if transform is None:
            if mode == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        guid = str(row['guid'])
        
        # Load text
        text_path = os.path.join(self.data_dir, f"{guid}.txt")
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except:
            text = ""  # Empty text if file not found
        
        # Load image
        image_path = os.path.join(self.data_dir, f"{guid}.jpg")
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            # Create blank image if file not found
            image = Image.new('RGB', (224, 224), color='white')
        
        # Process text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Process image
        image_tensor = self.transform(image)
        
        # Get label
        label = self.label_map[row['tag']]
        
        return {
            'guid': guid,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.long)
        }


class OversampledDataset(Dataset):
    """Wrapper dataset that applies oversampling indices"""
    
    def __init__(self, base_dataset, indices):
        """
        Args:
            base_dataset: Base dataset to sample from
            indices: List of indices to sample (with repetition for oversampling)
        """
        self.base_dataset = base_dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


def apply_oversampling(dataset, config):
    """
    Apply oversampling to the dataset
    
    Args:
        dataset: PyTorch dataset or subset
        config: Configuration dictionary
    
    Returns:
        oversampled_dataset: Dataset with oversampling applied
    """
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    
    strategy = config.get('class_imbalance', {}).get('oversample_strategy', None)
    if strategy is None or strategy == 'null' or strategy == '':
        return dataset
    
    # Get labels from dataset
    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        labels.append(item['label'].item())
    labels = np.array(labels)
    
    # Create indices array
    indices = np.arange(len(dataset)).reshape(-1, 1)
    
    # Get sampling strategy
    sampling_strategy = config.get('class_imbalance', {}).get('sampling_strategy', 'auto')
    
    # Apply oversampling
    if strategy == 'random':
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=config['seed'])
        indices_resampled, labels_resampled = sampler.fit_resample(indices, labels)
        indices_resampled = indices_resampled.flatten()
        
        print(f"  Original class distribution: {Counter(labels)}")
        print(f"  Resampled class distribution: {Counter(labels_resampled)}")
        
        return OversampledDataset(dataset, indices_resampled.tolist())
    
    elif strategy == 'smote':
        # SMOTE requires feature vectors, not indices
        # For SMOTE, we would need to extract features first
        # This is more complex for multimodal data, so we'll use RandomOverSampler as fallback
        print("Warning: SMOTE not fully implemented for multimodal data, falling back to RandomOverSampler")
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=config['seed'])
        indices_resampled, labels_resampled = sampler.fit_resample(indices, labels)
        indices_resampled = indices_resampled.flatten()
        
        print(f"  Original class distribution: {Counter(labels)}")
        print(f"  Resampled class distribution: {Counter(labels_resampled)}")
        
        return OversampledDataset(dataset, indices_resampled.tolist())
    
    else:
        print(f"Warning: Unknown oversample strategy '{strategy}', no oversampling applied")
        return dataset


def get_data_loaders(config, tokenizer, fold_idx=None, k_folds=None):
    """
    Create train and validation data loaders
    
    Args:
        config: Configuration dictionary
        tokenizer: BERT tokenizer
        fold_idx: Optional fold index for k-fold cross-validation
        k_folds: Optional total number of folds for k-fold cross-validation
    
    Returns:
        train_loader, val_loader
    """
    # Load full training data
    full_dataset = MultimodalDataset(
        data_file=config['train_file'],
        data_dir=config['data_dir'],
        tokenizer=tokenizer,
        max_length=config['data']['max_text_length'],
        image_size=config['data']['image_size'],
        mode='train'
    )
    
    # Get labels for stratification
    all_labels = []
    for i in range(len(full_dataset)):
        all_labels.append(full_dataset[i]['label'].item())
    all_labels = np.array(all_labels)
    
    # Split into train and validation
    if k_folds is not None and fold_idx is not None:
        # Stratified k-fold cross-validation
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config['seed'])
        folds = list(skf.split(np.arange(len(full_dataset)), all_labels))
        train_indices, val_indices = folds[fold_idx]
        
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        
        print(f"Fold {fold_idx + 1}/{k_folds}: Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    else:
        # Simple train/val split with stratification
        from sklearn.model_selection import train_test_split
        train_indices, val_indices = train_test_split(
            np.arange(len(full_dataset)),
            test_size=config['training']['val_split'],
            random_state=config['seed'],
            stratify=all_labels
        )
        
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
    
    # Apply oversampling to training data only
    if config.get('class_imbalance', {}).get('oversample_strategy'):
        print("Applying oversampling to training data...")
        train_dataset = apply_oversampling(train_dataset, config)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_test_loader(config, tokenizer):
    """
    Create test data loader
    
    Args:
        config: Configuration dictionary
        tokenizer: BERT tokenizer
    
    Returns:
        test_loader
    """
    test_dataset = MultimodalDataset(
        data_file=config['test_file'],
        data_dir=config['data_dir'],
        tokenizer=tokenizer,
        max_length=config['data']['max_text_length'],
        image_size=config['data']['image_size'],
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return test_loader
