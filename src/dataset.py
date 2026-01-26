"""
Data loading utilities for multimodal sentiment classification
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer


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


def get_data_loaders(config, tokenizer):
    """
    Create train and validation data loaders
    
    Args:
        config: Configuration dictionary
        tokenizer: BERT tokenizer
    
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
    
    # Split into train and validation
    val_size = int(len(full_dataset) * config['training']['val_split'])
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    
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
