"""
Multimodal sentiment classification model
Combines BERT (text) and ResNet (image) with fusion layer
"""
import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models


class MultimodalSentimentClassifier(nn.Module):
    """
    Multimodal model that fuses text and image features for sentiment classification
    """
    
    def __init__(self, config):
        super(MultimodalSentimentClassifier, self).__init__()
        
        self.config = config
        self.mode = config.get('experiment_mode', 'multimodal')
        
        # Text encoder (BERT)
        self.text_encoder = BertModel.from_pretrained(config['model']['text_model'])
        text_hidden_size = self.text_encoder.config.hidden_size
        
        # Image encoder (ResNet50)
        resnet = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        image_hidden_size = 2048  # ResNet50 output dimension
        
        # Freeze some layers for faster training (optional)
        # Freeze early ResNet layers
        for param in list(self.image_encoder.parameters())[:100]:
            param.requires_grad = False
        
        # Fusion layer dimensions
        hidden_dim = config['model']['hidden_dim']
        num_classes = config['model']['num_classes']
        dropout = config['model']['dropout']
        
        # Projection layers to common dimension
        self.text_projection = nn.Sequential(
            nn.Linear(text_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.image_projection = nn.Sequential(
            nn.Linear(image_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        if self.mode == 'multimodal':
            fusion_input_dim = hidden_dim * 2  # Concatenation of text and image
        else:
            fusion_input_dim = hidden_dim  # Single modality
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, image):
        """
        Forward pass
        
        Args:
            input_ids: Text token IDs (batch_size, seq_len)
            attention_mask: Text attention mask (batch_size, seq_len)
            image: Image tensor (batch_size, 3, 224, 224)
        
        Returns:
            logits: Class predictions (batch_size, num_classes)
        """
        # Process text
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token representation
        text_features = text_output.last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_features)
        
        # Process image
        image_features = self.image_encoder(image)
        image_features = image_features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
        image_features = self.image_projection(image_features)
        
        # Fusion based on mode
        if self.mode == 'multimodal':
            # Concatenate text and image features
            combined_features = torch.cat([text_features, image_features], dim=1)
        elif self.mode == 'text_only':
            combined_features = text_features
        elif self.mode == 'image_only':
            combined_features = image_features
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Classification
        logits = self.fusion(combined_features)
        
        return logits


class TextOnlyClassifier(nn.Module):
    """Text-only baseline model for ablation study"""
    
    def __init__(self, config):
        super(TextOnlyClassifier, self).__init__()
        
        self.text_encoder = BertModel.from_pretrained(config['model']['text_model'])
        text_hidden_size = self.text_encoder.config.hidden_size
        
        hidden_dim = config['model']['hidden_dim']
        num_classes = config['model']['num_classes']
        dropout = config['model']['dropout']
        
        self.classifier = nn.Sequential(
            nn.Linear(text_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, image=None):
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_output.last_hidden_state[:, 0, :]
        logits = self.classifier(text_features)
        return logits


class ImageOnlyClassifier(nn.Module):
    """Image-only baseline model for ablation study"""
    
    def __init__(self, config):
        super(ImageOnlyClassifier, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        image_hidden_size = 2048
        
        hidden_dim = config['model']['hidden_dim']
        num_classes = config['model']['num_classes']
        dropout = config['model']['dropout']
        
        self.classifier = nn.Sequential(
            nn.Linear(image_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, input_ids=None, attention_mask=None, image=None):
        image_features = self.image_encoder(image)
        image_features = image_features.squeeze(-1).squeeze(-1)
        logits = self.classifier(image_features)
        return logits


def get_model(config):
    """
    Factory function to get the appropriate model based on experiment mode
    
    Args:
        config: Configuration dictionary
    
    Returns:
        model: PyTorch model
    """
    mode = config.get('experiment_mode', 'multimodal')
    
    if mode == 'text_only':
        return TextOnlyClassifier(config)
    elif mode == 'image_only':
        return ImageOnlyClassifier(config)
    else:  # multimodal
        return MultimodalSentimentClassifier(config)
