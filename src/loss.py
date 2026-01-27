"""
Loss functions for multimodal sentiment classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for each class (list or tensor), or None for no weighting
            gamma: Focusing parameter (default: 2.0)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, list):
                alpha = torch.tensor(alpha)
            self.alpha = alpha
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
        
        Returns:
            loss: Focal loss
        """
        # Get class probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get the probability of the true class
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_criterion(config, device):
    """
    Factory function to get the appropriate loss function
    
    Args:
        config: Configuration dictionary
        device: Device to place the criterion on
    
    Returns:
        criterion: Loss function
    """
    loss_type = config.get('class_imbalance', {}).get('loss_function', 'ce')
    
    # Get class weights if specified
    class_weights = config.get('class_imbalance', {}).get('class_weights')
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    
    if loss_type == 'focal':
        # Focal Loss
        focal_alpha = config.get('class_imbalance', {}).get('focal_alpha')
        focal_gamma = config.get('class_imbalance', {}).get('focal_gamma', 2.0)
        
        if focal_alpha is not None:
            focal_alpha = torch.tensor(focal_alpha, dtype=torch.float32, device=device)
        
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
        print(f"Using Focal Loss (gamma={focal_gamma}, alpha={focal_alpha})")
    else:
        # Standard Cross Entropy Loss
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        if class_weights is not None:
            print(f"Using CrossEntropyLoss with class weights: {class_weights.cpu().numpy()}")
        else:
            print("Using CrossEntropyLoss without class weights")
    
    return criterion
