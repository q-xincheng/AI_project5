"""
Utility functions for prediction adjustments
"""
import torch
import torch.nn.functional as F


def apply_prediction_adjustments(logits, config):
    """
    Apply temperature scaling and threshold adjustments to logits
    
    Args:
        logits: Raw model outputs (batch_size, num_classes)
        config: Configuration dictionary with prediction settings
    
    Returns:
        predictions: Class predictions (batch_size,) as numpy array
    """
    # Get prediction config
    temperature = config.get('prediction', {}).get('temperature', 1.0)
    positive_threshold = config.get('prediction', {}).get('positive_threshold', 0.5)
    
    # Apply temperature scaling
    if temperature != 1.0:
        logits = logits / temperature
    
    # Get probabilities
    probs = F.softmax(logits, dim=1)
    
    # Apply threshold adjustment for positive class
    if positive_threshold != 0.5:
        # Adjust the decision boundary for positive class (class 2)
        # If positive_threshold > 0.5, make it harder to predict positive
        adjusted_probs = probs.clone()
        # Scale positive class probability
        scale_factor = 0.5 / positive_threshold
        adjusted_probs[:, 2] = adjusted_probs[:, 2] * scale_factor
        # Re-normalize
        adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=1, keepdim=True)
        predictions = torch.argmax(adjusted_probs, dim=1)
    else:
        predictions = torch.argmax(probs, dim=1)
    
    return predictions, probs
