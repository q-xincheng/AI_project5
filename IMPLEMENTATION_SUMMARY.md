# Implementation Summary: Class Imbalance & Monitoring Improvements

## Overview
This document summarizes the comprehensive improvements made to the multimodal sentiment classification system to address class imbalance and enhance monitoring capabilities.

## Problem Statement
The original dataset had significant class imbalance:
- **Positive**: 59.7% (2,388 samples)
- **Negative**: 29.8% (1,193 samples)  
- **Neutral**: 10.5% (419 samples)

This imbalance led to:
1. Poor performance on minority classes (especially neutral)
2. Over-prediction of positive class
3. Limited training stability and reliability
4. Insufficient monitoring of per-class performance

## Solutions Implemented

### 1. Data-Level Solutions (dataset.py)
- **RandomOverSampler**: Randomly duplicates minority class samples
- **SMOTE Support**: Framework for synthetic sample generation
- **Stratified Splitting**: Ensures consistent class distribution in train/val splits
- **Stratified K-Fold CV**: Maintains class ratios across all folds

**Key Features**:
- Only applies to training data (validation remains untouched)
- Configurable sampling strategies (auto, manual targets)
- Detailed logging of class distributions before/after resampling

### 2. Loss-Level Solutions (loss.py, train.py)
- **Class Weights**: Configurable weights for CrossEntropyLoss
- **Focal Loss**: Automatically down-weights easy examples, focuses on hard cases
- **Mixed Strategies**: Support for combining oversampling + weighted loss

**Key Features**:
- Focal Loss with configurable gamma (focusing parameter) and alpha (class weights)
- Flexible loss selection via configuration
- Proper device handling for all loss computations

### 3. Prediction-Level Solutions (utils.py, train.py, predict.py)
- **Temperature Scaling**: Calibrates prediction probabilities
  - T > 1.0 = softer/smoother probabilities
  - T < 1.0 = sharper/more confident probabilities
- **Threshold Adjustment**: Raises bar for positive class predictions
  - threshold > 0.5 = harder to predict positive
  - Reduces false positive rate for majority class

**Key Features**:
- Shared utility function for consistent application
- Applied during validation and prediction (not training)
- Configurable per experiment

### 4. Stability Solutions (train.py)
- **Multi-Seed Training**: 
  - Train with multiple random seeds
  - Reports average metrics with standard deviation
  - Helps assess model robustness
  
- **Stratified K-Fold Cross-Validation**:
  - Maintains class distribution in each fold
  - Reports per-fold and average metrics
  - Better assessment of model performance

### 5. Monitoring Enhancements (train.py)
- **Enhanced Metrics**:
  - Macro-F1 score (balanced across classes)
  - Per-class recall (negative, neutral, positive)
  - Detailed classification reports

- **Neutral Error Analysis**:
  - Tracks all neutral samples misclassified as positive/negative
  - Saves to `outputs/neutral_errors.txt` with:
    - Sample GUID
    - Predicted label
    - Probability distribution
  - Groups errors by predicted class

- **Improved Visualizations**:
  - 6-panel training history plot
  - Confusion matrix with recall annotations
  - Per-class recall curves over epochs

## Configuration Options

All features are configurable via `configs/config.yaml`:

```yaml
class_imbalance:
  oversample_strategy: "random"  # null, "random", "smote"
  sampling_strategy: "auto"       # "auto" or {0: count, 1: count, 2: count}
  class_weights: [1.0, 2.5, 1.0] # null or [neg, neu, pos]
  loss_function: "ce"             # "ce" or "focal"
  focal_gamma: 2.0                # Focal loss focusing parameter
  focal_alpha: null               # null or [neg, neu, pos]

prediction:
  positive_threshold: 0.5         # 0.5 = standard, >0.5 = harder to predict positive
  temperature: 1.0                # 1.0 = no scaling, >1.0 = softer probs

stability:
  multi_seed: null                # null or [seed1, seed2, ...]
  k_fold: null                    # null or integer k

monitoring:
  save_neutral_errors: true
  verbose_metrics: true
```

## Output Files

Training generates comprehensive outputs in `outputs/`:

1. **best_model.pth** - Best model checkpoint
2. **training_history.png** - 6-panel training curves
3. **confusion_matrix.png** - Matrix with recall annotations
4. **results.txt** - Validation metrics and classification report
5. **detailed_metrics.txt** - Complete metrics summary
6. **neutral_errors.txt** - Neutral misclassification analysis
7. **multiseed_results.txt** - Multi-seed results (if enabled)
8. **kfold_results.txt** - K-fold CV results (if enabled)

## Code Quality

### Security
- ✓ Passed CodeQL security analysis (0 alerts)
- ✓ No hardcoded credentials or sensitive data
- ✓ Proper input validation

### Code Review
- ✓ Addressed all review feedback
- ✓ Extracted shared utilities to avoid duplication
- ✓ Fixed device placement issues
- ✓ Improved documentation clarity

### Testing
All core components tested:
- ✓ Focal Loss computation
- ✓ RandomOverSampler functionality
- ✓ Stratified splitting
- ✓ Temperature scaling
- ✓ Threshold adjustment
- ✓ Configuration loading

## Usage Examples

### Basic Training with Class Imbalance Handling
```bash
# Edit configs/config.yaml to set:
# class_imbalance.oversample_strategy: "random"
# class_imbalance.class_weights: [1.0, 2.5, 1.0]

python main.py --mode train --config configs/config.yaml
```

### K-Fold Cross-Validation
```bash
# Edit configs/config.yaml to set:
# stability.k_fold: 5

python main.py --mode train --config configs/config.yaml
```

### Multi-Seed Robustness Testing
```bash
# Edit configs/config.yaml to set:
# stability.multi_seed: [42, 123, 456, 789, 2024]

python main.py --mode train --config configs/config.yaml
```

### Prediction with Adjusted Thresholds
```bash
# Edit configs/config.yaml to set:
# prediction.positive_threshold: 0.6
# prediction.temperature: 1.2

python main.py --mode predict --config configs/config.yaml
```

## Performance Expectations

With these improvements, you should expect:

1. **Better Neutral Class Performance**:
   - Higher neutral recall (from error analysis and focused training)
   - Better separation from positive/negative classes

2. **Reduced Positive Bias**:
   - More balanced predictions across all three classes
   - Lower false positive rate for positive class

3. **More Reliable Metrics**:
   - Macro-F1 better reflects performance across all classes
   - Per-class recall shows true performance for each sentiment

4. **Increased Confidence**:
   - Multi-seed/K-fold results show model stability
   - Standard deviation indicates prediction variance

## Maintenance and Extension

### Adding New Oversampling Methods
1. Implement in `src/dataset.py::apply_oversampling()`
2. Add configuration option in config schema
3. Update documentation in README.md

### Adding New Loss Functions
1. Implement in `src/loss.py`
2. Add to `get_criterion()` factory function
3. Update configuration and documentation

### Customizing Prediction Adjustments
1. Modify `src/utils.py::apply_prediction_adjustments()`
2. Add configuration parameters
3. Update both train.py and predict.py usage

## References

- **Focal Loss**: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
- **Imbalanced Learning**: He & Garcia "Learning from Imbalanced Data" (IEEE TKDE 2009)
- **SMOTE**: Chawla et al. "SMOTE: Synthetic Minority Over-sampling Technique" (JAIR 2002)
- **imbalanced-learn**: https://imbalanced-learn.org/

## Contributors

**Author**: GitHub Copilot Workspace Agent
**Repository**: q-xincheng/AI_project5
**Date**: January 2026
