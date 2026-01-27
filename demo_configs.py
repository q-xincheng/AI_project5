#!/usr/bin/env python3
"""
Demo script to show all available configuration options for class imbalance handling
"""
import yaml

# Example configurations for different scenarios

print("=" * 80)
print("Configuration Examples for Class Imbalance & Monitoring")
print("=" * 80)

print("\n1. LIGHT OVERSAMPLING + CLASS WEIGHTS (Recommended for most cases)")
print("-" * 80)
config1 = """
class_imbalance:
  oversample_strategy: "random"
  sampling_strategy: "auto"
  class_weights: [1.0, 2.5, 1.0]
  loss_function: "ce"

prediction:
  positive_threshold: 0.55
  temperature: 1.0

monitoring:
  save_neutral_errors: true
  verbose_metrics: true
"""
print(config1)

print("\n2. FOCAL LOSS ONLY (For severe imbalance)")
print("-" * 80)
config2 = """
class_imbalance:
  oversample_strategy: null
  class_weights: null
  loss_function: "focal"
  focal_gamma: 2.0
  focal_alpha: [1.0, 3.0, 1.0]

monitoring:
  save_neutral_errors: true
  verbose_metrics: true
"""
print(config2)

print("\n3. CUSTOM OVERSAMPLING TARGETS")
print("-" * 80)
config3 = """
class_imbalance:
  oversample_strategy: "random"
  # Manually set target counts: {negative: 1200, neutral: 600, positive: 2000}
  sampling_strategy: {0: 1200, 1: 600, 2: 2000}
  class_weights: [1.0, 2.0, 1.0]
  loss_function: "ce"
"""
print(config3)

print("\n4. MULTI-SEED TRAINING (Robustness testing)")
print("-" * 80)
config4 = """
class_imbalance:
  oversample_strategy: "random"
  class_weights: [1.0, 2.5, 1.0]
  loss_function: "ce"

stability:
  multi_seed: [42, 123, 456, 789, 2024]

monitoring:
  save_neutral_errors: true
"""
print(config4)

print("\n5. K-FOLD CROSS-VALIDATION")
print("-" * 80)
config5 = """
class_imbalance:
  oversample_strategy: "random"
  class_weights: [1.0, 2.5, 1.0]
  loss_function: "ce"

stability:
  k_fold: 5

monitoring:
  save_neutral_errors: true
  verbose_metrics: true
"""
print(config5)

print("\n6. AGGRESSIVE POSITIVE THRESHOLD + TEMPERATURE SCALING")
print("-" * 80)
config6 = """
class_imbalance:
  oversample_strategy: "random"
  class_weights: [1.0, 2.5, 1.0]
  loss_function: "ce"

prediction:
  positive_threshold: 0.65  # Higher threshold = less likely to predict positive
  temperature: 1.5           # Softer probabilities

monitoring:
  save_neutral_errors: true
"""
print(config6)

print("\n" + "=" * 80)
print("How to use these configurations:")
print("=" * 80)
print("""
1. Copy the desired configuration section into configs/config.yaml
2. Adjust values based on your needs
3. Run training: python main.py --mode train --config configs/config.yaml
4. Check outputs in the outputs/ directory

Key Files Generated:
- outputs/training_history.png - Training curves with per-class recall
- outputs/confusion_matrix.png - Confusion matrix with recall annotations
- outputs/neutral_errors.txt - Misclassified neutral samples
- outputs/detailed_metrics.txt - Complete metrics summary
- outputs/multiseed_results.txt - Multi-seed training results (if enabled)
- outputs/kfold_results.txt - K-fold CV results (if enabled)
""")
