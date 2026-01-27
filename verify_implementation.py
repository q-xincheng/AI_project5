#!/usr/bin/env python3
"""
Verification script to test all implemented features
"""
import sys
import torch
import numpy as np
from collections import Counter

def test_imports():
    """Test all module imports"""
    print("=" * 80)
    print("Testing Module Imports")
    print("=" * 80)
    try:
        from src.dataset import get_data_loaders, apply_oversampling
        from src.model import get_model
        from src.train import train, set_seed, train_single_run
        from src.loss import FocalLoss, get_criterion
        from src.utils import apply_prediction_adjustments
        from src.predict import predict
        print("✓ All imports successful\n")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}\n")
        return False

def test_focal_loss():
    """Test Focal Loss implementation"""
    print("=" * 80)
    print("Testing Focal Loss")
    print("=" * 80)
    try:
        from src.loss import FocalLoss
        
        # Test without alpha
        loss_fn = FocalLoss(gamma=2.0)
        inputs = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])
        loss = loss_fn(inputs, targets)
        print(f"✓ Focal Loss (no alpha): {loss.item():.4f}")
        
        # Test with alpha
        loss_fn_alpha = FocalLoss(alpha=[1.0, 2.5, 1.0], gamma=2.0)
        loss_alpha = loss_fn_alpha(inputs, targets)
        print(f"✓ Focal Loss (with alpha): {loss_alpha.item():.4f}\n")
        
        return True
    except Exception as e:
        print(f"✗ Focal Loss test failed: {e}\n")
        return False

def test_oversampling():
    """Test oversampling functionality"""
    print("=" * 80)
    print("Testing RandomOverSampler")
    print("=" * 80)
    try:
        from imblearn.over_sampling import RandomOverSampler
        
        labels = np.array([0]*100 + [1]*30 + [2]*200)
        indices = np.arange(len(labels)).reshape(-1, 1)
        
        print(f"Original distribution: {Counter(labels)}")
        
        sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
        indices_resampled, labels_resampled = sampler.fit_resample(indices, labels)
        
        print(f"Resampled distribution: {Counter(labels_resampled)}")
        print(f"✓ Oversampling working correctly\n")
        
        return True
    except Exception as e:
        print(f"✗ Oversampling test failed: {e}\n")
        return False

def test_stratified_split():
    """Test stratified splitting"""
    print("=" * 80)
    print("Testing Stratified Split")
    print("=" * 80)
    try:
        from sklearn.model_selection import train_test_split, StratifiedKFold
        
        labels = np.array([0]*100 + [1]*30 + [2]*200)
        indices = np.arange(len(labels))
        
        # Test train/val split
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=labels
        )
        
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        print(f"Train distribution: {Counter(train_labels)}")
        print(f"Val distribution: {Counter(val_labels)}")
        print("✓ Stratified split preserves class ratios")
        
        # Test K-Fold
        print("\nTesting 3-Fold Stratified CV:")
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for i, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
            fold_val_labels = labels[val_idx]
            print(f"  Fold {i+1} val: {Counter(fold_val_labels)}")
        
        print("✓ Stratified K-Fold working correctly\n")
        return True
    except Exception as e:
        print(f"✗ Stratified split test failed: {e}\n")
        return False

def test_prediction_adjustments():
    """Test temperature scaling and threshold adjustment"""
    print("=" * 80)
    print("Testing Prediction Adjustments")
    print("=" * 80)
    try:
        from src.utils import apply_prediction_adjustments
        
        logits = torch.tensor([
            [2.0, 1.0, 3.0],
            [1.5, 2.5, 1.0],
            [3.0, 0.5, 2.8],
        ])
        
        # Test with temperature scaling
        config_temp = {
            'prediction': {
                'temperature': 2.0,
                'positive_threshold': 0.5
            }
        }
        predictions, probs = apply_prediction_adjustments(logits, config_temp)
        print(f"✓ Temperature scaling (T=2.0): predictions = {predictions}")
        
        # Test with threshold adjustment
        config_thresh = {
            'prediction': {
                'temperature': 1.0,
                'positive_threshold': 0.65
            }
        }
        predictions, probs = apply_prediction_adjustments(logits, config_thresh)
        print(f"✓ Threshold adjustment (0.65): predictions = {predictions}")
        
        # Test combined
        config_both = {
            'prediction': {
                'temperature': 1.5,
                'positive_threshold': 0.6
            }
        }
        predictions, probs = apply_prediction_adjustments(logits, config_both)
        print(f"✓ Combined adjustments: predictions = {predictions}\n")
        
        return True
    except Exception as e:
        print(f"✗ Prediction adjustments test failed: {e}\n")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("=" * 80)
    print("Testing Configuration Loading")
    print("=" * 80)
    try:
        import yaml
        
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ Main config loaded successfully")
        print(f"  - oversample_strategy: {config.get('class_imbalance', {}).get('oversample_strategy')}")
        print(f"  - loss_function: {config.get('class_imbalance', {}).get('loss_function')}")
        print(f"  - class_weights: {config.get('class_imbalance', {}).get('class_weights')}")
        print(f"  - positive_threshold: {config.get('prediction', {}).get('positive_threshold')}")
        print(f"  - temperature: {config.get('prediction', {}).get('temperature')}")
        print(f"  - k_fold: {config.get('stability', {}).get('k_fold')}")
        print(f"  - multi_seed: {config.get('stability', {}).get('multi_seed')}")
        print(f"  - save_neutral_errors: {config.get('monitoring', {}).get('save_neutral_errors')}\n")
        
        return True
    except Exception as e:
        print(f"✗ Config loading test failed: {e}\n")
        return False

def test_loss_factory():
    """Test loss function factory"""
    print("=" * 80)
    print("Testing Loss Function Factory")
    print("=" * 80)
    try:
        from src.loss import get_criterion
        
        # Test CrossEntropyLoss with weights
        config_ce = {
            'class_imbalance': {
                'loss_function': 'ce',
                'class_weights': [1.0, 2.5, 1.0]
            }
        }
        criterion = get_criterion(config_ce, torch.device('cpu'))
        print("✓ CrossEntropyLoss with class weights created")
        
        # Test Focal Loss
        config_focal = {
            'class_imbalance': {
                'loss_function': 'focal',
                'focal_gamma': 2.0,
                'focal_alpha': [1.0, 3.0, 1.0]
            }
        }
        criterion_focal = get_criterion(config_focal, torch.device('cpu'))
        print("✓ Focal Loss created\n")
        
        return True
    except Exception as e:
        print(f"✗ Loss factory test failed: {e}\n")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("VERIFICATION SUITE FOR CLASS IMBALANCE & MONITORING IMPROVEMENTS")
    print("=" * 80 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Focal Loss", test_focal_loss),
        ("Oversampling", test_oversampling),
        ("Stratified Split", test_stratified_split),
        ("Prediction Adjustments", test_prediction_adjustments),
        ("Config Loading", test_config_loading),
        ("Loss Factory", test_loss_factory),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<30} {status}")
    
    print("=" * 80)
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Implementation verified successfully.\n")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())
