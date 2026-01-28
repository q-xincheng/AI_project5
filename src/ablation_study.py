"""
Ablation study script - Compare text-only, image-only, and multimodal models
"""
import os
import yaml
import torch
from src.train import train


def run_ablation_study():
    """
    Run ablation study with three different modes:
    1. Text-only
    2. Image-only  
    3. Multimodal (full model)
    """
    base_config_path = 'configs/config.yaml'
    
    # Load base config
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    modes = ['text_only', 'image_only', 'multimodal']
    results = {}
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Running experiment: {mode.upper()}")
        print(f"{'='*60}\n")
        
        # Create mode-specific config
        config = base_config.copy()
        config['experiment_mode'] = mode
        config['output_dir'] = f'outputs/{mode}'
        config['model_save_path'] = f'outputs/{mode}/best_model.pth'
        config['predictions_file'] = f'outputs/{mode}/predictions.txt'
        
        # Save mode-specific config
        os.makedirs(config['output_dir'], exist_ok=True)
        mode_config_path = f'configs/config_{mode}.yaml'
        with open(mode_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Train model
        train(mode_config_path)
        
        # Load results
        results_file = os.path.join(config['output_dir'], 'results.txt')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results[mode] = f.read()
    
    # Create comparison report
    print(f"\n{'='*60}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*60}\n")
    
    comparison_report = "# Ablation Study Results\n\n"
    comparison_report += "Comparison of different modality combinations:\n\n"
    
    for mode, result in results.items():
        comparison_report += f"## {mode.upper()}\n"
        comparison_report += f"```\n{result}\n```\n\n"
    
    # Save comparison report
    with open('reports/ablation_study.md', 'w') as f:
        f.write(comparison_report)
    
    print("✓ Ablation study completed!")
    print("Results saved to: reports/ablation_study.md")


if __name__ == '__main__':
    os.makedirs('reports', exist_ok=True)
    run_ablation_study()
