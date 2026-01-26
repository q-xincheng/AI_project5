"""
Main entry point for multimodal sentiment classification
"""
import argparse
from src.train import train
from src.predict import predict
from src.ablation_study import run_ablation_study


def main():
    parser = argparse.ArgumentParser(
        description='Multimodal Sentiment Classification'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict', 'ablation'],
        default='train',
        help='Mode: train, predict, or ablation study'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model checkpoint (for prediction)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Training model...")
        train(args.config)
    elif args.mode == 'predict':
        print("Generating predictions...")
        predict(args.config, args.model)
    elif args.mode == 'ablation':
        print("Running ablation study...")
        run_ablation_study()
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
