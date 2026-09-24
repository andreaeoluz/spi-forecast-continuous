#!/usr/bin/env python3
"""main.py - Unified entry point for the continuous SPI forecasting framework.

Derived from the binary framework's main.py. Removed: --threshold-spi
(there is no drought threshold), --use-calibration/--no-calibration/
--recalibrate/--threshold (there is no probability to calibrate or
threshold). Added: --spi-scale, to select which SPI accumulation scale
(SPI-3, SPI-12, ...) the command operates on.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import ExperimentConfig, get_data_path
from utils import set_reproducible_seeds, Logger


def print_config_summary(config: ExperimentConfig, optimization_metric: str = None):
    """Print configuration summary."""
    ds_h, ds_w = config.get_downsample(config.region)
    ds_info = config.get_downsample_info(config.region)

    print(f"\n{'='*60}")
    print(f"📍 Region: {config.region}")
    print(f"📊 SPI-{config.spi.scale} (continuous, no severity threshold)")
    print(f"📉 Downsampling: {ds_h}x{ds_w} ({ds_info['preservation_estimate']})")
    print(f"🔄 Transfer Learning: {config.use_transfer_learning}")

    if optimization_metric:
        print(f"🎯 Optimization: {optimization_metric.upper()}")

    print(f"{'='*60}\n")


def setup_parser() -> argparse.ArgumentParser:
    """Configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Continuous SPI Forecasting Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Precompute SPI
  python main.py precompute-spi

  # Train autoencoder
  python main.py train-ae

  # Run grid search (model selection by validation WI)
  python main.py grid-search --optimization-metric wi

  # Force transfer learning from the pretrained autoencoder
  python main.py grid-search --use-transfer-learning

  # Run inference (uses the best configuration by validation WI)
  python main.py inference --region Sul

  # Specific model
  python main.py inference --region Sul --p 9 --q 1 --model-type pretrained
        """
    )

    subparsers = parser.add_subparsers(dest='command', required=True, help='Command to execute')

    # Precompute SPI
    spi_parser = subparsers.add_parser('precompute-spi', help='Precompute SPI for the region')
    spi_parser.add_argument('--region', type=str, default=None, help='Region')

    # Train Autoencoder
    ae_parser = subparsers.add_parser('train-ae', help='Train autoencoder')
    ae_parser.add_argument('--force', action='store_true', help='Force retraining')
    ae_parser.add_argument('--region', type=str, default=None, help='Region')

    # Grid Search
    gs_parser = subparsers.add_parser('grid-search', help='Run hyperparameter grid search')
    gs_parser.add_argument('--spi-scale', type=int, default=None, help='SPI accumulation scale')
    gs_parser.add_argument('--region', type=str, default=None, help='Region')
    gs_parser.add_argument('--use-transfer-learning', action='store_true', default=None, help='Use transfer learning')
    gs_parser.add_argument('--scratch', action='store_true',
                            help='Force training from scratch, overriding '
                                 'ExperimentConfig.use_transfer_learning\'s default of True '
                                 '(there is no CLI way to turn --use-transfer-learning off otherwise)')
    gs_parser.add_argument('--optimization-metric', choices=['wi', 'rmse', 'mae'], default='wi',
                      help='Metric for model selection [default: wi]')
    gs_parser.add_argument('--severity-weight', type=float, default=None,
                      help='Upweight the MSE loss on more extreme observed SPI months by a factor '
                           'of (1 + severity_weight * |SPI|), mitigating residual mean-reversion '
                           'attenuation under the persistence-anchored formulation '
                           '[default: 0.0, i.e. plain unweighted MSE]')

    # Inference
    inf_parser = subparsers.add_parser('inference', help='Run inference on test period')
    inf_parser.add_argument('--p', type=int, help='History length (default: best model)')
    inf_parser.add_argument('--q', type=int, help='Forecast horizon (default: best model)')
    inf_parser.add_argument('--model-type', choices=['pretrained', 'scratch'],
                           help='Model type (default: auto-detected)')

    inf_parser.add_argument('--list-models', action='store_true', help='List available models')

    inf_parser.add_argument('--region', type=str, default=None, help='Region')
    inf_parser.add_argument('--spi-scale', type=int, default=None, help='SPI accumulation scale')

    return parser


def apply_args_to_config(config: ExperimentConfig, args) -> None:
    """Apply command-line arguments to configuration."""
    if hasattr(args, 'region') and args.region is not None:
        config.region = args.region

    if hasattr(args, 'spi_scale') and args.spi_scale is not None:
        config.spi.scale = args.spi_scale

    if hasattr(args, 'use_transfer_learning') and args.use_transfer_learning is not None:
        config.use_transfer_learning = args.use_transfer_learning

    if getattr(args, 'scratch', False):
        config.use_transfer_learning = False

    if hasattr(args, 'severity_weight') and args.severity_weight is not None:
        config.loss.severity_weight = args.severity_weight


def execute_command(args, config: ExperimentConfig, base_data_path: Path):
    """Execute the requested command."""
    logger = Logger()

    if args.command == 'precompute-spi':
        from experiments import precompute_spi
        precompute_spi(region=args.region)

    elif args.command == 'train-ae':
        from experiments import train_autoencoder
        train_autoencoder(region=args.region)

    elif args.command == 'grid-search':
        from experiments import GridSearch
        optimization_metric = getattr(args, 'optimization_metric', 'wi')

        grid_search = GridSearch(
            config,
            base_data_path,
            optimization_metric=optimization_metric
        )
        grid_search.run()

    elif args.command == 'inference':
        from inference.run import main as inference_main

        argv = [sys.argv[0]]

        if args.p is not None:
            argv.extend(['--p', str(args.p)])
        if args.q is not None:
            argv.extend(['--q', str(args.q)])
        if args.model_type:
            argv.extend(['--model-type', args.model_type])

        if args.list_models:
            argv.append('--list-models')

        if args.region:
            argv.extend(['--region', args.region])
        if args.spi_scale is not None:
            argv.extend(['--spi-scale', str(args.spi_scale)])

        sys.argv = argv
        inference_main()

    else:
        logger.error(f"Unknown command: {args.command}")
        logger.info("Use --help to see available commands")


def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    config = ExperimentConfig()
    apply_args_to_config(config, args)

    optimization_metric = getattr(args, 'optimization_metric', None)
    print_config_summary(config, optimization_metric)

    set_reproducible_seeds(config.random_seed)
    base_data_path = get_data_path()

    execute_command(args, config, base_data_path)


if __name__ == "__main__":
    main()
