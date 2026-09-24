#!/usr/bin/env python3
"""run.py - Main inference script for continuous SPI forecasting.

Derived from the binary framework's inference/run.py. Removed: calibration
flags, decision-threshold flags/recalibration, --list-models' probing of
prevalence-threshold folder names (now SPI-scale folder names).
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ExperimentConfig
from config.paths import get_paths, get_data_path
from inference.predictor import InferencePredictor
from utils import set_reproducible_seeds
from utils.logger import Logger


# =============================================================================
# UTILITIES
# =============================================================================

def convert_to_serializable(obj):
    """Convert numpy objects to JSON-serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


# =============================================================================
# CONFIGURATION
# =============================================================================

def find_best_config(
    region: str,
    spi_scale: int,
    transfer_learning: bool = None
) -> dict:
    """Find the best configuration from the grid search results (selected
    on validation WI - see experiments.grid_search.GridSearch)."""
    paths = get_paths(region, spi_scale=spi_scale)

    base = f"spi{spi_scale}"

    # GridSearch's own suffix always includes "_tl_{bool}" (see
    # GridSearch.__init__), plus "_noattn" for the attention-ablation runs.
    tl_options = [True, False] if transfer_learning is None else [transfer_learning]

    config_files = []
    for tl in tl_options:
        suffix = f"{base}_tl_{tl}"
        suffix_variants = [suffix, f"{suffix}_noattn"] if tl else [suffix]
        for s in suffix_variants:
            config_files.append(paths["grid_search_results"] / s / f"best_configuration_{s}.json")

    results_path = None
    for f in config_files:
        if f.exists():
            results_path = f
            break

    if results_path is None:
        searched = "\n".join(f"  - {c}" for c in config_files)
        raise FileNotFoundError(
            f"No configuration found for region '{region}' with SPI-{spi_scale}. Searched:\n{searched}"
        )

    with open(results_path, "r") as f:
        data = json.load(f)

    if "best_configuration" in data:
        best = data["best_configuration"]
    else:
        best = data

    if "p" not in best or "q" not in best:
        raise ValueError("Configuration does not contain p and q")

    return best


def list_available_models(config: ExperimentConfig):
    """List available trained models for inference."""
    paths = get_paths(config.region, spi_scale=config.spi.scale)
    logger = Logger()

    logger.header(f"AVAILABLE MODELS - {config.region}")

    search_dirs = []

    if "grid_search_pretrained" in paths:
        search_dirs.append(("pretrained", paths["grid_search_pretrained"]))
        noattn_dir = paths["grid_search_pretrained"].parent / "pretrained_noattn"
        search_dirs.append(("pretrained_noattn", noattn_dir))
    if "grid_search_scratch" in paths:
        search_dirs.append(("scratch", paths["grid_search_scratch"]))

    gs_dir = paths.get("grid_search_dir")
    if gs_dir and gs_dir.exists():
        for model_type in ["pretrained", "scratch"]:
            d = gs_dir / model_type
            if d.exists():
                search_dirs.append((model_type, d))

    models = []
    for model_type, search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for model_file in search_dir.glob("model_*.pth"):
            name = model_file.stem
            parts = name.split("_")

            p = None
            q = None

            for part in parts:
                if part.startswith("p") and part[1:].isdigit():
                    p = int(part[1:])
                elif part.startswith("q") and part[1:].isdigit():
                    q = int(part[1:])

            if p is not None and q is not None:
                models.append({
                    "p": p,
                    "q": q,
                    "type": model_type,
                    "path": model_file,
                    "dirname": search_dir.relative_to(paths["base"]) if "base" in paths else search_dir.name
                })

    seen = set()
    unique_models = []
    for m in sorted(models, key=lambda x: (x["p"], x["q"], x["type"])):
        key = (m["p"], m["q"], m["type"])
        if key not in seen:
            seen.add(key)
            unique_models.append(m)

    if not unique_models:
        logger.warning("No models found!")
        return

    print(f"\n{'p':<6} {'q':<6} {'Type':<12} {'Directory':<40}")
    print(f"{'-'*70}")
    for m in unique_models:
        print(f"{m['p']:<6} {m['q']:<6} {m['type']:<12} {str(m['dirname']):<40}")

    print(f"\n✅ Total: {len(unique_models)} models found")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point for the inference script."""
    parser = argparse.ArgumentParser(
        description="Continuous SPI inference on the test period",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference with automatic best-configuration selection (by validation WI)
  python3 main.py inference --region Sul

  # Inference with a specific configuration
  python3 main.py inference --region Sul --p 9 --q 1 --model-type pretrained

  # List available models
  python3 main.py inference --region Sul --list-models
        """
    )

    parser.add_argument("--region", type=str, default=None,
                       help="Region (default: from config)")
    parser.add_argument("--spi-scale", type=int, default=None,
                       help="SPI accumulation scale (default: from config)")

    parser.add_argument("--p", type=int, help="History length (context months)")
    parser.add_argument("--q", type=int, help="Forecast horizon (months ahead)")
    parser.add_argument("--model-type", choices=["pretrained", "scratch"],
                       help="Model type (pretrained or scratch)")
    parser.add_argument("--model-dir", type=str, default=None,
                       help="Custom path to the model")

    parser.add_argument("--use-transfer-learning", dest="transfer_learning",
                       action="store_true", default=None,
                       help="Restrict automatic best-configuration search to transfer-learning runs")
    parser.add_argument("--no-transfer-learning", dest="transfer_learning",
                       action="store_false",
                       help="Restrict automatic best-configuration search to scratch (no transfer learning) runs")

    parser.add_argument("--list-models", action="store_true",
                       help="List available models and exit")

    args = parser.parse_args()

    # =====================================================================
    # CONFIGURATION
    # =====================================================================

    config = ExperimentConfig()

    if args.region:
        config.region = args.region

    if args.spi_scale is not None:
        config.spi.scale = args.spi_scale

    logger = Logger()

    if args.list_models:
        list_available_models(config)
        return

    set_reproducible_seeds(config.random_seed)
    base_data_path = get_data_path()

    # =====================================================================
    # DETERMINE p, q AND model_type
    # =====================================================================

    if args.model_dir:
        p = args.p if args.p else 6
        q = args.q if args.q else 1
        model_type = args.model_type if args.model_type else "pretrained"
        logger.info(f"📁 Using custom model: {args.model_dir}")

    elif args.p is None or args.q is None:
        logger.info("🔍 Searching for the best configuration (by validation WI)...")
        try:
            best_config = find_best_config(
                config.region,
                config.spi.scale,
                transfer_learning=args.transfer_learning,
            )
            p = best_config["p"]
            q = best_config["q"]
            logger.success(f"✅ Best configuration: p={p}, q={q}")

            if args.model_type is None:
                model_type = "pretrained" if best_config.get("transfer_learning", False) else "scratch"
            else:
                model_type = args.model_type

            logger.info(f"  Model type: {model_type}")

        except Exception as e:
            logger.error(f"Error finding the best configuration: {e}")
            logger.info("Use --p and --q to specify manually")
            return
    else:
        p = args.p
        q = args.q
        model_type = args.model_type if args.model_type else "pretrained"

    # =====================================================================
    # CREATE PREDICTOR
    # =====================================================================

    try:
        predictor = InferencePredictor(
            config=config,
            base_data_path=base_data_path,
            p=p,
            q=q,
            model_type=model_type,
            model_dir=Path(args.model_dir) if args.model_dir else None,
        )
    except Exception as e:
        logger.error(f"❌ Error creating predictor: {e}")
        return

    # =====================================================================
    # RUN INFERENCE
    # =====================================================================

    result = predictor.run_inference()

    if not result or not result.get("success", True):
        logger.error("❌ Inference failed!")
        return

    # =====================================================================
    # SAVE RASTERS
    # =====================================================================

    predictor.save_rasters(result)

    # =====================================================================
    # SAVE PREDICTIONS AS NUMPY (for downstream analysis/reuse)
    # =====================================================================

    predictions_path = predictor.paths.get("analysis_metrics", predictor.paths["inference_dir"] / "metrics")
    predictions_path.mkdir(parents=True, exist_ok=True)

    np.save(predictions_path / "spi_predicted.npy", result["pred"])
    np.save(predictions_path / "spi_observed.npy", result["obs"])

    # =====================================================================
    # SAVE METRICS
    # =====================================================================

    import pandas as pd

    metrics_data = {
        "region": config.region,
        "spi_scale": config.spi.scale,
        "p": p,
        "q": q,
        "model_type": model_type,
        "n_samples": result["n_samples"],
    }

    if result.get("metrics") is not None:
        for key, value in result["metrics"].items():
            metrics_data[f"metrics_{key}"] = value

    df = pd.DataFrame([metrics_data])
    excel_file = predictions_path / "metrics.xlsx"
    df.to_excel(excel_file, index=False)
    logger.success(f"✅ Metrics saved to: {excel_file}")

    json_file = predictions_path / "metrics.json"
    metrics_data_serializable = convert_to_serializable(metrics_data)
    with open(json_file, "w") as f:
        json.dump(metrics_data_serializable, f, indent=2)
    logger.success(f"✅ Metrics JSON saved to: {json_file}")

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================

    logger.header("📊 INFERENCE SUMMARY")
    logger.info(f"  Region: {config.region}")
    logger.info(f"  SPI scale: {config.spi.scale}")
    logger.info(f"  p: {p}, q: {q}")
    logger.info(f"  Model: {model_type}")
    logger.info(f"  Period: {config.split.test[0]} to {config.split.test[1]}")
    logger.info(f"  Samples: {result['n_samples']}")
    logger.info(f"  WI: {result['metrics']['wi']:.4f}")
    logger.info(f"  RMSE: {result['metrics']['rmse']:.4f}")
    logger.info(f"  MAE: {result['metrics']['mae']:.4f}")

    logger.info(f"\n📁 Rasters saved to: {predictor.paths['inference_dir']}")


if __name__ == "__main__":
    main()
