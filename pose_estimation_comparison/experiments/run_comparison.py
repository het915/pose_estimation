#!/usr/bin/env python3
"""Run comparison experiments across all three models."""

import argparse
import json
import yaml
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import PVN3DModel, FoundationPoseModel, AttentionPoseModel
from data import PoseEstimationDataset, collate_fn
from evaluation import PoseEvaluator
from evaluation.visualize import plot_occlusion_comparison
from torch.utils.data import DataLoader


def load_config(config_path: str) -> dict:
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(model_name: str, config: dict, device: str):
    """Create model based on configuration."""
    if model_name == "pvn3d":
        model = PVN3DModel(
            num_points=config["model"]["num_points"],
            num_keypoints=config["model"]["num_keypoints"],
            pointnet_features=config["model"]["pointnet_features"],
            mlp_features=config["model"]["mlp_features"],
            use_rgb=config["model"]["use_rgb"],
        )
    elif model_name == "foundation_pose":
        model = FoundationPoseModel(
            backbone=config["model"]["encoder"]["backbone"],
            feature_dim=config["model"]["encoder"]["feature_dim"],
            num_reference_views=config["model"]["renderer"]["num_reference_views"],
            num_refinement_iterations=config["model"]["refinement"]["num_iterations"],
        )
    elif model_name == "attention_pose":
        model = AttentionPoseModel(
            backbone=config["model"]["encoder"]["backbone"],
            feature_dim=config["model"]["encoder"]["feature_dim"],
            num_reference_views=config["model"]["renderer"]["num_reference_views"],
            num_refinement_iterations=config["model"]["refinement"]["num_iterations"],
            num_attention_heads=config["model"]["attention"]["spatial_attention"]["num_heads"],
            enable_spatial_attention=config["model"]["attention"]["spatial_attention"]["enabled"],
            enable_cross_modal=config["model"]["attention"]["cross_modal_attention"]["enabled"],
            enable_cross_reference=config["model"]["attention"]["cross_reference_attention"]["enabled"],
            enable_uncertainty=config["model"]["attention"]["uncertainty_net"]["enabled"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def run_experiment(
    model_name: str,
    config_path: str,
    data_root: str,
    checkpoint_path: str = None,
    device: str = "cuda",
) -> dict:
    """
    Run evaluation experiment for a single model.

    Args:
        model_name: Name of the model
        config_path: Path to configuration file
        data_root: Root directory of dataset
        checkpoint_path: Optional path to model checkpoint
        device: Device to run on

    Returns:
        Dictionary of results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name.upper()}")
    print(f"{'='*60}\n")

    # Load config
    config = load_config(config_path)

    # Create model
    model = create_model(model_name, config, device)

    # Load checkpoint if provided
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Create dataset
    dataset = PoseEstimationDataset(
        data_root=data_root,
        split="test",
        objects=config["data"]["test_objects"],
        occlusion_levels=config["data"]["occlusion_levels"],
        num_points=config["model"].get("num_points", 1024),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # Create evaluator
    evaluator = PoseEvaluator(
        model=model,
        dataloader=dataloader,
        device=device,
        occlusion_bins=config["data"]["occlusion_levels"],
    )

    # Run evaluation
    results = evaluator.evaluate()

    print(f"\nResults for {model_name}:")
    print(f"Overall ADD Accuracy: {results['overall']['add_accuracy']:.2f}%")
    print(f"Overall ADD-S Accuracy: {results['overall']['adds_accuracy']:.2f}%")
    print(f"Overall Rotation Error: {results['overall']['rotation_error_mean']:.2f}°")
    print(f"Overall Translation Error: {results['overall']['translation_error_mean']:.4f}m")

    print(f"\nResults by occlusion:")
    for occ_range, metrics in sorted(results["by_occlusion"].items()):
        print(f"  {occ_range}: ADD={metrics['add_accuracy']:.2f}%, " +
              f"Rot={metrics['rotation_error_mean']:.2f}°")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run pose estimation comparison")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--models", type=str, nargs="+",
                       default=["pvn3d", "foundation_pose", "attention_pose"],
                       help="Models to evaluate")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Model configurations
    config_dir = Path(__file__).parent.parent / "config"
    model_configs = {
        "pvn3d": config_dir / "pvn3d_config.yaml",
        "foundation_pose": config_dir / "foundation_pose_config.yaml",
        "attention_pose": config_dir / "attention_pose_config.yaml",
    }

    # Run experiments
    all_results = {}

    for model_name in args.models:
        if model_name not in model_configs:
            print(f"Unknown model: {model_name}")
            continue

        results = run_experiment(
            model_name=model_name,
            config_path=str(model_configs[model_name]),
            data_root=args.data_root,
            device=args.device,
        )

        all_results[model_name] = results

        # Save individual results
        results_file = output_dir / f"{model_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {results_file}")

    # Create comparison plot
    print(f"\nGenerating comparison plots...")
    plot_occlusion_comparison(
        all_results,
        save_path=str(output_dir / "occlusion_comparison.png")
    )

    # Save combined results
    combined_file = output_dir / "combined_results.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Comparison complete!")
    print(f"Results saved to {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
