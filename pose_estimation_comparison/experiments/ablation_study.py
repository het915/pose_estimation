#!/usr/bin/env python3
"""Ablation study for AttentionPose."""

import argparse
import json
import yaml
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import AttentionPoseModel
from data import PoseEstimationDataset, collate_fn
from evaluation import PoseEvaluator
from torch.utils.data import DataLoader


def run_ablation(
    config_path: str,
    data_root: str,
    device: str = "cuda",
) -> dict:
    """
    Run ablation study for AttentionPose.

    Tests contribution of each attention module:
    1. Full model (all attention modules)
    2. Without spatial attention
    3. Without cross-modal attention
    4. Without cross-reference attention
    5. Without uncertainty estimation
    6. Baseline (no attention modules)

    Returns:
        Dictionary of results for each ablation
    """
    print(f"\n{'='*60}")
    print(f"AttentionPose Ablation Study")
    print(f"{'='*60}\n")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create dataset
    dataset = PoseEstimationDataset(
        data_root=data_root,
        split="test",
        objects=config["data"]["test_objects"],
        occlusion_levels=config["data"]["occlusion_levels"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # Ablation configurations
    ablations = {
        "full_model": {
            "enable_spatial_attention": True,
            "enable_cross_modal": True,
            "enable_cross_reference": True,
            "enable_uncertainty": True,
        },
        "no_spatial": {
            "enable_spatial_attention": False,
            "enable_cross_modal": True,
            "enable_cross_reference": True,
            "enable_uncertainty": True,
        },
        "no_cross_modal": {
            "enable_spatial_attention": True,
            "enable_cross_modal": False,
            "enable_cross_reference": True,
            "enable_uncertainty": True,
        },
        "no_cross_reference": {
            "enable_spatial_attention": True,
            "enable_cross_modal": True,
            "enable_cross_reference": False,
            "enable_uncertainty": True,
        },
        "no_uncertainty": {
            "enable_spatial_attention": True,
            "enable_cross_modal": True,
            "enable_cross_reference": True,
            "enable_uncertainty": False,
        },
        "baseline": {
            "enable_spatial_attention": False,
            "enable_cross_modal": False,
            "enable_cross_reference": False,
            "enable_uncertainty": False,
        },
    }

    all_results = {}

    for ablation_name, ablation_config in ablations.items():
        print(f"\nRunning ablation: {ablation_name}")
        print(f"Configuration: {ablation_config}")

        # Create model with ablation settings
        model = AttentionPoseModel(
            backbone=config["model"]["encoder"]["backbone"],
            feature_dim=config["model"]["encoder"]["feature_dim"],
            num_reference_views=config["model"]["renderer"]["num_reference_views"],
            num_refinement_iterations=config["model"]["refinement"]["num_iterations"],
            **ablation_config,
        )

        model = model.to(device)

        # Create evaluator
        evaluator = PoseEvaluator(
            model=model,
            dataloader=dataloader,
            device=device,
            occlusion_bins=config["data"]["occlusion_levels"],
        )

        # Run evaluation
        results = evaluator.evaluate()

        all_results[ablation_name] = results

        print(f"Results for {ablation_name}:")
        print(f"  ADD Accuracy: {results['overall']['add_accuracy']:.2f}%")
        print(f"  60% Occlusion ADD: {results['by_occlusion'].get('60-80%', {}).get('add_accuracy', 0):.2f}%")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Ablation study for AttentionPose")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--config", type=str,
                       default="config/attention_pose_config.yaml",
                       help="Configuration file")
    parser.add_argument("--output-dir", type=str, default="results/ablation",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Run ablation study
    results = run_ablation(
        config_path=args.config,
        data_root=args.data_root,
        device=args.device,
    )

    # Save results
    results_file = output_dir / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAblation study complete!")
    print(f"Results saved to {results_file}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Ablation Study Summary")
    print(f"{'='*60}")
    print(f"{'Configuration':<25} {'ADD Acc':<12} {'60% Occ ADD':<15}")
    print(f"{'-'*60}")

    for ablation_name, result in results.items():
        overall_add = result['overall']['add_accuracy']
        occ_60_add = result['by_occlusion'].get('60-80%', {}).get('add_accuracy', 0)
        print(f"{ablation_name:<25} {overall_add:>10.2f}% {occ_60_add:>13.2f}%")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
