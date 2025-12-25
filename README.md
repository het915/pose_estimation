# Open-World Semantic-Based Zero-Shot 6D Pose Estimation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)

> **A novel open-vocabulary 6D object pose tracking framework combining SAM-3, Moondream2, and FoundationPose for language-guided, zero-shot tracking of arbitrary objects without CAD models.**

<p align="center">
  <img src="https://img.shields.io/badge/CS543-Computer%20Vision-blue" alt="CS543">
  <img src="https://img.shields.io/badge/Institution-UIUC-orange" alt="UIUC">
  <img src="https://img.shields.io/badge/Date-December%202025-green" alt="Date">
</p>

---

## Team Members

- **Het Patel** (hcp4)
- **Sunny Deshpande** (sunnynd2)
- **Ansh Bhansali** (anshb3)
- **Keisuke Ogawa** (ogawa3)

---

## Overview

This project extends NVIDIA's **FoundationPose** architecture to enable **language-guided, zero-shot 6D pose tracking** of arbitrary objects without pre-registered CAD models. By integrating:

- **Moondream2** vision-language model for semantic scene understanding
- **SAM-3** for language-driven segmentation
- **Objaverse-XL** for on-the-fly 3D mesh generation
- **TripoSR** for single-image 3D reconstruction

The system achieves **real-time, occlusion-robust pose estimation** with **dynamic target switching** via natural language prompts—enabling robotic manipulators to seamlessly transition between tracking different objects (e.g., "grasp the red bottle" → "now grasp the blue cup") in unstructured environments.

<p align="center">
  <img src="GIF.gif" alt="6D Pose Estimation Demo" width="800">
</p>

---

## Key Features

- **Open-Vocabulary Detection**: Lightweight Moondream2 VLM for edge-compatible semantic scene understanding
- **Zero-Shot Mesh Generation**: On-the-fly 3D proxy generation via Objaverse-XL retrieval and TripoSR
- **Language-Driven Segmentation**: SAM-3 integration for text-prompt-based, occlusion-robust target segmentation
- **Dynamic Target Switching**: Seamless mid-task object switching via natural language without reinitialization
- **Hierarchical Mesh Acquisition**: Three-tier system (benchmark CAD → Objaverse retrieval → TripoSR generation)
- **Real-Time Tracking**: Render-and-compare architecture with transformer-based pose refinement
- **Multi-Object Support**: Concurrent tracking of multiple objects with individual semantic labels

---

## Problem Statement

Traditional 6D pose estimation methods face critical limitations:

1. **CAD Model Dependency**: Require pre-provided 3D models
2. **Manual Annotation**: Need manual mask annotation in initial frames
3. **Occlusion Failures**: Poor performance in heavily occluded scenes (especially LineMOD dataset)
4. **No Language Interface**: Cannot switch between objects using natural language

Our system addresses all these limitations through open-vocabulary, language-driven pose estimation.

---

## Technical Architecture

### Stage 1: Semantic Scene Analysis
- **Moondream2 VLM** generates comprehensive object inventory from RGB stream
- Produces discrete candidate list (e.g., "red bottle," "blue cup," "black keyboard")
- **Gemini API** fallback for semantic query enhancement
- Enables prompt-based object specification for unseen objects

### Stage 2: On-the-Fly 3D Mesh Generation

Hierarchical mesh acquisition strategy:
1. **Primary**: Load ground-truth CAD models (benchmark datasets)
2. **Retrieval**: Query Objaverse-XL database via language-guided similarity search
3. **Generation**: TripoSR generates candidate mesh from single observed image
4. **Selection**: Mesh manager scores candidates (silhouette, depth, IoU alignment)

**Mesh Dictionary**: Asynchronous fetching and caching eliminates redundant queries

### Stage 3: Language-Driven Segmentation
- **SAM-3 Integration**: Natural language prompts → pixel-level masks
- **Occlusion Robustness**: Outperforms traditional R-CNN detectors in clutter
- **Temporal Consistency**: Video tracking with frame-to-frame coherence
- **Multi-Object Support**: Simultaneous segmentation via distinct text prompts

### Stage 4: Unified 6D Pose Estimation & Tracking
- **Render-and-Compare**: FoundationPose aligns retrieved mesh with video observation
- **Pose Scoring**: Composite scoring (IoU + Depth + Silhouette)
- **Iterative Refinement**: Transformer-based pose refinement
- **Dynamic Switching**: Instant mask + mesh updates for seamless target transitions

---

## Results & Performance

### Overall Performance (44 Evaluations, 12 Scenes, 18 Objects, 787 Frames)

| Metric | Mean ± Std |
|--------|------------|
| **ADD AUC** | **71.61% ± 39.10%** |
| **ADD-S AUC** | **88.31% ± 28.56%** |
| **Rotation Error** | **44.23° ± 58.38°** |
| **Translation Error** | **2.44cm ± 6.13cm** |
| **Processing Time** | **13.08s ± 0.15s** |

### Best Performing Objects (Top 5 by ADD-S AUC)

| Object | Scenes | ADD AUC | ADD-S AUC | Rotation Error | Translation Error |
|--------|--------|---------|-----------|----------------|-------------------|
| **Pudding Box** | 1 | 100.0% | **100.0%** | 2.7° | 0.24cm |
| **Power Drill** | 4 | 100.0% | **100.0%** | 2.0° ± 0.4° | 0.24cm ± 0.06cm |
| **Bleach Cleanser** | 3 | 100.0% | **100.0%** | 2.4° ± 0.8° | 0.34cm ± 0.01cm |
| **Banana** | 2 | 100.0% | **100.0%** | 5.3° ± 0.4° | 0.40cm ± 0.04cm |
| **Mustard Bottle** | 2 | 90.0% ± 14.0% | **100.0%** | 19.7° ± 25.6° | 0.22cm ± 0.01cm |

### Runtime Performance
- **Average Processing Time**: 13.08 seconds/frame
- **SAM-3 Contribution**: ~12.5 seconds (95.6% of total)
- **FoundationPose Tracking**: ~0.58 seconds (near real-time)

### Comparison with State-of-the-Art

| Method | Zero-Shot Objects | Text Prompt | ADD-S AUC (%) |
|--------|------------------|-------------|---------------|
| PoseCNN | No | No | 75.4 |
| DenseFusion | No | No | 82.3 |
| FoundationPose | Yes | No | 89.2 |
| **Ours (SAM3+FP)** | **Yes** | **Yes** | **88.31 ± 28.56** |

**Unique Contribution**: Only method combining zero-shot capability with text-prompted segmentation for dynamic, interactive pose tracking.

---

## Technologies Used

### Core Frameworks
- **FoundationPose** (CVPR 2024) - Render-and-compare pose estimation
- **SAM-3** - Segment Anything Model 3 for semantic segmentation
- **Moondream2** - Lightweight VLM for edge devices
- **TripoSR** - Single-image to 3D mesh generation
- **Objaverse-XL** - 10M+ 3D asset database

### Deep Learning Stack
- **PyTorch** 2.0/2.7
- **CUDA** 11.8/12.6
- **NVDiffRast** - Differentiable rendering

### Datasets
- **YCB-Video**: 21 objects, 92 sequences
- **LineMOD**: 13 texture-less objects

---

## Project Structure

```
pose_estimation/
├── Code/
│   ├── DenseFusion_mod/          # DenseFusion integration (ablation study)
│   ├── Florence2/                # Florence2 VLM experiments
│   ├── MoonDream2/               # Moondream2 VLM + Objaverse mesh retrieval
│   │   ├── objects/              # Retrieved 3D meshes by category
│   │   ├── get_object_labels.py  # Scene understanding
│   │   └── get_objects_labels_and_meshes_xl.py  # End-to-end pipeline
│   ├── SAM3/                     # SAM-3 segmentation integration
│   ├── SAM-6D/                   # SAM-6D baseline comparison
│   ├── TripoSR/                  # Single-image 3D reconstruction
│   ├── scripts/                  # Main pipeline scripts
│   │   ├── main_pipeline.py      # Unified tracking pipeline
│   │   ├── run_ycbv_sam3_integration.py  # YCB-V evaluation
│   │   ├── run_linemod_sam3.py   # LineMOD evaluation
│   │   ├── scene_analyzer.py     # Semantic scene analysis
│   │   ├── sam3_mask_service.py  # SAM-3 mask generation service
│   │   └── scale_estimator.py    # Mesh scale alignment
│   └── video_process/            # Video frame extraction & evaluation
├── Report/
│   └── CV_Project_Final.pdf      # Comprehensive project report
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ / 12.6+
- PyTorch 2.0+
- 16GB+ GPU RAM (recommended)

### Setup
```bash
# Clone repository
git clone https://github.com/het915/pose_estimation.git
cd pose_estimation

# Install FoundationPose (follow official instructions)
# https://github.com/NVlabs/FoundationPose

# Install SAM-3
pip install segment-anything-3

# Install TripoSR
pip install triposr

# Install Moondream2
pip install moondream

# Install additional dependencies
pip install objaverse torch torchvision trimesh opencv-python
```

---

## Usage

### End-to-End Pipeline

```bash
# Run on YCB-Video dataset with text prompt
python Code/scripts/run_ycbv_sam3_integration.py \
    --scene_id 48 \
    --prompt "red bottle"

# Run on LineMOD dataset
python Code/scripts/run_linemod_sam3.py \
    --object_id 1 \
    --prompt "the target object"
```

### Individual Components

#### 1. Semantic Scene Analysis (Moondream2)
```bash
python Code/MoonDream2/get_object_labels.py \
    --image_path /path/to/image.jpg
```

#### 2. Mesh Retrieval (Objaverse-XL)
```bash
python Code/MoonDream2/get_objects_labels_and_meshes_xl.py \
    --image_path /path/to/image.jpg \
    --output_dir ./meshes
```

#### 3. SAM-3 Segmentation
```bash
python Code/SAM3/seg_obj.py \
    --image_path /path/to/image.jpg \
    --prompt "red bottle"
```

#### 4. TripoSR 3D Generation
```bash
python Code/TripoSR/run.py \
    --image_path /path/to/image.jpg \
    --output_path ./output.obj
```

---

## Challenges and Solutions

### Challenge 1: Mesh Quality vs. Availability
**Problem**: Direct 3D reconstruction produced low-quality meshes with artifacts
**Solution**: Shifted to Objaverse-XL retrieval-based strategy for clean geometry

### Challenge 2: Symmetric Object Ambiguity
**Problem**: Cylindrical objects exhibit rotational symmetry
**Solution**: Adopted ADD-S metric for pose equivalence (achieved 99% ADD-S on symmetric objects)

### Challenge 3: Real-Time Performance
**Problem**: SAM-3 mask generation dominates processing time (~12.5s/frame)
**Solution**: Modular architecture allows parallel optimization; FoundationPose achieves ~0.58s tracking

### Challenge 4: Retrieved Mesh Mismatch
**Problem**: Objaverse meshes provide category-level templates
**Solution**: Depth-based scale estimator + composite scoring (IoU + Depth + Silhouette)

---

## Future Work

- **Performance Optimization**: GPU-accelerated SAM-3, asynchronous mask generation
- **Enhanced Mesh Generation**: Text-to-3D models (Shap-E) for higher-fidelity proxies
- **Temporal Coherence**: Explicit motion models for smoother tracking
- **Multi-View Fusion**: Leverage multiple cameras for improved occlusion handling
- **BOP Challenge Submission**: Formal evaluation on full benchmark suite

---

## Citation

If you use this work, please cite:

```bibtex
@misc{patel2025openworldpose,
  title={Open-World Semantic-Based Zero-Shot 6D Pose Estimation Using SAM3 And FoundationPose},
  author={Patel, Het and Deshpande, Sunny and Bhansali, Ansh and Ogawa, Keisuke},
  year={2025},
  institution={University of Illinois Urbana-Champaign},
  course={CS543 Computer Vision}
}
```

---

## Acknowledgments

- **FoundationPose**: [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose)
- **SAM-3**: Meta AI's Segment Anything Model
- **Moondream2**: [vikhyat/moondream](https://github.com/vikhyat/moondream)
- **TripoSR**: [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR)
- **Objaverse-XL**: [allenai/objaverse-xl](https://github.com/allenai/objaverse-xl)

---

## License

MIT License - See [LICENSE](LICENSE) for details

---

## Contact

For questions or collaborations:
- **Het Patel**: hcp4@illinois.edu
- **Course**: CS543 Computer Vision, UIUC
- **Project Date**: December 2025

---

<p align="center">
  <i>Built with PyTorch, CUDA, and a passion for computer vision</i>
</p>
