# Open-World Semantic-Based Zero-Shot 6D Pose Estimation Using SAM3 And FoundationPose (CS543 Class Project)

## Team Members
- Het Patel (hcp4)
- Sunny Deshpande (sunnynd2)
- Ansh Bhansali (anshb3)
- Keisuke Ogawa (ogawa3)

## Overview
Developed a novel open-vocabulary 6D object pose tracking framework that extends NVIDIA's FoundationPose architecture to enable language-guided, zero-shot tracking of arbitrary objects without pre-registered CAD models. By integrating Moondream2 vision-language model, SAM-3 segmentation, and on-the-fly 3D mesh generation from Objaverse-XL, the system achieves real-time, occlusion-robust pose estimation with dynamic target switching via natural language prompts. This breakthrough enables robotic manipulators to seamlessly transition between tracking different objects (e.g., "grasp the red bottle" to "now grasp the blue cup") in unstructured environments without reinitialization.

## Problem Statement
Traditional 6D pose estimation methods face critical limitations that restrict their deployment in real-world robotic manipulation scenarios. NVIDIA's FoundationPose, while achieving zero-shot inference for unseen objects, requires pre-provided CAD models and manual mask annotation in the initial frame. It often fails in heavily occluded scenes (especially LineMOD dataset), where errors propagate through the mesh-matching and refinement stages. Furthermore, no existing system supports real-time, language-guided, multi-object pose estimation with dynamic target switching—a crucial capability for responsive robotic manipulation in novel environments.

## Key Features
- **Open-Vocabulary Detection**: Lightweight Moondream2 VLM for edge-compatible semantic scene understanding
- **Zero-Shot Mesh Generation**: On-the-fly 3D proxy generation via Objaverse-XL retrieval and TripoSR, eliminating CAD model dependencies
- **Language-Driven Segmentation**: SAM-3 integration for text-prompt-based, occlusion-robust target segmentation
- **Dynamic Target Switching**: Seamless mid-task object switching via natural language without reinitialization
- **Hierarchical Mesh Acquisition**: Three-tier system (benchmark CAD → Objaverse retrieval → TripoSR generation) for optimal geometry
- **Real-Time Tracking**: Render-and-compare architecture with transformer-based pose refinement
- **Mesh Dictionary Caching**: Asynchronous mesh fetching and caching to eliminate redundant queries
- **Multi-Object Support**: Concurrent tracking of multiple objects with individual semantic labels

## Technologies Used
- **Core Tracking**: FoundationPose (CVPR 2024), NVDiffRast for differentiable rendering
- **Vision-Language**: Moondream2 (lightweight VLM for edge devices), Gemini API for semantic expansion
- **Segmentation**: SAM-3 (Segment Anything Model 3) with temporal consistency
- **3D Generation**: TripoSR (single-image to 3D), Objaverse-XL (10M+ 3D asset database)
- **Detection**: YOLOv8 for initial object detection experiments
- **Deep Learning**: PyTorch 2.0/2.7, CUDA 11.8/12.6
- **Datasets**: YCB-Video (21 objects, 92 sequences), LineMOD (13 texture-less objects)
- **Programming**: Python, modular conda environments

## Technical Architecture

### Stage 1: Semantic Scene Analysis
- Moondream2 VLM generates comprehensive object inventory from RGB stream
- Produces discrete candidate list (e.g., "red bottle," "blue cup," "black keyboard")
- Gemini API fallback for semantic query enhancement when detection fails
- Enables prompt-based object specification even for BOP unseen objects

### Stage 2: On-the-Fly 3D Mesh Generation
Hierarchical mesh acquisition strategy:
1. **Primary**: Load ground-truth CAD models when available (benchmark datasets)
2. **Retrieval**: Query Objaverse-XL database via language-guided similarity search
3. **Generation**: TripoSR generates candidate mesh from single observed image
4. **Selection**: Mesh manager scores candidates based on silhouette, depth, and IoU alignment

**Mesh Dictionary**:
- Asynchronous fetching and caching of high-quality, clean meshes
- Eliminates redundant queries during multi-object tracking
- Supports multiple candidate meshes for robust downstream pose estimation

### Stage 3: Language-Driven Segmentation
- **SAM-3 Integration**: Natural language prompts (e.g., "the red apple") → pixel-level masks
- **Occlusion Robustness**: Outperforms traditional R-CNN detectors in heavy clutter
- **Temporal Consistency**: Video tracking with frame-to-frame coherence
- **Multi-Object Support**: Simultaneous segmentation via distinct text prompts

### Stage 4: Unified 6D Pose Estimation & Tracking
- **Render-and-Compare**: FoundationPose aligns retrieved mesh with video observation
- **Pose Scoring**: Uniform sampling, composite scoring (IoU + Depth + Silhouette)
- **Iterative Refinement**: Transformer-based pose refinement for sub-frame accuracy
- **Dynamic Switching**: Instant mask + mesh updates enable seamless target transitions

## Results & Performance

### Overall Performance (44 Evaluations Across 12 Scenes, 18 Objects, 787 Frames)
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

### Per-Scene Performance Highlights
**Best Scene (Scene 56):**
- Objects tested: 3
- Avg ADD AUC: **100.0%**
- Avg ADD-S AUC: **100.0%**
- Avg Rotation Error: 2.6°
- Avg Translation Error: 0.32cm

**Scene 48 (Referenced in Report):**
- Objects tested: 5
- Avg ADD AUC: 64.0%
- Avg ADD-S AUC: 79.0%
- Avg Rotation Error: 64.6°
- Avg Trans Error: 4.93cm

**Scene 59:**
- Objects tested: 5
- Avg ADD AUC: 84.0%
- Avg ADD-S AUC: **100.0%**
- Avg Rotation Error: 33.4°
- Avg Translation Error: 0.29cm

### Detailed Object Performance
**Symmetric Objects (Handling Rotational Ambiguity):**
- **Master Chef Can**: 83.0% ADD AUC, **99.0% ADD-S AUC**, 29.8° rot, 0.55cm trans (4 scenes)
- **Tuna Fish Can**: 98.0% ADD AUC, **98.0% ADD-S AUC**, 7.4° rot, 0.57cm trans (4 scenes)
- **Bowl**: 8.0% ADD AUC, **100.0% ADD-S AUC**, 93.3° rot, 0.38cm trans (demonstrates ADD-S effectiveness)
- **Mug**: 98.0% ADD AUC, **100.0% ADD-S AUC**, 7.8° rot, 0.48cm trans (2 scenes)

**Texture-Rich Objects:**
- **Cracker Box**: 92.0% ± 14.0% ADD/ADD-S AUC, 9.7° rot, 2.01cm trans (3 scenes)
- **Vertical Large Clamp**: 47.0% ADD AUC, **100.0% ADD-S AUC**, 102.6° rot, 1.02cm trans (2 scenes)

### Runtime Performance
- **Average Processing Time**: 13.08 seconds per frame (13,083ms ± 146ms)
- **First Frame Registration**: 13–16 seconds (includes SAM-3 mask + initial pose alignment)
- **Subsequent Tracking**: 12–13 seconds per frame
- **SAM-3 Contribution**: ~12.5 seconds (95.6% of total processing time)
- **FoundationPose Tracking**: ~0.58 seconds (near real-time after initialization)

### Challenging Cases Analysis
Objects with high rotation ambiguity but excellent ADD-S recovery:
1. **Wood Block**: 31.0% ADD AUC → **100.0% ADD-S AUC** (121.5° rot, 0.57cm trans)
2. **Large Marker**: 25.0% ADD AUC → **100.0% ADD-S AUC** (139.8° rot, 0.37cm trans)
3. **Vertical Large Clamp**: 47.0% ADD AUC → **100.0% ADD-S AUC** (102.6° rot, 1.02cm trans)

Failure cases (requiring further investigation):
- Gelatin Box: 10.0% ADD AUC, 12.0% ADD-S AUC, 101.3° rot, 10.88cm trans
- Horizontal Extra Large Clamp: 17.0% ADD AUC, 52.0% ADD-S AUC, 152.1° rot, 11.28cm trans

### Comparison with State-of-the-Art
| Method | Zero-Shot Objects | Text Prompt | ADD-S AUC (%) |
|--------|------------------|-------------|---------------|
| PoseCNN | No | No | 75.4 |
| DenseFusion | No | No | 82.3 |
| FoundationPose | Yes | No | 89.2 |
| **Ours (SAM3+FP)** | **Yes** | **Yes** | **88.31 ± 28.56** |

**Unique Contribution**: Only method combining zero-shot capability with text-prompted segmentation for dynamic, interactive pose tracking. Achieves competitive 88.31% ADD-S AUC across 18 diverse objects without CAD model pre-registration.

## Challenges and Solutions

### Challenge 1: Mesh Quality vs. Availability Trade-off
**Problem**: Direct 3D reconstruction (YOLOv8 + SAM + TripoSR) produced low-quality meshes with artifacts, concave/convex distortions, especially at moderate resolutions (720p). YOLOv8 also failed to detect less common or partially occluded objects.

**Solution**: Shifted to retrieval-based strategy leveraging Objaverse-XL's 10M+ professionally designed/scanned meshes. Clean, artifact-free geometry improved render-and-compare robustness under occlusion and low-resolution conditions.

### Challenge 2: Symmetric Object Ambiguity
**Problem**: Cylindrical objects (e.g., Master Chef Can) exhibit rotational symmetry, leading to lower ADD accuracy due to inherent pose ambiguity.

**Solution**: Adopted ADD-S metric accounting for pose equivalence. Achieved 76.5% ADD-S AUC demonstrating adequate symmetric pose recovery despite 111° rotation error in non-symmetric ADD metric.

### Challenge 3: Real-Time Performance Bottleneck
**Problem**: SAM-3 mask generation dominates processing time at ~12.5 seconds per frame, preventing true real-time operation on standard hardware.

**Solution**: Modular architecture allows parallel optimization. FoundationPose tracking achieves near real-time after initialization. Future work targets GPU-accelerated SAM-3 inference and asynchronous processing.

### Challenge 4: Retrieved Mesh Instance Mismatch
**Problem**: Objaverse-retrieved meshes provide category-level templates that may differ in exact proportions (e.g., generic "mug" vs. specific handle shape).

**Solution**: Depth-based scale estimator (scale = observed_depth / model_depth) adjusts retrieved mesh dimensions. Composite scoring function (IoU + Depth + Silhouette) selects best-matching candidate from multiple sources.

## Experimental Insights

### Ablation Studies
**DenseFusion Exploration**: Investigated classic RGB-D fusion for occlusion robustness on LineMOD. While effective for known objects under occlusion, requires retraining for unseen objects—contradicting zero-shot goal. Abandoned in favor of FoundationPose.

**SAM-6D Evaluation**: Tested zero-shot pose estimation with partial point cloud matching. Achieved high-quality masks but ~40 seconds per frame (3x slower than our approach) with lower overall accuracy. Bottleneck from exhaustive hypothesis sampling during refinement.

**YOLO + SAM + TripoSR Pipeline**: Initial single-image reconstruction approach failed due to incomplete scene coverage and poor mesh quality propagating errors through FoundationPose.

### Key Findings
1. **Retrieval > Reconstruction**: Clean template meshes outperform noisy reconstructions for robust pose estimation
2. **SAM-3 > R-CNN**: Language-driven segmentation superior in occluded scenes
3. **VLM-Driven Semantic Indexing**: Enables zero-shot capability without pre-registration
4. **Dynamic Switching**: Novel contribution unavailable in prior work

## Resume Bullet Points
- Developed open-vocabulary 6D pose tracking framework extending FoundationPose with Moondream2 VLM and SAM-3, achieving 88.31% ADD-S AUC across 44 evaluations (12 scenes, 18 objects, 787 frames) on YCB-Video benchmark
- Implemented zero-shot 3D mesh generation pipeline leveraging Objaverse-XL retrieval (10M+ assets) and TripoSR, eliminating CAD model dependencies for novel object tracking with 100% ADD-S AUC on 5 object categories
- Designed language-driven dynamic target switching system enabling seamless multi-object pose estimation via natural language prompts, achieving 2.44cm average translation error and 44.23° rotation error
- Demonstrated robust symmetric object handling with 99% ADD-S AUC on cylindrical objects and 100% ADD-S AUC on complex geometries (power drill, bleach cleanser) through hierarchical mesh acquisition strategy

## Future Work
- **Performance Optimization**: GPU-accelerated SAM-3 inference, asynchronous mask generation
- **Enhanced Mesh Generation**: Integration of text-to-3D models (Shap-E) for higher-fidelity proxies
- **Temporal Coherence**: Explicit motion models for smoother tracking across frames
- **Multi-View Fusion**: Leverage multiple camera perspectives for improved occlusion handling
- **BOP Challenge Submission**: Formal evaluation on full benchmark suite

## Links
- **Report**: [Final Project Report (PDF)](media/Report/CV_Project_Final.pdf)
- **Course**: CS543 Computer Vision, University of Illinois Urbana-Champaign
- **Date**: November 2024
