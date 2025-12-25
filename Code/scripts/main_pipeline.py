#!/usr/bin/env python3
"""
Complete Pipeline: Moondream → Objaverse → SAM3 → FoundationPose
Supports YCB-V dataset with automatic scene-object mapping
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import numpy as np
import cv2
import trimesh
from pathlib import Path
import time

# FoundationPose imports
from estimater import *
from datareader import YcbVideoReader
from Utils import *

# Local modules
from mesh_manager import MeshManager
from scale_estimator import MeshScaleEstimator

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OBJECT_NAMES = {
    1: "master chef can", 2: "cracker box", 3: "sugar box",
    4: "tomato soup can", 5: "mustard bottle", 6: "tuna fish can",
    7: "pudding box", 8: "gelatin box", 9: "potted meat can",
    10: "banana", 11: "pitcher base", 12: "bleach cleanser",
    13: "bowl", 14: "mug", 15: "power drill",
    16: "wood block", 17: "scissors", 18: "large marker",
    19: "large clamp", 20: "extra large clamp", 21: "foam brick"
}

class MoondreamBridge:
    """Bridge to Moondream detection service"""
    
    def __init__(self, env_name="moondream", script_path="~/moondream/moondream_detection_service.py"):
        self.env = env_name
        self.script = os.path.expanduser(script_path)
    
    def detect(self, image_path, get_boxes=False):
        """Detect objects in image"""
        cmd = ['conda', 'run', '-n', self.env, 'python', self.script,
               '--image', image_path]
        
        if get_boxes:
            cmd.append('--boxes')
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            logging.error(f"Moondream failed: {result.stderr}")
            return None
        
        return json.loads(result.stdout)

class SAM3Bridge:
    """Bridge to SAM3 segmentation service"""
    
    def __init__(self, env_name="sam3", script_path="~/sam3/sam3_mask_service.py"):
        self.env = env_name
        self.script = os.path.expanduser(script_path)
    
    def segment(self, image_path, prompt, threshold=0.5):
        """Generate segmentation mask"""
        import tempfile
        from PIL import Image
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            mask_path = tmp.name
        
        try:
            cmd = ['conda', 'run', '-n', self.env, 'python', self.script,
                   '--image', image_path,
                   '--prompt', prompt,
                   '--output', mask_path,
                   '--threshold', str(threshold)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                return None, 0.0
            
            score = float(result.stdout.strip().split('\n')[-1])
            
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path)) > 127
                os.remove(mask_path)
                return mask, score
            
            return None, 0.0
            
        except Exception as e:
            logging.error(f"SAM3 error: {e}")
            return None, 0.0

class PipelineExecutor:
    """Main pipeline execution"""
    
    def __init__(self, args):
        self.args = args
        
        # Set environment
        os.environ["YCB_VIDEO_DIR"] = args.ycbv_dir
        
        # Initialize components
        logging.info("Initializing pipeline components...")
        
        self.moondream = MoondreamBridge() if args.use_moondream else None
        self.sam3 = SAM3Bridge()
        self.mesh_manager = MeshManager(cache_dir=args.mesh_cache_dir)
        
        # FoundationPose setup
        set_logging_format()
        set_seed(0)
        
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        
        # Output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_single_scene(self, scene_id, object_id):
        """Process single scene-object pair"""
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Scene {scene_id}, Object {object_id}")
        logging.info(f"{'='*60}")
        
        # Load scene
        scene_path = Path(self.args.ycbv_dir) / "test" / f"{scene_id:06d}"
        reader = YcbVideoReader(str(scene_path), zfar=1.5)
        
        # Get object name
        object_name = OBJECT_NAMES.get(object_id, "object")
        logging.info(f"Object: {object_name}")
        
        # Step 1: Get meshes
        logging.info("\n[STEP 1] Retrieving meshes...")
        
        precomputed_mesh_path = None
        if self.args.use_precomputed_mesh:
            precomputed_mesh_path = f"{self.args.ycbv_dir}/models/obj_{object_id:06d}.ply"
        
        mesh_data = self.mesh_manager.get_meshes_for_label(
            label=object_name,
            max_meshes=self.args.num_mesh_candidates,
            use_precomputed=self.args.use_precomputed_mesh,
            precomputed_path=precomputed_mesh_path
        )
        
        candidate_meshes = mesh_data['meshes']
        mesh_metadata = mesh_data['metadata']
        
        if len(candidate_meshes) == 0:
            logging.error("No meshes found!")
            return None
        
        logging.info(f"✓ Got {len(candidate_meshes)} mesh candidates")
        
        # Process frames
        results = []
        
        for frame_idx in range(min(self.args.num_frames, len(reader.color_files))):
            logging.info(f"\n{'─'*60}")
            logging.info(f"Frame {frame_idx}")
            logging.info(f"{'─'*60}")
            
            # Load frame data
            color = reader.get_color(frame_idx)
            depth = reader.get_depth(frame_idx)
            K = reader.K
            
            # Get ground truth (if available)
            gt_pose = None
            try:
                gt_pose = reader.get_gt_pose(frame_idx, object_id)
            except:
                pass
            
            # Step 2: Segmentation
            logging.info("[STEP 2] Segmentation...")
            
            # Use SAM3 with object name as prompt
            mask, sam3_score = self.sam3.segment(
                reader.color_files[frame_idx],
                object_name,
                self.args.sam3_threshold
            )
            
            if mask is None or sam3_score < 0.01:
                logging.warning("Segmentation failed, skipping frame")
                continue
            
            logging.info(f"✓ Mask generated (score: {sam3_score:.3f})")
            
            # Step 3: Scale estimation
            logging.info("[STEP 3] Scale estimation...")
            
            scaled_meshes = MeshScaleEstimator.estimate_scale_multiple_meshes(
                candidate_meshes, mask, depth, K
            )
            
            # Step 4: Pose estimation for each mesh
            logging.info(f"[STEP 4] Pose estimation ({len(scaled_meshes)} candidates)...")
            
            best_result = None
            best_score = -np.inf
            
            for mesh_result in scaled_meshes:
                mesh_idx = mesh_result['mesh_index']
                scaled_mesh = mesh_result['scaled_mesh']
                scale_factor = mesh_result['scale_factor']
                
                logging.info(f"  Trying mesh {mesh_idx} (scale: {scale_factor:.3f})...")
                
                # Initialize FoundationPose for this mesh
                est = FoundationPose(
                    model_pts=scaled_mesh.vertices.copy(),
                    model_normals=scaled_mesh.vertex_normals.copy(),
                    mesh=scaled_mesh,
                    scorer=self.scorer,
                    refiner=self.refiner,
                    debug_dir=str(self.output_dir),
                    debug=0,
                    glctx=self.glctx
                )
                
                # Estimate pose
                try:
                    if frame_idx == 0:
                        pose = est.register(
                            K=K, rgb=color, depth=depth,
                            ob_mask=mask, iteration=self.args.est_refine_iter
                        )
                    else:
                        pose = est.track_one(
                            rgb=color, depth=depth, K=K, iteration=2
                        )

                    # Score this result with detailed breakdown
                    scores = MeshScaleEstimator.score_mesh_pose(
                        pose, scaled_mesh, color, depth, mask, K, glctx=self.glctx
                    )
                    score = scores['total']

                    logging.info(f"    Total: {score:.4f} | IoU: {scores['mask_iou']:.3f} | "
                               f"Depth: {scores['depth_agreement']:.3f} | "
                               f"Silhouette: {scores['silhouette_match']:.3f}")

                    if score > best_score:
                        best_score = score
                        best_result = {
                            'pose': pose,
                            'mesh': scaled_mesh,
                            'mesh_index': mesh_idx,
                            'scale_factor': scale_factor,
                            'score': score,
                            'score_details': scores,
                            'metadata': mesh_metadata[mesh_idx]
                        }

                except Exception as e:
                    logging.warning(f"    Failed: {e}")
                    continue
            
            if best_result is None:
                logging.warning("All mesh candidates failed")
                continue
            
            logging.info(f"✓ Best: mesh {best_result['mesh_index']} (score: {best_result['score']:.4f})")
            
            # Step 5: Evaluate (if GT available)
            frame_result = {
                'frame': frame_idx,
                'pose': best_result['pose'],
                'mesh_index': best_result['mesh_index'],
                'scale_factor': best_result['scale_factor'],
                'score': best_result['score'],
                'score_details': best_result['score_details'],
                'sam3_score': sam3_score
            }
            
            if gt_pose is not None:
                # Compute metrics
                diameter = np.linalg.norm(best_result['mesh'].extents)
                
                add_score, add_auc = self._compute_ADD(
                    best_result['pose'], gt_pose,
                    best_result['mesh'].vertices, diameter
                )
                
                frame_result.update({
                    'has_gt': True,
                    'add_score_mm': add_score * 1000,
                    'add_auc': add_auc
                })
                
                logging.info(f"📊 ADD: {add_score*1000:.2f}mm, AUC: {add_auc*100:.1f}%")
            else:
                frame_result['has_gt'] = False
            
            results.append(frame_result)
            
            # Save visualization
            if self.args.save_visualizations:
                self._save_visualization(
                    frame_idx, scene_id, object_id,
                    color, best_result['pose'],
                    best_result['mesh'], K, mask
                )
        
        # Save results
        self._save_results(scene_id, object_id, results)
        
        return results
    
    def _score_pose(self, pose, rgb, depth, mask, mesh, K):
        """Score pose using comprehensive mesh evaluation"""
        try:
            scores = MeshScaleEstimator.score_mesh_pose(
                pose, mesh, rgb, depth, mask, K, glctx=self.glctx
            )
            return scores['total']
        except Exception as e:
            logging.warning(f"Scoring failed: {e}")
            return 0.0
    
    def _compute_ADD(self, pose_pred, pose_gt, model_points, diameter):
        """Compute ADD metric"""
        R_pred, t_pred = pose_pred[:3, :3], pose_pred[:3, 3]
        R_gt, t_gt = pose_gt[:3, :3], pose_gt[:3, 3]
        
        pts_pred = (R_pred @ model_points.T).T + t_pred
        pts_gt = (R_gt @ model_points.T).T + t_gt
        
        distances = np.linalg.norm(pts_pred - pts_gt, axis=1)
        add_score = distances.mean()
        
        threshold = 0.1 * diameter
        add_auc = (distances < threshold).mean()
        
        return add_score, add_auc
    
    def _save_visualization(self, frame_idx, scene_id, obj_id, 
                           color, pose, mesh, K, mask):
        """Save visualization"""
        vis_dir = self.output_dir / f"scene_{scene_id:06d}_obj_{obj_id:02d}" / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Overlay mask
        mask_overlay = color.copy()
        mask_overlay[mask] = mask_overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
        
        # Draw 3D bbox
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
        center_pose = pose @ np.linalg.inv(to_origin)
        
        vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, 
                          K=K, thickness=3, is_input_rgb=True)
        
        cv2.imwrite(str(vis_dir / f"frame_{frame_idx:04d}.png"), vis[..., ::-1])
    
    def _save_results(self, scene_id, obj_id, results):
        """Save results JSON"""
        output_file = self.output_dir / f"results_scene_{scene_id:06d}_obj_{obj_id:02d}.json"

        # Convert numpy to python types
        serializable_results = []
        for r in results:
            sr = {
                'frame': r['frame'],
                'pose': r['pose'].tolist(),
                'mesh_index': int(r['mesh_index']),
                'scale_factor': float(r['scale_factor']),
                'score': float(r['score']),
                'score_details': {
                    k: float(v) for k, v in r['score_details'].items()
                },
                'sam3_score': float(r['sam3_score']),
                'has_gt': r['has_gt']
            }
            if r['has_gt']:
                sr['add_score_mm'] = float(r['add_score_mm'])
                sr['add_auc'] = float(r['add_auc'])

            serializable_results.append(sr)

        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logging.info(f"✓ Saved results to {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description='Complete Pipeline: Moondream → Objaverse → SAM3 → FoundationPose')
    
    # Dataset
    parser.add_argument('--ycbv_dir', default='/home/hcp4/FoundationPose/datasets/ycbv')
    parser.add_argument('--scene_id', type=int, help='Specific scene to process')
    parser.add_argument('--object_id', type=int, help='Specific object to process')
    parser.add_argument('--num_frames', type=int, default=10)
    
    # Mesh retrieval
    parser.add_argument('--use_moondream', action='store_true', help='Use Moondream for detection')
    parser.add_argument('--use_precomputed_mesh', action='store_true', help='Try YCB-V ground truth mesh first')
    parser.add_argument('--num_mesh_candidates', type=int, default=5, help='Number of Objaverse meshes to try')
    parser.add_argument('--mesh_cache_dir', default='./mesh_cache')
    
    # Segmentation
    parser.add_argument('--sam3_threshold', type=float, default=0.5)
    
    # Pose estimation
    parser.add_argument('--est_refine_iter', type=int, default=5)
    
    # Output
    parser.add_argument('--output_dir', default='./pipeline_results')
    parser.add_argument('--save_visualizations', action='store_true')
    
    # Batch mode
    parser.add_argument('--batch_all', action='store_true', help='Process all scenes and objects')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    pipeline = PipelineExecutor(args)
    
    if args.batch_all:
        # Load scene mapping
        with open('scene_object_mapping.json') as f:
            mapping = json.load(f)
        
        for scene_id, obj_ids in mapping['scene_to_objects'].items():
            for obj_id in obj_ids:
                try:
                    pipeline.run_single_scene(int(scene_id), obj_id)
                except Exception as e:
                    logging.error(f"Failed scene {scene_id} obj {obj_id}: {e}")
    
    elif args.scene_id and args.object_id:
        pipeline.run_single_scene(args.scene_id, args.object_id)
    
    else:
        logging.error("Specify --scene_id and --object_id, or use --batch_all")

if __name__ == '__main__':
    main()