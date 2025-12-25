import os
import os.path as osp
import sys
import argparse
import numpy as np
import imageio
import cv2
import json
import torch 
import glob
import trimesh
import random
import importlib
from tqdm import tqdm
from PIL import Image
import shutil

# --- Configure BASE_DIR and sys.path ---
# BASE_DIR is the project root (where run_sequence_inference.py resides)
BASE_DIR = osp.dirname(osp.abspath(__file__))

# CRITICAL FIX: Add the project root to sys.path first.
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
    
# Add the main model directories for hierarchical package imports
sys.path.append(osp.join(BASE_DIR, 'Instance_Segmentation_Model'))
sys.path.append(osp.join(BASE_DIR, 'Instance_Segmentation_Model/model'))

# --- CRITICAL PEM PATH ADDITIONS ---
# 1. Add the Pose_Estimation_Model root (needed for top-level modules like utils/data_utils)
if osp.join(BASE_DIR, 'Pose_Estimation_Model') not in sys.path:
    sys.path.append(osp.join(BASE_DIR, 'Pose_Estimation_Model'))

# 2. Explicitly add the folder containing 'pose_estimation_model.py' and 'model_utils.py'
if osp.join(BASE_DIR, 'Pose_Estimation_Model/model') not in sys.path:
    sys.path.append(osp.join(BASE_DIR, 'Pose_Estimation_Model/model'))


# --- Imports (Now excluding the utils.inout dependency) ---
try:
    import gorilla
    
    # Imports from Hydra & Pytorch Ecosystem
    from hydra import compose, initialize
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import torchvision.transforms as transforms
    import importlib
    
    # Imports from Segment Anything
    from segment_anything.utils.amg import rle_to_mask
    
    # --- ISM/Instance_Segmentation_Model imports ---
    from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
    from utils.bbox_utils import CropResizePad
    from skimage.feature import canny
    from skimage.morphology import binary_dilation
    from model.utils import Detections 
    import distinctipy
    
    # --- PEM/Pose_Estimation_Model imports ---
    from utils.data_utils import (
        load_im, get_bbox, get_point_cloud_from_depth, get_resize_rgb_choose
    )
    from utils.draw_utils import draw_detections
    
    # --- External Imports ---
    import pycocotools.mask as cocomask
    
except ImportError as e:
    print(f"Failed to import necessary modules: {e}")
    print("FATAL ERROR: Check your project structure, ensure all utility folders (e.g., 'utils') contain an '__init__.py' file, and required packages are installed.")
    sys.exit(1)


# --- Global Configs and Transforms ---

rgb_transform_pem = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])

# --- ISM Visualization and Data Utilities ---
def visualize_ism(rgb, detections_list, save_path="tmp.png"):
    # The input 'rgb' is now guaranteed to be a NumPy array
    
    # CRITICAL FIX: Ensure detections_list is not None or empty
    if not detections_list or (isinstance(detections_list[0], dict) and detections_list[0].get('score', 0) == 0):
        # Fallback: Save the original image (convert array to PIL first)
        img_pil = Image.fromarray(rgb)
        img_pil.save(save_path)
        # Return the original RGB image as a PIL object for further concatenation
        return img_pil 

    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    colors = distinctipy.get_colors(len(detections_list))
    alpha = 0.33

    best_score = 0.
    best_det = None
    # Find the best detection
    for mask_idx, det in enumerate(detections_list):
        # CRITICAL FIX: Check if 'det' is a dictionary before accessing 'score'
        if not isinstance(det, dict) or 'score' not in det:
            continue
            
        if best_score < det['score']:
            best_score = det['score']
            best_det = detections_list[mask_idx]
            
    if best_det is None:
        img_pil = Image.fromarray(rgb)
        img_pil.save(save_path)
        return img_pil

    # Process and draw the best mask
    mask = rle_to_mask(best_det["segmentation"])
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    obj_id = best_det.get("category_id", 1) 
    temp_id = (obj_id - 1) % len(colors) 

    r = int(255*colors[temp_id][0])
    g = int(255*colors[temp_id][1])
    b = int(255*colors[temp_id][2])
    
    img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
    img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
    img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
    img[edge, :] = 255
    
    # Return the processed segmentation image
    img_pil = Image.fromarray(np.uint8(img))
    img_pil.save(save_path)
    return img_pil # Return the visualization image


def batch_input_data_ism(depth_path, cam_path, device):
    batch = {}
    try:
        with open(cam_path, 'r') as f:
            cam_info = json.load(f)
    except Exception as e:
        print(f"Error loading camera info from {cam_path}: {e}")
        return None

    # DeprecationWarning fix: use imageio.v2
    depth = np.array(imageio.v2.imread(depth_path)).astype(np.int32)
    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam_info['depth_scale'])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch


# --- PEM Template and Visualization Utilities ---

# MODIFIED: Accepts the ISM PIL image for concatenation
def visualize_pem(rgb, ism_vis_pil, pred_rot, pred_trans, model_points, K, save_path):
    # 1. Draw 6D Pose on RGB image
    img_pose_np = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
    
    # 2. Prepare images for concatenation and live view
    # ISM Visualization (PIL) -> NumPy BGR
    ism_bgr = np.array(ism_vis_pil)[:,:,::-1]
    
    # Pose Visualization (NumPy RGB) -> NumPy BGR
    img_pose_bgr = img_pose_np[:,:,::-1]
    
    # Original RGB (NumPy RGB) -> NumPy BGR
    rgb_bgr = rgb[:,:,::-1] 
    
    # Concatenate for live view (RGB | ISM | PEM)
    h, w, _ = img_pose_bgr.shape
    combined_img = np.zeros((h, 3*w, 3), dtype=np.uint8)
    combined_img[:, 0:w, :] = rgb_bgr # Original RGB
    combined_img[:, w:2*w, :] = ism_bgr # ISM Segmentation
    combined_img[:, 2*w:3*w, :] = img_pose_bgr # PEM Pose
    
    # NOTE: cv2.imshow is block execution, so we use waitKey(1) for live update
    cv2.imshow("Live SAM-6D Pipeline (RGB | ISM | PEM)", combined_img)
    cv2.waitKey(1) 

    # Save the final concatenated image (PIL object)
    ism_vis_np = np.array(ism_vis_pil)
    img_pose_pil = Image.fromarray(np.uint8(img_pose_np))
    
    concat = Image.new('RGB', (rgb.shape[1] + ism_vis_np.shape[1] + img_pose_pil.size[0], rgb.shape[0]))
    concat.paste(Image.fromarray(np.uint8(rgb)), (0, 0))
    concat.paste(ism_vis_pil, (rgb.shape[1], 0))
    concat.paste(img_pose_pil, (rgb.shape[1] + ism_vis_np.shape[1], 0))
    
    return concat


def _get_template(path, tem_index=1, cfg=None):
    rgb_path = os.path.join(path, 'rgb_'+str(tem_index)+'.png')
    mask_path = os.path.join(path, 'mask_'+str(tem_index)+'.png')
    xyz_path = os.path.join(path, 'xyz_'+str(tem_index)+'.npy')

    rgb = load_im(rgb_path).astype(np.uint8)
    xyz = np.load(xyz_path).astype(np.float32) / 1000.0  
    mask = load_im(mask_path).astype(np.uint8) == 255

    bbox = get_bbox(mask)
    y1, y2, x1, x2 = bbox
    mask = mask[y1:y2, x1:x2]

    rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]
    if cfg.rgb_mask_flag:
        rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

    rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb_transform_pem(np.array(rgb))

    choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
    if len(choose) <= cfg.n_sample_template_point:
        # Fixed: use replace=True if not enough points
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=True) 
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=False)
    choose = choose[choose_idx]
    xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

    rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)
    return rgb, rgb_choose, xyz


# --- PEM Template Loader (Correctly returns lists of Tensors) ---
def get_templates(path, cfg):
    n_template_view = cfg.n_template_view
    all_tem = []
    all_tem_choose = []
    all_tem_pts = []

    total_nView = 42
    for v in range(n_template_view):
        i = int(total_nView / n_template_view * v)
        tem, tem_choose, tem_pts = _get_template(path, i, cfg) 
        
        # Ensure single-item Tensors [1, C, H, W] are added to the list
        all_tem.append(torch.FloatTensor(tem).unsqueeze(0).cuda()) 
        all_tem_choose.append(torch.IntTensor(tem_choose).long().unsqueeze(0).cuda())
        all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).cuda())
        
    # CRITICAL CHANGE: DO NOT CONCATENATE! Return lists of single-view tensors.
    
    return all_tem, all_tem_pts, all_tem_choose # Returns lists of Tensors


# --- PEM Data Loading (MODIFIED for In-Memory Detections) ---
# CRITICAL FIX 3: Change signature to accept list of dictionaries (detections_list)
def get_test_data_pem_in_memory(rgb_path, depth_path, cam_path, cad_path, detections_list, det_score_thresh, cfg):
    """
    Loads PEM data using the pre-processed list of detection dictionaries from ISM.
    """
    
    # 1. Filter Detections based on score threshold
    filtered_dets = []
    
    # Since detections_list is now guaranteed to be a list of dictionaries (or empty if failed), 
    # we can remove all complex unpacking logic and directly filter.
    if isinstance(detections_list, list):
        for det in detections_list:
            # Use .get() defensively against missing keys
            if isinstance(det, dict) and det.get('score', 0) > det_score_thresh:
                filtered_dets.append(det)

    if not filtered_dets:
        return None, None, None, None, []

    # 2. Load Camera and Image Data
    try:
        with open(cam_path, 'r') as f:
            cam_info = json.load(f)
    except:
        print("Error loading camera info.")
        return None, None, None, None, []
        
    K = np.array(cam_info['cam_K']).reshape(3, 3)

    whole_image = load_im(rgb_path).astype(np.uint8)
    if len(whole_image.shape)==2:
        whole_image = np.concatenate([whole_image[:,:,None], whole_image[:,:,None], whole_image[:,:,None]], axis=2)
    
    try:
        # DeprecationWarning fix: use imageio.v2
        raw_depth = imageio.v2.imread(depth_path).astype(np.float32)
    except:
        raw_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        
    whole_depth = raw_depth * cam_info['depth_scale'] / 1000.0 # Convert to meters
    whole_pts = get_point_cloud_from_depth(whole_depth, K)

    # 3. Load CAD Model
    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(cfg.n_sample_model_point).astype(np.float32) / 1000.0
    radius = np.max(np.linalg.norm(model_points, axis=1))


    all_rgb = []
    all_cloud = []
    all_rgb_choose = []
    all_score = []
    all_dets = [] # The final list of dictionaries passed to PEM
    
    # 4. Process each filtered detection
    for inst in filtered_dets:
        seg = inst['segmentation']
        score = inst['score']

        # mask
        h,w = seg['size']
        try:
            rle = cocomask.frPyObjects(seg, h, w)
        except:
            rle = seg
        mask = cocomask.decode(rle)
        mask = np.logical_and(mask > 0, whole_depth > 0)
        if np.sum(mask) > 32:
            bbox = get_bbox(mask)
            y1, y2, x1, x2 = bbox
        else:
            continue
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]

        # pts
        cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
        center = np.mean(cloud, axis=0)
        tmp_cloud = cloud - center[None, :]
        flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
        if np.sum(flag) < 4:
            continue
        choose = choose[flag]
        cloud = cloud[flag]

        if len(choose) <= cfg.n_sample_observed_point:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point, replace=True)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        cloud = cloud[choose_idx]

        # rgb
        rgb = whole_image.copy()[y1:y2, x1:x2, :][:,:,::-1]
        if cfg.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = rgb_transform_pem(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)

        all_rgb.append(torch.FloatTensor(rgb))
        all_cloud.append(torch.FloatTensor(cloud))
        all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
        all_score.append(score)
        all_dets.append(inst)

    ret_dict = {}
    if not all_cloud: 
        return None, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets 

    ret_dict['pts'] = torch.stack(all_cloud).cuda()
    ret_dict['rgb'] = torch.stack(all_rgb).cuda()
    ret_dict['rgb_choose'] = torch.stack(all_rgb_choose).cuda()
    ret_dict['score'] = torch.FloatTensor(all_score).cuda()

    ninstance = ret_dict['pts'].size(0)
    ret_dict['model'] = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    ret_dict['K'] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    return ret_dict, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets


# --- Argument Parser ---
def get_parser():
    parser = argparse.ArgumentParser(description="SAM-6D Sequence Inference")
    parser.add_argument("--scene_dir", required=True, help="Path to the scene directory containing frame folders (e.g., Data/mustard0/000000).")
    parser.add_argument("--num_frames", type=int, required=True, help="Total number of frames to process (e.g., 19).")
    parser.add_argument("--root_output_dir", required=True, help="Path to root output directory (where templates are saved).")
    parser.add_argument("--cad_path", required=True, help="Path to CAD model (mm).")
    parser.add_argument("--cam_path", required=True, help="Path to camera information.")
    parser.add_argument("--segmentor_model", default='sam', help="The segmentor model in ISM ('sam' or 'fastsam').")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="Stability score threshold of SAM.")
    parser.add_argument("--det_score_thresh", default=0.2, type=float, help="The score threshold for final detection in PEM.")
    parser.add_argument("--gpus", type=str, default="0", help="GPU IDs to use.")
    # CRITICAL FIX: Return the parser object, not the parsed arguments
    return parser 


# --- Initialization (Run Once) ---
def init_models(args):
    # Set up CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gorilla.utils.set_cuda_visible_devices(gpu_ids = args.gpus)
    
    # 1. Initialize ISM
    print("--- 1. Initializing Instance Segmentation Model (ISM) ---")
    
    # FIX: Use RELATIVE paths for Hydra, based on the structure being run from BASE_DIR
    ISM_CONFIG_DIR_RELATIVE = "Instance_Segmentation_Model/configs"
    
    # We must initialize Hydra using the directory name, not the absolute path.
    with initialize(version_base=None, config_path=ISM_CONFIG_DIR_RELATIVE):
        cfg_ism = compose(config_name='run_inference.yaml')
    
    if args.segmentor_model == "sam":
        # The inner compose config_path must also be relative
        with initialize(version_base=None, config_path=osp.join(ISM_CONFIG_DIR_RELATIVE, "model")):
            cfg_ism.model = compose(config_name='ISM_sam.yaml')
        cfg_ism.model.segmentor_model.stability_score_thresh = args.stability_score_thresh
    else:
        raise ValueError("Only 'sam' is supported in the unified script for now.")

    ism_model = instantiate(cfg_ism.model)
    ism_model.descriptor_model.model = ism_model.descriptor_model.model.to(device)
    ism_model.descriptor_model.model.device = device
    if hasattr(ism_model.segmentor_model, "predictor"):
        ism_model.segmentor_model.predictor.model = (
            ism_model.segmentor_model.predictor.model.to(device)
        )
    else:
        try:
            ism_model.segmentor_model.model.setup_model(device=device, verbose=True)
        except AttributeError:
             pass 
             
    ism_model.eval()
    print(f"ISM models moved to {device} done!")
    
    # Load ISM Templates
    print("Loading ISM templates...")
    template_dir = os.path.join(args.root_output_dir, 'templates')
    num_templates = len(glob.glob(f"{template_dir}/xyz_*.npy"))
    if num_templates == 0:
        raise FileNotFoundError(f"No templates found in: {template_dir}. Run rendering step first.")
        
    boxes, masks, templates = [], [], []
    for idx in range(num_templates):
        image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
        mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
        boxes.append(mask.getbbox())

        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
        image = image * mask[:, :, None]
        templates.append(image)
        masks.append(mask.unsqueeze(-1))
        
    templates = torch.stack(templates).permute(0, 3, 1, 2)
    masks = torch.stack(masks).permute(0, 3, 1, 2)
    boxes = torch.tensor(np.array(boxes))
    
    processing_config = OmegaConf.create({"image_size": 224})
    proposal_processor = CropResizePad(processing_config.image_size)
    templates = proposal_processor(images=templates, boxes=boxes).to(device)
    masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

    ism_model.ref_data = {}
    ism_model.ref_data["descriptors"] = ism_model.descriptor_model.compute_features(
                    templates, token_name="x_norm_clstoken"
                ).unsqueeze(0).data
    ism_model.ref_data["appe_descriptors"] = ism_model.descriptor_model.compute_masked_patch_feature(
                    templates, masks_cropped[:, 0, :, :]
                ).unsqueeze(0).data
    print("ISM templates loaded and processed.")
    
    # 2. Initialize PEM
    print("\n--- 2. Initializing Pose Estimation Model (PEM) ---")
    
    # Load PEM Config
    cfg_pem_file = osp.join(BASE_DIR, "Pose_Estimation_Model/config/base.yaml")
    cfg_pem = gorilla.Config.fromfile(cfg_pem_file)
    
    # *** CRITICAL FIX: Set template sample size defensively ***
    # This value (128) was mathematically determined to prevent the feature index crash.
    cfg_pem.test_dataset.n_sample_template_point = 128 
    
    # Model Setup
    pem_model_dir = osp.join(BASE_DIR, 'Pose_Estimation_Model/model')
    if pem_model_dir not in sys.path:
        sys.path.append(pem_model_dir) 
        
    MODEL = importlib.import_module('pose_estimation_model') 
    pem_model = MODEL.Net(cfg_pem.model)
    pem_model = pem_model.cuda()
    pem_model.eval()
    checkpoint = os.path.join(BASE_DIR, 'Pose_Estimation_Model/checkpoints', 'sam-6d-pem-base.pth')
    gorilla.solver.load_checkpoint(model=pem_model, filename=checkpoint)
    print("PEM model loaded.")

    # Load PEM Templates and Features (Run Once)
    print("Loading and extracting PEM template features...")
    tem_path = os.path.join(args.root_output_dir, 'templates')
    
    # CRITICAL FIX: Get LISTS of templates for correct feature extraction flow
    tem_rgb_list, tem_pts_list, tem_choose_list = get_templates(tem_path, cfg_pem.test_dataset)
    
    with torch.no_grad():
        # NOTE: pem_model.feature_extraction.get_obj_feats performs the necessary 
        # internal feature-level concatenation and returns two single TENSORS.
        all_tem_pts, all_tem_feat = pem_model.feature_extraction.get_obj_feats(
            tem_rgb_list, tem_pts_list, tem_choose_list
        )
        
    # DO NOT use torch.cat() here, as all_tem_pts and all_tem_feat are already 
    # single TENSORS from the result of get_obj_feats. The variables are now ready.


    # PEM Geometric/Object Data (Run Once)
    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(template_poses).to(torch.float32).to(device)
    ism_model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]
    
    mesh = trimesh.load_mesh(args.cad_path)
    cfg_pem.test_dataset.n_sample_model_point = 512
    model_points = mesh.sample(cfg_pem.test_dataset.n_sample_model_point).astype(np.float32) / 1000.0
    ism_model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)

    return ism_model, pem_model, cfg_pem, all_tem_pts, all_tem_feat, model_points

# --- Main Inference Loop ---

def run_sequence(args, ism_model, pem_model, cfg_pem, all_tem_pts, all_tem_feat, model_points):
    
    sequence_vis_dir = os.path.join(args.root_output_dir, 'sequence_vis')
    os.makedirs(sequence_vis_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in tqdm(range(args.num_frames), desc="Processing Frames"):
        frame_id = f"{i:06d}"
        
        frame_dir = os.path.join(args.scene_dir, frame_id)
        rgb_path = frame_dir+ ".png"
        depth_path = frame_dir+ "_depth.png"
        
        if not os.path.exists(rgb_path):
            tqdm.write(f"Warning: Frame {frame_id} missing RGB file at {rgb_path}. Skipping.")
            continue
            
        frame_output_dir = os.path.join(args.root_output_dir, f"frame_{frame_id}")
        os.makedirs(os.path.join(frame_output_dir, 'sam6d_results'), exist_ok=True)
        
        # --- 1. Run ISM (Segmentation & Refinement) ---
        
        rgb_img_pil = Image.open(rgb_path).convert("RGB")
        rgb_np = np.array(rgb_img_pil) # <--- NumPy array for visualization
        
        # Segmentation step
        with torch.no_grad():
            detections_list = ism_model.segmentor_model.generate_masks(rgb_np)
            detections = Detections(detections_list)
        
        if len(detections) == 0:
            tqdm.write(f"Warning: No detections found for frame {frame_id}. Skipping PEM.")
            # ISM Visualization: Save/return the original image if no detections
            ism_save_path = f"{frame_output_dir}/sam6d_results/vis_ism_{frame_id}.png"
            ism_vis_pil = Image.fromarray(rgb_np)
            ism_vis_pil.save(ism_save_path)
            continue

        # Descriptor Matching
        with torch.no_grad():
            query_decriptors, query_appe_descriptors = ism_model.descriptor_model.forward(rgb_np, detections)
            (
                idx_selected_proposals,
                pred_idx_objects,
                semantic_score,
                best_template,
            ) = ism_model.compute_semantic_score(query_decriptors)

            detections.filter(idx_selected_proposals)
            query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]
            
            if len(detections) == 0:
                tqdm.write(f"Warning: All detections filtered out for frame {frame_id}. Skipping PEM.")
                # ISM Visualization: Save/return the original image if filtered out
                ism_save_path = f"{frame_output_dir}/sam6d_results/vis_ism_{frame_id}.png"
                ism_vis_pil = Image.fromarray(rgb_np)
                ism_vis_pil.save(ism_save_path)
                continue
                
            # Appearance Score
            appe_scores, ref_aux_descriptor= ism_model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

            # Geometric Score setup
            batch_ism = batch_input_data_ism(depth_path, args.cam_path, device)
            image_uv = ism_model.project_template_to_image(best_template, pred_idx_objects, batch_ism, detections.masks)

            # Geometric Score
            geometric_score, visible_ratio = ism_model.compute_geometric_score(
                image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=ism_model.visible_thred
            )

            # Final ISM Score
            final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

            # --- DIAGNOSTIC PATCH: Check Final Scores ---
            tqdm.write(f"Frame {frame_id}: Final ISM Scores (before thresholding):")
            tqdm.write(str(final_score.detach().cpu().numpy()))
            # ---------------------------------------------
            
            detections.add_attribute("scores", final_score)
            detections.add_attribute("object_ids", torch.zeros_like(final_score))   
                
            # NOTE: detections is now the object passed to PEM, no file I/O here!
            detections.to_numpy() # Converts internal data to NumPy arrays/dictionaries

            # --- CRITICAL FIX 1: Explicitly convert Detections to List of Dicts now ---
            try:
                # This requires the Detections.to_list() method to be implemented in model/utils.py
                final_ism_detections = detections.to_list()
            except AttributeError:
                # Fallback: After to_numpy() is called, the internal list is often accessible via _data
                if hasattr(detections, '_data') and isinstance(detections._data, list):
                    final_ism_detections = detections._data
                else:
                    tqdm.write(f"FATAL: Cannot unpack Detections object for PEM in frame {frame_id}. Ensure Detections.to_list() is implemented.")
                    continue # Skip frame if we can't get the data
        
        # ISM Visualization (Saved and returned as PIL object)
        ism_save_path = f"{frame_output_dir}/sam6d_results/vis_ism_{frame_id}.png"
        ism_vis_pil = visualize_ism(rgb_np, final_ism_detections, ism_save_path)

        # --- 2. Run PEM (Pose Estimation) - IN-MEMORY PASSING ---
        
        # CRITICAL FIX 2: Pass the list of dicts, not the Detections object itself
        input_data, img_pem, _, model_points_pem, pem_detections = get_test_data_pem_in_memory(
            rgb_path, depth_path, args.cam_path, args.cad_path, final_ism_detections, 
            args.det_score_thresh, cfg_pem.test_dataset
        )
        
        if input_data is None:
            tqdm.write(f"Warning: No valid PEM input found for frame {frame_id}. Skipping pose estimation (likely due to low score).")
            # Move ISM visualization to final directory (only ISM available)
            final_vis_path = os.path.join(sequence_vis_dir, f"vis_ism_{frame_id}.png")
            shutil.move(ism_save_path, final_vis_path)
            continue

        ninstance = input_data['pts'].size(0)
        
        # Inference
        with torch.no_grad():
            # Pass the CONCATENATED template features (all_tem_pts and all_tem_feat are single TENSORS)
            input_data['dense_po'] = all_tem_pts.repeat(ninstance,1,1)
            input_data['dense_fo'] = all_tem_feat.repeat(ninstance,1,1)
            out = pem_model(input_data)

            if 'pred_pose_score' in out.keys():
                pose_scores = out['pred_pose_score'].detach().cpu().numpy() * input_data['score'].detach().cpu().numpy()
            else:
                pose_scores = input_data['score'].detach().cpu().numpy()

            pred_rot = out['pred_R'].detach().cpu().numpy()
            pred_trans = out['pred_t'].detach().cpu().numpy() * 1000 # Convert back to mm

        # Find the best pose for visualization
        max_score_idx = np.argmax(pose_scores)
        K_matrix = input_data['K'].detach().cpu().numpy()[max_score_idx] # This is a 3x3 array
        
        # Visualization (and saving the single final image)
        pem_save_path = os.path.join(f"{frame_output_dir}/sam6d_results", f'vis_pem_full_{frame_id}.png')
        
        # MODIFIED CALL: Pass the PIL object of the ISM visualization
        vis_pem_img_full = visualize_pem(
            img_pem, # Original RGB/Input for PEM
            ism_vis_pil, # ISM Segmentation Visualization (PIL)
            pred_rot[max_score_idx:max_score_idx+1], 
            pred_trans[max_score_idx:max_score_idx+1], 
            model_points_pem*1000, 
            K_matrix, 
            pem_save_path
        )
        vis_pem_img_full.save(pem_save_path)

        # Move final viz to sequence folder
        final_vis_path = os.path.join(sequence_vis_dir, f"vis_pem_full_{frame_id}.png")
        shutil.move(pem_save_path, final_vis_path)
        
        # Clean up intermediate frame directory
        # shutil.rmtree(frame_output_dir) # Uncomment this if you want to clean up intermediate files

    # Final cleanup of the live window
    cv2.destroyAllWindows()
    print("\n--- Sequence processing finished. Final visualizations (RGB | ISM | PEM) are in the 'sequence_vis' folder. ---")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args() # FIX: Correctly call parse_args() on the parser object
    
    # 1. Run all model loading only once
    print("--- 2. Starting Unified Sequential Inference ---")
    try:
        ism_model, pem_model, cfg_pem, all_tem_pts, all_tem_feat, model_points = init_models(args)
    except Exception as e:
        print(f"\nFATAL ERROR during model initialization: {e}")
        print("Check your configuration files (e.g., base.yaml) or ensure the necessary external fixes (e.g., feature_extraction.py patches) are still in place.")
        sys.exit(1)
    
    # 2. Run the frame loop
    run_sequence(args, ism_model, pem_model, cfg_pem, all_tem_pts, all_tem_feat, model_points)