import os
import re
import time
import torch
import trimesh
import objaverse
import google.genai as genai
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import fast_simplification as fs

# ======================================================
# Configuration (Merged from both scripts)
# ======================================================
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
IMAGE_PATH = "../FoundationPose/demo_data/kinect_driller_seq/rgb/0000001.png"
OUTPUT_DIR = "./objects"

MAX_MESHES_PER_LABEL = 10
MAX_VERTICES = 100_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# Core Functions
# ======================================================

def setup_models():
    """Combines initialization with the status prints from the second script."""
    print(f"Loading models to {DEVICE}...") # From script 2
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    model_id = "vikhyatk/moondream2"
    vision_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    return client, vision_model, tokenizer

def decimate_to_vertex_budget(mesh, max_vertices):
    """Refined decimation from the first script."""
    if len(mesh.vertices) <= max_vertices:
        return mesh
    
    points, faces = mesh.vertices, mesh.faces
    # Using fast_simplification for high-performance reduction
    target_reduction = 1 - (max_vertices / len(mesh.vertices))
    points_out, faces_out = fs.simplify(points, faces, target_reduction)
    return trimesh.Trimesh(vertices=points_out, faces=faces_out)

def get_labels_with_timing(vision_model, tokenizer, image_path):
    """Inference logic with timing prints from the second script."""
    print(f"Running detection on: {image_path}...")
    
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    image = Image.open(image_path)
    prompt = "List all the individual objects in this image as a comma separated list. Only include the object names."
    
    # Vision-Language Model Inference
    encoded_image = vision_model.encode_image(image)
    answer = vision_model.answer_question(encoded_image, prompt, tokenizer)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    
    labels = [obj.strip().lower() for obj in answer.split(',')]
    print(f"Detected {len(labels)} objects in {total_time:.2f}s: {labels}")
    return labels

def main():
    # 1. Setup
    client, vision_model, tokenizer = setup_models()
    
    # 2. Get Labels
    labels = get_labels_with_timing(vision_model, tokenizer, IMAGE_PATH)
    
    # 3. Load Taxonomy (Objaverse LVIS)
    print("Loading Objaverse LVIS annotations...") # From script 2
    lvis_annotations = objaverse.load_lvis_annotations()
    all_categories = list(lvis_annotations.keys())

    # 4. Process each label using Gemini mapping
    for label in labels:
        print(f"\nSearching for '{label}' in Objaverse...")
        
        # Smart mapping via Gemini (Logic from script 1)
        prompt = f"Map the object '{label}' to the best category from this list: {all_categories}. Return ONLY the category name."
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        category = response.text.strip()

        if category not in lvis_annotations:
            print(f"  -> No match found for '{label}'")
            continue

        print(f"  -> Found match: '{category}'. Downloading top meshes...") # From script 2
        
        save_dir = os.path.join(OUTPUT_DIR, label.replace(" ", "_"))
        os.makedirs(save_dir, exist_ok=True)
        
        uids = lvis_annotations[category][:MAX_MESHES_PER_LABEL]
        meshes_dict = objaverse.load_objects(uids=uids)

        for i, (uid, path) in enumerate(meshes_dict.items()):
            out_path = os.path.join(save_dir, f"{label.replace(' ', '_')}_{i}.obj")
            try:
                # Processing and conversion prints (From script 2)
                mesh = trimesh.load(path, force="mesh")
                
                # Apply decimation if needed (Logic from script 1)
                if len(mesh.vertices) > MAX_VERTICES:
                    mesh = decimate_to_vertex_budget(mesh, MAX_VERTICES)
                
                mesh.export(out_path)
                print(f"    -> Converted & Saved: {out_path} ({len(mesh.vertices)} verts)")
            except Exception as e:
                print(f"    -> Skipping {uid[:8]}: Conversion failed ({e})")

if __name__ == "__main__":
    main()