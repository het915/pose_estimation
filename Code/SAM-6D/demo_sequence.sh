#!/bin/bash

# --- Configuration ---
# Set base directory. We assume the script is run from the project root.
BASE_DIR=$(pwd)
# The scene directory contains all the processed RGB/Depth frames (e.g., 000000.png, 000000_depth.png)
SCENE_DIR=$BASE_DIR/Data/mustard0/000000
# The root output directory, where templates are stored.
OUTPUT_ROOT=$BASE_DIR/Data/mustard0/outputs
# Global paths (constant for all frames)
CAD_PATH=$BASE_DIR/Data/mustard0/mustard.ply
CAMERA_PATH=$BASE_DIR/Data/mustard0/camera.json
# SET THIS: Total number of frames (0 to 18 is 19 frames based on your data example)
NUM_FRAMES=19 

echo "Base Directory: $BASE_DIR"
echo "Scene Directory: $SCENE_DIR"
echo "Root Output Directory (Templates): $OUTPUT_ROOT"

# --- 1. Render CAD templates (ONLY RUN ONCE) ---
echo "--- 1. Rendering CAD templates (Once) ---"
# Templates are saved to $OUTPUT_ROOT/templates
export RENDER_OUTPUT_DIR=$OUTPUT_ROOT 
cd Render
blenderproc run render_custom_templates_test.py --output_dir $RENDER_OUTPUT_DIR --cad_path $CAD_PATH 
cd $BASE_DIR

# --- 2. Run Unified Inference Script (Loading Models Once) ---
echo "--- 2. Starting Unified Sequential Inference ---"

# The unified script now handles the loop, the frame paths, and the cleanup internally.
python run_sequence_inference.py \
    --scene_dir $SCENE_DIR \
    --num_frames $NUM_FRAMES \
    --root_output_dir $OUTPUT_ROOT \
    --cad_path $CAD_PATH \
    --cam_path $CAMERA_PATH \
    --segmentor_model sam \
    --det_score_thresh 0.2

echo "--- All frames processed. Final visualizations are in $OUTPUT_ROOT/sequence_vis ---"

# --- 3. (Optional) Create a Video ---
# Requires ffmpeg installed on your system.
# echo "--- 3. Creating sequence video ---"
# ffmpeg -framerate 10 -i $OUTPUT_ROOT/sequence_vis/vis_pem_%06d.png -c:v libx264 -pix_fmt yuv420p $OUTPUT_ROOT/pose_sequence.mp4 -y
# echo "Video saved to $OUTPUT_ROOT/pose_sequence.mp4"