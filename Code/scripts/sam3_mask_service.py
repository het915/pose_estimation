#!/usr/bin/env python
"""
SAM3 Mask Generation Service
Run this with: python sam3_mask_service.py --image <path> --prompt <text> --output <path>
"""
import argparse
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import sys

def generate_mask(image_path, text_prompt, output_path, score_threshold=0.5):
    """Generate mask using SAM3 and save to file"""
    print(f"[SAM3] Loading model...", file=sys.stderr)
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    
    print(f"[SAM3] Loading image: {image_path}", file=sys.stderr)
    image = Image.open(image_path).convert('RGB')
    
    print(f"[SAM3] Running inference with prompt: '{text_prompt}'", file=sys.stderr)
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
    
    masks = output["masks"]
    scores = output["scores"]
    
    # Filter by score
    valid_indices = scores >= score_threshold
    if not valid_indices.any():
        print(f"[SAM3] WARNING: No detections found with score >= {score_threshold}", file=sys.stderr)
        # Return empty mask
        mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        best_score = 0.0
    else:
        # Get highest confidence detection
        best_idx = scores[valid_indices].argmax()
        best_mask = masks[valid_indices][best_idx]
        best_score = scores[valid_indices][best_idx].item()
        print(f"[SAM3] Found '{text_prompt}' with confidence {best_score:.3f}", file=sys.stderr)
        
        # Convert to numpy and squeeze to 2D
        mask_np = best_mask.cpu().numpy().squeeze()
        print(f"[SAM3] Mask shape after squeeze: {mask_np.shape}", file=sys.stderr)
        
        # Ensure it's 2D
        if mask_np.ndim != 2:
            print(f"[SAM3] ERROR: Unexpected mask dimensions: {mask_np.ndim}", file=sys.stderr)
            mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
            best_score = 0.0
        else:
            mask = (mask_np * 255).astype(np.uint8)
    
    # Save mask as PNG
    print(f"[SAM3] Saving mask to: {output_path}", file=sys.stderr)
    Image.fromarray(mask, mode='L').save(output_path)
    print(f"[SAM3] Done!", file=sys.stderr)
    
    return best_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate segmentation mask using SAM3')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for segmentation')
    parser.add_argument('--output', type=str, required=True, help='Output mask path (PNG)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    score = generate_mask(args.image, args.prompt, args.output, args.threshold)
    print(score)  # Print score to stdout for parsing
