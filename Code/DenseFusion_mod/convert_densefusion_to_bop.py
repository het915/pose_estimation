#!/usr/bin/env python3
"""
Convert DenseFusion output to BOP challenge format
BOP format: CSV file with columns: scene_id, im_id, obj_id, score, R, t, time
"""

import numpy as np
import csv
import json
import argparse
from pathlib import Path

def rotation_matrix_to_list(R):
    """Convert 3x3 rotation matrix to space-separated string"""
    return ' '.join([str(x) for x in R.flatten()])

def translation_to_list(t):
    """Convert translation vector to space-separated string"""
    return ' '.join([str(x) for x in t.flatten()])

def convert_densefusion_to_bop(predictions_file, output_csv, scene_id=1):
    """
    Convert DenseFusion predictions to BOP format
    
    Args:
        predictions_file: JSON file with DenseFusion predictions
        output_csv: Output CSV file path
        scene_id: Scene ID for BOP format
    
    Expected predictions_file format:
    {
        "image_0": {
            "obj_id": 1,
            "R": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
            "t": [tx, ty, tz],
            "score": 0.95,
            "time": 0.05
        },
        ...
    }
    """
    
    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Write to BOP CSV format
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'])
        
        # Write predictions
        for img_key, pred in predictions.items():
            im_id = int(img_key.split('_')[1])  # Extract image ID
            obj_id = pred['obj_id']
            score = pred.get('score', 1.0)
            R = np.array(pred['R'])
            t = np.array(pred['t']) * 1000.0  # Convert to mm for BOP
            time = pred.get('time', -1)
            
            # Write row
            writer.writerow([
                scene_id,
                im_id,
                obj_id,
                score,
                rotation_matrix_to_list(R),
                translation_to_list(t),
                time
            ])
    
    print(f"BOP format saved to: {output_csv}")
    print(f"Total predictions: {len(predictions)}")

def save_densefusion_output_for_bop(R, t, obj_id, image_id, score=1.0, 
                                    output_json='predictions.json'):
    """
    Helper function to save DenseFusion output in intermediate format
    Call this during evaluation to collect predictions
    
    Args:
        R: 3x3 rotation matrix
        t: 3D translation vector (in meters)
        obj_id: Object ID
        image_id: Image ID
        score: Confidence score
        output_json: Output JSON file
    """
    
    # Load existing predictions or create new dict
    predictions = {}
    if Path(output_json).exists():
        with open(output_json, 'r') as f:
            predictions = json.load(f)
    
    # Add new prediction
    img_key = f"image_{image_id}"
    predictions[img_key] = {
        'obj_id': int(obj_id),
        'R': R.tolist(),
        't': t.tolist(),
        'score': float(score),
        'time': -1  # Will be filled later
    }
    
    # Save
    with open(output_json, 'w') as f:
        json.dump(predictions, f, indent=2)

def create_example_predictions():
    """Create example predictions file"""
    predictions = {}
    
    for i in range(10):
        # Random rotation and translation for example
        theta = np.random.rand() * 2 * np.pi
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        t = np.random.rand(3) * 0.5  # Random translation
        
        predictions[f"image_{i}"] = {
            'obj_id': 1,
            'R': R.tolist(),
            't': t.tolist(),
            'score': 0.9 + np.random.rand() * 0.1,
            'time': 0.05
        }
    
    with open('example_predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print("Created example_predictions.json")
    
    # Convert to BOP format
    convert_densefusion_to_bop('example_predictions.json', 'example_bop_output.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert DenseFusion to BOP format')
    parser.add_argument('--input', type=str, help='Input JSON file with predictions')
    parser.add_argument('--output', type=str, help='Output CSV file in BOP format')
    parser.add_argument('--scene_id', type=int, default=1, help='Scene ID')
    parser.add_argument('--example', action='store_true', help='Create example files')
    
    args = parser.parse_args()
    
    if args.example:
        create_example_predictions()
    elif args.input and args.output:
        convert_densefusion_to_bop(args.input, args.output, args.scene_id)
    else:
        print("Usage:")
        print("  Create example: python convert_densefusion_to_bop.py --example")
        print("  Convert: python convert_densefusion_to_bop.py --input predictions.json --output output.csv")
