#!/usr/bin/env python3
"""Analyze YCB-V dataset to map scenes to objects"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_ycbv_dataset(dataset_dir="/home/hcp4/FoundationPose/datasets/ycbv"):
    """Map which objects appear in which scenes"""
    
    dataset_path = Path(dataset_dir)
    test_dir = dataset_path / "test"
    
    scene_objects = {}  # {scene_id: [obj_ids]}
    object_scenes = defaultdict(list)  # {obj_id: [scene_ids]}
    
    for scene_dir in sorted(test_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        
        scene_id = int(scene_dir.name)
        scene_gt_file = scene_dir / "scene_gt.json"
        
        if not scene_gt_file.exists():
            continue
        
        with open(scene_gt_file) as f:
            scene_gt = json.load(f)
        
        # Extract unique object IDs
        obj_ids = set()
        for frame_data in scene_gt.values():
            for obj in frame_data:
                obj_ids.add(obj['obj_id'])
        
        scene_objects[scene_id] = sorted(obj_ids)
        
        for obj_id in obj_ids:
            object_scenes[obj_id].append(scene_id)
    
    return scene_objects, dict(object_scenes)

def save_scene_mapping(output_file="scene_object_mapping.json"):
    """Save scene-object mapping"""
    scene_objects, object_scenes = analyze_ycbv_dataset()
    
    mapping = {
        'scene_to_objects': scene_objects,
        'object_to_scenes': object_scenes,
        'statistics': {
            'total_scenes': len(scene_objects),
            'total_objects': len(object_scenes),
            'avg_objects_per_scene': sum(len(objs) for objs in scene_objects.values()) / len(scene_objects)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"✓ Saved mapping to {output_file}")
    print(f"  Scenes: {mapping['statistics']['total_scenes']}")
    print(f"  Objects: {mapping['statistics']['total_objects']}")
    
    return mapping

if __name__ == '__main__':
    save_scene_mapping()