#!/usr/bin/env python3
"""
LeRobot Dataset Validation Script
Comprehensive validation for SO-100 arm datasets to ensure compatibility with LeRobot platform
"""

import os
import json
import h5py
import numpy as np
import cv2
from pathlib import Path
import sys


def validate_lerobot_dataset(dataset_path: str) -> bool:
    """Validate LeRobot dataset compatibility"""
    print(f"🔍 Validating LeRobot dataset: {dataset_path}")
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_path}")
        return False
    
    # Check required files and directories
    required_files = ["meta.json"]
    data_dir = dataset_dir / "data"
    videos_dir = dataset_dir / "videos" / "chunk-000"
    
    for file in required_files:
        if not (dataset_dir / file).exists():
            print(f"❌ Missing required file: {file}")
            return False
    
    if not data_dir.exists():
        print(f"❌ Missing data directory")
        return False
    
    if not videos_dir.exists():
        print(f"❌ Missing videos directory")
        return False
    
    # Check metadata
    try:
        with open(dataset_dir / "meta.json", 'r') as f:
            metadata = json.load(f)
        
        required_fields = ["codebase_version", "robot_type", "features", "fps"]
        for field in required_fields:
            if field not in metadata:
                print(f"❌ Missing metadata field: {field}")
                return False
        
        print(f"✅ Metadata structure valid")
        
        # Check features
        features = metadata["features"]
        required_features = ["observation.state", "action"]
        for feature in required_features:
            if feature not in features:
                print(f"❌ Missing feature: {feature}")
                return False
        
        print(f"✅ Required features present")
        
        # Check video features
        video_features = [k for k in features.keys() if k.startswith("observation.images.")]
        if len(video_features) == 0:
            print(f"❌ No video features found")
            return False
        
        print(f"✅ Video features found: {len(video_features)}")
        
        # Check episode files
        episode_files = list(data_dir.glob("episode_*.hdf5"))
        if len(episode_files) == 0:
            print(f"❌ No episode files found")
            return False
        
        print(f"✅ Found {len(episode_files)} episode files")
        
        # Check video files
        video_files = []
        for cam_dir in videos_dir.iterdir():
            if cam_dir.is_dir():
                video_files.extend(list(cam_dir.glob("episode_*.mp4")))
        
        if len(video_files) == 0:
            print(f"❌ No video files found")
            return False
        
        print(f"✅ Found {len(video_files)} video files")
        
        # Check HDF5 structure
        sample_episode = episode_files[0]
        with h5py.File(sample_episode, 'r') as f:
            required_datasets = ["observation.state", "action"]
            for dataset in required_datasets:
                if dataset not in f:
                    print(f"❌ Missing dataset in HDF5: {dataset}")
                    return False
            
            # Check data shapes
            state_shape = f["observation.state"].shape
            action_shape = f["action"].shape
            
            if len(state_shape) != 2 or state_shape[1] != 24:
                print(f"❌ Invalid state shape: {state_shape} (expected: [N, 24])")
                return False
            
            if len(action_shape) != 2 or action_shape[1] != 6:
                print(f"❌ Invalid action shape: {action_shape} (expected: [N, 6])")
                return False
        
        print(f"✅ HDF5 data structure valid")
        print(f"✅ Dataset is LeRobot compatible and ready for training!")
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python validate_lerobot_dataset.py <dataset_path>")
        return 1
    
    dataset_path = sys.argv[1]
    
    if validate_lerobot_dataset(dataset_path):
        print("🎉 Dataset validation passed!")
        return 0
    else:
        print("❌ Dataset validation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 