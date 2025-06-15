#!/usr/bin/env python3
"""
Dataset Inspection Script
Inspects and validates LeRobot compatible datasets
"""

import json
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def inspect_dataset(dataset_path: str = "./datasets/so100_sim_dataset"):
    """Inspect the collected dataset"""
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return
        
    print(f"Inspecting dataset: {dataset_dir}")
    print("="*60)
    
    # Check metadata
    meta_file = dataset_dir / "meta.json"
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        print("METADATA:")
        print(f"  Robot type: {metadata.get('robot_type', 'N/A')}")
        print(f"  Total episodes: {metadata.get('total_episodes', 0)}")
        print(f"  Total frames: {metadata.get('total_frames', 0)}")
        print(f"  FPS: {metadata.get('fps', 'N/A')}")
        print(f"  Created: {metadata.get('created_at', 'N/A')}")
        print(f"  Workspace bounds: {metadata.get('workspace_bounds', 'N/A')}")
        print()
    else:
        print("No metadata file found")
        return
    
    # Check data directory
    data_dir = dataset_dir / "data"
    if not data_dir.exists():
        print("No data directory found")
        return
        
    # List episode files
    episode_files = sorted(list(data_dir.glob("episode_*.hdf5")))
    print(f"EPISODES FOUND: {len(episode_files)}")
    
    if not episode_files:
        print("No episode files found")
        return
    
    # Inspect first episode in detail
    first_episode = episode_files[0]
    print(f"\nInspecting first episode: {first_episode.name}")
    print("-" * 40)
    
    with h5py.File(first_episode, 'r') as f:
        print("Episode attributes:")
        for attr_name, attr_value in f.attrs.items():
            print(f"  {attr_name}: {attr_value}")
        
        print("\nDatasets in episode:")
        def print_dataset_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        
        f.visititems(print_dataset_info)
        
        # Show some sample data
        if "observation/state" in f:
            state_data = f["observation/state"][:]
            print(f"\nState data sample (first 3 timesteps):")
            print(f"  Shape: {state_data.shape}")
            print(f"  First 3 rows:\n{state_data[:3]}")
            
        if "action" in f:
            action_data = f["action"][:]
            print(f"\nAction data sample (first 3 timesteps):")
            print(f"  Shape: {action_data.shape}")
            print(f"  First 3 rows:\n{action_data[:3]}")
            
        if "timestamp" in f:
            timestamp_data = f["timestamp"][:]
            print(f"\nTimestamp data:")
            print(f"  Shape: {timestamp_data.shape}")
            print(f"  Duration: {timestamp_data[-1]:.2f} seconds")
            print(f"  Sample rate: {len(timestamp_data)/timestamp_data[-1]:.1f} Hz")
    
    # Summary of all episodes
    print(f"\n{'='*60}")
    print("EPISODE SUMMARY:")
    print(f"{'Episode':<15} {'Duration (s)':<12} {'Steps':<8} {'File Size':<12}")
    print("-" * 50)
    
    total_steps = 0
    total_duration = 0
    
    for i, episode_file in enumerate(episode_files):
        try:
            with h5py.File(episode_file, 'r') as f:
                duration = f.attrs.get('duration', 0)
                length = f.attrs.get('length', 0)
                file_size = episode_file.stat().st_size / 1024  # KB
                
                print(f"{episode_file.stem:<15} {duration:<12.2f} {length:<8} {file_size:<12.1f} KB")
                
                total_steps += length
                total_duration += duration
                
        except Exception as e:
            print(f"{episode_file.stem:<15} ERROR: {e}")
    
    print("-" * 50)
    print(f"{'TOTAL':<15} {total_duration:<12.2f} {total_steps:<8}")
    print(f"\nAverage episode duration: {total_duration/len(episode_files):.2f} seconds")
    print(f"Average steps per episode: {total_steps/len(episode_files):.1f}")
    
    return metadata, episode_files

def inspect_dataset_safe(dataset_path: str = "./datasets/so100_sim_dataset"):
    """Safe version that handles empty datasets"""
    try:
        return inspect_dataset(dataset_path)
    except:
        return None, []

def plot_episode_data(episode_file: Path):
    """Plot data from a single episode"""
    print(f"\nPlotting data from: {episode_file.name}")
    
    with h5py.File(episode_file, 'r') as f:
        if "observation/state" not in f or "action" not in f or "timestamp" not in f:
            print("Required datasets not found in episode file")
            return
            
        state_data = f["observation/state"][:]
        action_data = f["action"][:]
        timestamp_data = f["timestamp"][:]
    
    # Create plots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Episode Data: {episode_file.name}', fontsize=16)
    
    # Joint positions
    axes[0, 0].plot(timestamp_data, state_data[:, :6])
    axes[0, 0].set_title('Joint Positions')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position (rad)')
    axes[0, 0].legend(['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw'])
    axes[0, 0].grid(True)
    
    # Joint velocities
    axes[0, 1].plot(timestamp_data, state_data[:, 6:12])
    axes[0, 1].set_title('Joint Velocities')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (rad/s)')
    axes[0, 1].legend(['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw'])
    axes[0, 1].grid(True)
    
    # End-effector position
    axes[1, 0].plot(timestamp_data, state_data[:, 12:15])
    axes[1, 0].set_title('End-Effector Position')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Position (m)')
    axes[1, 0].legend(['X', 'Y', 'Z'])
    axes[1, 0].grid(True)
    
    # End-effector orientation
    axes[1, 1].plot(timestamp_data, state_data[:, 15:18])
    axes[1, 1].set_title('End-Effector Orientation (Euler)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Angle (rad)')
    axes[1, 1].legend(['Roll', 'Pitch', 'Yaw'])
    axes[1, 1].grid(True)
    
    # Actions
    axes[2, 0].plot(timestamp_data, action_data)
    axes[2, 0].set_title('Joint Commands (Actions)')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Command (rad)')
    axes[2, 0].legend(['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw'])
    axes[2, 0].grid(True)
    
    # End-effector trajectory (X-Y plane)
    axes[2, 1].plot(state_data[:, 12], state_data[:, 13])
    axes[2, 1].set_title('End-Effector Trajectory (X-Y)')
    axes[2, 1].set_xlabel('X Position (m)')
    axes[2, 1].set_ylabel('Y Position (m)')
    axes[2, 1].grid(True)
    axes[2, 1].axis('equal')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = episode_file.parent.parent / f"{episode_file.stem}_plot.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    
    plt.show()

if __name__ == "__main__":
    # Inspect the dataset
    metadata, episode_files = inspect_dataset_safe()
    
    # Plot first episode if available
    if episode_files:
        plot_episode_data(episode_files[0])
    else:
        print("No episodes found to plot") 