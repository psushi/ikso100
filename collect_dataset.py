#!/usr/bin/env python3
"""
Dataset Collection Script for LeRobot Platform
Collects demonstration data from MuJoCo SO-100 arm simulation
Compatible with LeRobot's HDF5 format and Hugging Face visualization tools
"""

import os
import time
import json
import h5py
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import cv2

# Import boundary check function from main.py
from main import check_workspace_boundaries, WORKSPACE_BOUNDS

class LeRobotDatasetCollector:
    """Dataset collector for LeRobot compatible format"""
    
    def __init__(self, dataset_name: str = "so100_sim_dataset", 
                 dataset_path: str = "./datasets"):
        self.dataset_name = dataset_name
        self.dataset_path = Path(dataset_path)
        self.dataset_dir = self.dataset_path / dataset_name
        
        # Create dataset directory structure
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "data").mkdir(exist_ok=True)
        
        # Camera configurations for multiple views
        self.camera_configs = {
            "cam_high": {
                "distance": 1.5,
                "azimuth": 45,
                "elevation": -30,
                "lookat": [0.0, 0.0, 0.3]
            },
            "cam_side": {
                "distance": 1.2,
                "azimuth": 90,
                "elevation": -15,
                "lookat": [0.0, 0.0, 0.3]
            },
            "cam_front": {
                "distance": 1.0,
                "azimuth": 0,
                "elevation": -20,
                "lookat": [0.0, 0.0, 0.3]
            }
        }
        
        # Image dimensions
        self.img_height = 480
        self.img_width = 640
        
        # Episode tracking
        self.current_episode = 0
        self.episode_data = []
        self.episode_start_time = None
        
        # Simulation parameters (from main.py)
        self.dt = 0.002
        self.integration_dt = 1.0
        self.damping = 1e-4
        
        print(f"Dataset collector initialized: {self.dataset_dir}")
        
    def create_metadata(self) -> Dict:
        """Create dataset metadata compatible with LeRobot"""
        metadata = {
            "codebase_version": "1.0.0",
            "robot_type": "so100_arm",
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 1,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": int(1.0 / self.dt),  # Frames per second
            "splits": {
                "train": "0:80%"
            },
            "data_path": "data",
            "video_path": "videos",
            "features": {
                "observation.images.cam_high": {
                    "dtype": "video",
                    "shape": [self.img_height, self.img_width, 3],
                    "names": ["height", "width", "channels"]
                },
                "observation.images.cam_side": {
                    "dtype": "video", 
                    "shape": [self.img_height, self.img_width, 3],
                    "names": ["height", "width", "channels"]
                },
                "observation.images.cam_front": {
                    "dtype": "video",
                    "shape": [self.img_height, self.img_width, 3], 
                    "names": ["height", "width", "channels"]
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [18],  # 6 joint pos + 6 joint vel + 3 ee pos + 3 ee ori
                    "names": ["joint_positions", "joint_velocities", "ee_position", "ee_orientation"]
                },
                "action": {
                    "dtype": "float32", 
                    "shape": [6],  # 6 joint positions
                    "names": ["joint_commands"]
                }
            },
            "created_at": datetime.now().isoformat(),
            "workspace_bounds": WORKSPACE_BOUNDS
        }
        return metadata
        
    def setup_cameras(self, model: mujoco.MjModel) -> Dict:
        """Setup multiple camera views for data collection"""
        cameras = {}
        
        for cam_name, config in self.camera_configs.items():
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = config["distance"]
            cam.azimuth = config["azimuth"] 
            cam.elevation = config["elevation"]
            cam.lookat[:] = config["lookat"]
            cameras[cam_name] = cam
            
        return cameras
        
    def render_camera_view(self, model: mujoco.MjModel, data: mujoco.MjData, 
                          camera: mujoco.MjvCamera) -> np.ndarray:
        """Render a single camera view and return RGB image"""
        # Create rendering context
        scene = mujoco.MjvScene(model, maxgeom=10000)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        # Update scene
        mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera, 
                              mujoco.mjtCatBit.mjCAT_ALL, scene)
        
        # Render
        viewport = mujoco.MjrRect(0, 0, self.img_width, self.img_height)
        mujoco.mjr_render(viewport, scene, context)
        
        # Read pixels
        rgb_array = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb_array, None, viewport, context)
        
        # Flip vertically (OpenGL convention)
        rgb_array = np.flipud(rgb_array)
        
        return rgb_array
        
    def collect_observation(self, model: mujoco.MjModel, data: mujoco.MjData, 
                           cameras: Dict) -> Dict:
        """Collect multi-modal observation data"""
        obs = {
            "images": {},
            "state": np.zeros(18, dtype=np.float32),
            "timestamp": time.time()
        }
        
        # Collect camera images
        for cam_name, camera in cameras.items():
            obs["images"][cam_name] = self.render_camera_view(model, data, camera)
            
        # Collect proprioceptive state
        joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        joint_ids = [model.joint(name).id for name in joint_names]
        
        # Joint positions (6)
        obs["state"][:6] = data.qpos[joint_ids]
        
        # Joint velocities (6) 
        obs["state"][6:12] = data.qvel[joint_ids]
        
        # End-effector position (3)
        ee_site_id = model.site("attachment_site").id
        obs["state"][12:15] = data.site(ee_site_id).xpos
        
        # End-effector orientation as euler angles (3)
        ee_mat = data.site(ee_site_id).xmat.reshape(3, 3)
        ee_euler = self.mat2euler(ee_mat)
        obs["state"][15:18] = ee_euler
        
        return obs
        
    def mat2euler(self, mat: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to euler angles (ZYX convention)"""
        sy = np.sqrt(mat[0, 0] * mat[0, 0] + mat[1, 0] * mat[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(mat[2, 1], mat[2, 2])
            y = np.arctan2(-mat[2, 0], sy)
            z = np.arctan2(mat[1, 0], mat[0, 0])
        else:
            x = np.arctan2(-mat[1, 2], mat[1, 1])
            y = np.arctan2(-mat[2, 0], sy)
            z = 0
            
        return np.array([x, y, z])
        
    def collect_action(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        """Collect action data (joint commands)"""
        joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        actuator_ids = [model.actuator(name).id for name in joint_names]
        return data.ctrl[actuator_ids].copy()
        
    def start_episode(self):
        """Start a new episode"""
        self.episode_data = []
        self.episode_start_time = time.time()
        print(f"Starting episode {self.current_episode}")
        
    def add_step(self, observation: Dict, action: np.ndarray):
        """Add a step to current episode"""
        step_data = {
            "observation": observation,
            "action": action,
            "timestamp": time.time() - self.episode_start_time
        }
        self.episode_data.append(step_data)
        
    def save_episode(self):
        """Save current episode to HDF5 file"""
        if not self.episode_data:
            print("No data to save for current episode")
            return
            
        episode_file = self.dataset_dir / "data" / f"episode_{self.current_episode:06d}.hdf5"
        
        with h5py.File(episode_file, 'w') as f:
            # Episode metadata
            f.attrs['episode_id'] = self.current_episode
            f.attrs['length'] = len(self.episode_data)
            f.attrs['duration'] = self.episode_data[-1]["timestamp"]
            
            # Pre-allocate datasets
            episode_length = len(self.episode_data)
            
            # Images
            for cam_name in self.camera_configs.keys():
                img_group = f.create_group(f"observation/images/{cam_name}")
                img_data = np.stack([step["observation"]["images"][cam_name] 
                                   for step in self.episode_data])
                img_group.create_dataset("data", data=img_data, compression="gzip")
                
            # State observations
            state_data = np.stack([step["observation"]["state"] for step in self.episode_data])
            f.create_dataset("observation/state", data=state_data)
            
            # Actions
            action_data = np.stack([step["action"] for step in self.episode_data])
            f.create_dataset("action", data=action_data)
            
            # Timestamps
            timestamp_data = np.array([step["timestamp"] for step in self.episode_data])
            f.create_dataset("timestamp", data=timestamp_data)
            
        print(f"Saved episode {self.current_episode} with {episode_length} steps to {episode_file}")
        self.current_episode += 1
        
    def save_metadata(self):
        """Save dataset metadata"""
        metadata = self.create_metadata()
        metadata["total_episodes"] = self.current_episode
        metadata["total_frames"] = sum(len(ep) for ep in [self.episode_data] if ep)
        
        metadata_file = self.dataset_dir / "meta.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Saved metadata to {metadata_file}")


def demonstrate_trajectory(data: mujoco.MjData, mocap_id: int, t: float, 
                          trajectory_type: str = "circle") -> np.ndarray:
    """Generate demonstration trajectories"""
    if trajectory_type == "circle":
        # Circular trajectory
        r = 0.1  # radius
        h, k = 0.0, 0.0  # center
        f = 0.1  # frequency
        x = r * np.cos(2 * np.pi * f * t) + h
        y = r * np.sin(2 * np.pi * f * t) + k
        z = 0.3  # fixed height
        return np.array([x, y, z])
        
    elif trajectory_type == "figure8":
        # Figure-8 trajectory
        f = 0.05
        x = 0.15 * np.sin(2 * np.pi * f * t)
        y = 0.1 * np.sin(4 * np.pi * f * t)
        z = 0.25 + 0.05 * np.sin(2 * np.pi * f * t)
        return np.array([x, y, z])
        
    elif trajectory_type == "square":
        # Square trajectory
        period = 20.0  # seconds for full cycle
        phase = (t % period) / period * 4  # 0-4 for four sides
        
        if phase < 1:  # Bottom edge
            x = -0.1 + 0.2 * phase
            y = -0.1
        elif phase < 2:  # Right edge
            x = 0.1
            y = -0.1 + 0.2 * (phase - 1)
        elif phase < 3:  # Top edge
            x = 0.1 - 0.2 * (phase - 2)
            y = 0.1
        else:  # Left edge
            x = -0.1
            y = 0.1 - 0.2 * (phase - 3)
            
        z = 0.3
        return np.array([x, y, z])
        
    else:
        # Default: stationary
        return np.array([0.0, 0.0, 0.3])


def collect_dataset_episodes(num_episodes: int = 5, episode_duration: float = 30.0):
    """Main function to collect dataset episodes"""
    
    # Initialize dataset collector
    collector = LeRobotDatasetCollector()
    
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    
    # Override timestep
    model.opt.timestep = collector.dt
    
    # Setup simulation components (from main.py)
    end_effector = model.site("attachment_site").id
    
    # Enable gravity compensation
    body_names = ["Rotation_Pitch", "Upper_Arm", "Lower_Arm", 
                  "Wrist_Pitch_Roll", "Fixed_Jaw", "Moving_Jaw"]
    body_ids = [model.body(name).id for name in body_names]
    model.body_gravcomp[body_ids] = 1.0
    
    joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    dof_ids = [model.joint(name).id for name in joint_names]
    actuator_ids = [model.actuator(name).id for name in joint_names]
    
    key_id = model.key("home").id
    mocap_id = model.body("target").mocapid[0]
    
    # Setup cameras
    cameras = collector.setup_cameras(model)
    
    # Pre-allocate arrays for IK
    jac = np.zeros((6, model.nv))
    diag = collector.damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    ee_quat = np.zeros(4)
    ee_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    
    # Trajectory types for different episodes
    trajectories = ["circle", "figure8", "square", "circle", "figure8"]
    
    print(f"Starting dataset collection: {num_episodes} episodes")
    print(f"Episode duration: {episode_duration} seconds each")
    print(f"Dataset will be saved to: {collector.dataset_dir}")
    
    try:
        for episode_idx in range(num_episodes):
            print(f"\n{'='*50}")
            print(f"EPISODE {episode_idx + 1}/{num_episodes}")
            print(f"Trajectory: {trajectories[episode_idx % len(trajectories)]}")
            print(f"{'='*50}")
            
            # Reset simulation
            mujoco.mj_resetDataKeyframe(model, data, key_id)
            
            # Start episode
            collector.start_episode()
            
            episode_start_time = time.time()
            step_count = 0
            
            while (time.time() - episode_start_time) < episode_duration:
                step_start = time.time()
                
                # Generate trajectory
                trajectory_type = trajectories[episode_idx % len(trajectories)]
                target_pos = demonstrate_trajectory(data, mocap_id, data.time, trajectory_type)
                
                # Apply boundary constraints
                target_pos = check_workspace_boundaries(target_pos)
                data.mocap_pos[mocap_id] = target_pos
                
                # Inverse kinematics (from main.py)
                error_pos[:] = data.mocap_pos[mocap_id] - data.site(end_effector).xpos
                
                mujoco.mju_mat2Quat(ee_quat, data.site(end_effector).xmat)
                mujoco.mju_negQuat(ee_quat_conj, ee_quat)
                mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], ee_quat_conj)
                mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)
                
                mujoco.mj_jacSite(model, data, jac[:3], jac[3:], end_effector)
                
                # Solve IK
                dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)
                
                # Integrate joint positions
                q = data.qpos.copy()
                mujoco.mj_integratePos(model, q, dq, collector.integration_dt)
                
                # Set control
                np.clip(q, *model.jnt_range.T, out=q)
                data.ctrl[actuator_ids] = q[dof_ids]
                
                # Step simulation
                mujoco.mj_step(model, data)
                
                # Collect data every few steps to reduce dataset size
                if step_count % 5 == 0:  # Collect every 5th step
                    observation = collector.collect_observation(model, data, cameras)
                    action = collector.collect_action(model, data)
                    collector.add_step(observation, action)
                
                step_count += 1
                
                # Maintain real-time
                time_until_next_step = collector.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                    
                # Progress update
                if step_count % 100 == 0:
                    elapsed = time.time() - episode_start_time
                    progress = elapsed / episode_duration * 100
                    print(f"Episode {episode_idx + 1} progress: {progress:.1f}% "
                          f"({len(collector.episode_data)} samples collected)")
            
            # Save episode
            collector.save_episode()
            
        # Save final metadata
        collector.save_metadata()
        
        print(f"\n{'='*60}")
        print("DATASET COLLECTION COMPLETED!")
        print(f"Total episodes: {collector.current_episode}")
        print(f"Dataset location: {collector.dataset_dir}")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\nDataset collection interrupted by user")
        if collector.episode_data:
            collector.save_episode()
        collector.save_metadata()
        
    except Exception as e:
        print(f"Error during dataset collection: {e}")
        if collector.episode_data:
            collector.save_episode()
        collector.save_metadata()


if __name__ == "__main__":
    # Collect 5 episodes, 30 seconds each
    collect_dataset_episodes(num_episodes=5, episode_duration=30.0)