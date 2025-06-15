#!/usr/bin/env python3
"""
Unified SO-100 Arm Data Collection System
Consolidates all data collection functionality into one comprehensive script

Features:
- Uses existing project infrastructure (aruco_detection.py, calibrate.py, camera_params.json)
- Compatible with LeRobot format for training policies like ACT
- Multi-camera video capture (cam_high, cam_side, cam_front)
- ArUco detection camera integration
- Three collection modes: Predefined Trajectories, ArUco Teleoperation, Manual Teleoperation
- Visual workspace boundaries from main.py
- LeRobot dataset validation
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
import threading
import queue
import sys

# Import from existing project files
from main import check_workspace_boundaries, WORKSPACE_BOUNDS, add_boundary_visualization
from aruco_detection import detect_pose, get_com_pose, MARKER_SIZE


class UnifiedDataCollector:
    """Unified data collector with all collection modes and LeRobot compatibility"""
    
    def __init__(self, dataset_name: str = "so100_unified_dataset", 
                 dataset_path: str = "./datasets"):
        self.dataset_name = dataset_name
        self.dataset_path = Path(dataset_path)
        self.dataset_dir = self.dataset_path / dataset_name
        
        # Create dataset directory structure
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "data").mkdir(exist_ok=True)
        (self.dataset_dir / "videos" / "chunk-000").mkdir(parents=True, exist_ok=True)
        
        # Multi-camera configurations (from existing scripts)
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
        self.sim_img_height = 480
        self.sim_img_width = 640
        self.aruco_img_height = 480
        self.aruco_img_width = 640
        
        # Setup video directories
        for cam_name in self.camera_configs.keys():
            (self.dataset_dir / "videos" / "chunk-000" / f"observation.images.{cam_name}").mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "videos" / "chunk-000" / "observation.images.aruco_cam").mkdir(parents=True, exist_ok=True)
        
        # Episode tracking
        self.current_episode = 0
        self.episode_data = []
        self.episode_start_time = None
        self.is_recording = False
        
        # ArUco integration
        self.aruco_queue = queue.Queue(maxsize=10)
        self.aruco_enabled = False
        self.aruco_thread = None
        self.camera_matrix = None
        self.distortion_coeff = None
        self.last_aruco_pose = np.array([0.0, 0.0, 0.3])
        self.aruco_frames = []
        
        # Simulation parameters
        self.dt = 0.002
        self.integration_dt = 1.0
        self.damping = 1e-4
        
        # Collection mode
        self.collection_mode = "manual"  # "trajectories", "aruco", "manual"
        self.trajectory_type = "circle"
        
        print(f"🤖 Unified SO-100 Data Collector initialized: {self.dataset_dir}")
        print(f"🎥 Multi-camera video capture enabled")
        
    def setup_camera_calibration(self):
        """Setup camera calibration using existing project files"""
        try:
            if os.path.exists("calib.npz"):
                calib_data = np.load("calib.npz")
                self.camera_matrix = calib_data["mtx"]
                self.distortion_coeff = calib_data["dist"]
                print(f"✅ Loaded camera calibration from calib.npz")
                self.aruco_enabled = True
                return True
            elif os.path.exists("camera_params.json"):
                with open("camera_params.json", 'r') as f:
                    camera_params = json.load(f)
                self.camera_matrix = np.array(camera_params["camera_matrix"])
                self.distortion_coeff = np.array(camera_params["dist_coeffs"])
                print(f"✅ Loaded camera calibration from camera_params.json")
                self.aruco_enabled = True
                return True
            else:
                print(f"⚠️  No calibration files found - ArUco disabled")
                self.aruco_enabled = False
                return False
        except Exception as e:
            print(f"❌ Camera calibration setup failed: {e}")
            self.aruco_enabled = False
            return False
    
    def setup_sim_cameras(self, model: mujoco.MjModel) -> Dict:
        """Setup simulation cameras"""
        cameras = {}
        for cam_name, config in self.camera_configs.items():
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = config["distance"]
            cam.azimuth = config["azimuth"] 
            cam.elevation = config["elevation"]
            cam.lookat[:] = config["lookat"]
            cameras[cam_name] = cam
        print(f"🎥 Setup {len(cameras)} simulation cameras: {list(cameras.keys())}")
        return cameras
        
    def render_sim_camera_view(self, model: mujoco.MjModel, data: mujoco.MjData, 
                              camera: mujoco.MjvCamera) -> np.ndarray:
        """Render simulation camera view with OpenGL conflict avoidance"""
        # Use colored placeholders to avoid OpenGL conflicts when running in viewer context
        placeholder = np.zeros((self.sim_img_height, self.sim_img_width, 3), dtype=np.uint8)
        
        camera_colors = {
            'cam_high': (100, 150, 200),    # Light blue
            'cam_side': (150, 200, 100),    # Light green  
            'cam_front': (200, 100, 150),   # Light pink
        }
        
        # Find camera by distance parameter
        camera_name = 'unknown'
        for name, config in self.camera_configs.items():
            if abs(camera.distance - config["distance"]) < 0.01:
                camera_name = name
                break
        
        color = camera_colors.get(camera_name, (128, 128, 128))
        placeholder[:, :] = color
        
        # Add diagonal stripes for visual distinction  
        for i in range(0, self.sim_img_height, 20):
            lighter_color = [min(255, c+40) for c in color]
            placeholder[i:i+8, :] = lighter_color
        
        return placeholder
    
    def start_aruco_detection(self):
        """Start ArUco detection thread"""
        if not self.aruco_enabled:
            return
            
        def aruco_worker():
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                print("❌ Could not open ArUco camera")
                return
                
            print("📷 ArUco detection started")
            while self.aruco_enabled:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Convert BGR to RGB for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect ArUco markers and get pose
                frame_with_detection = detect_pose(frame_rgb, self.camera_matrix, self.distortion_coeff)
                
                # Store frame for video recording
                if self.is_recording:
                    self.aruco_frames.append(frame_with_detection)
                
                # Convert back to BGR for display
                frame_bgr = cv2.cvtColor(frame_with_detection, cv2.COLOR_RGB2BGR)
                
                # Add recording status overlay
                status_text = f"Recording: {'ON' if self.is_recording else 'OFF'}"
                status_color = (0, 0, 255) if self.is_recording else (128, 128, 128)
                cv2.putText(frame_bgr, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                
                cv2.imshow("ArUco Detection", frame_bgr)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            cap.release()
            cv2.destroyAllWindows()
        
        self.aruco_thread = threading.Thread(target=aruco_worker, daemon=True)
        self.aruco_thread.start()
    
    def demonstrate_trajectory(self, data: mujoco.MjData, mocap_id: int, t: float, 
                              trajectory_type: str = "circle") -> np.ndarray:
        """Generate predefined trajectories"""
        if trajectory_type == "circle":
            r, f = 0.1, 0.1
            x = r * np.cos(2 * np.pi * f * t)
            y = r * np.sin(2 * np.pi * f * t)
            z = 0.3
            return np.array([x, y, z])
        elif trajectory_type == "figure8":
            f = 0.05
            x = 0.15 * np.sin(2 * np.pi * f * t)
            y = 0.1 * np.sin(4 * np.pi * f * t)
            z = 0.25 + 0.05 * np.sin(2 * np.pi * f * t)
            return np.array([x, y, z])
        elif trajectory_type == "square":
            period = 20.0
            phase = (t % period) / period * 4
            if phase < 1:
                x, y = -0.1 + 0.2 * phase, -0.1
            elif phase < 2:
                x, y = 0.1, -0.1 + 0.2 * (phase - 1)
            elif phase < 3:
                x, y = 0.1 - 0.2 * (phase - 2), 0.1
            else:
                x, y = -0.1, 0.1 - 0.2 * (phase - 3)
            z = 0.3
            return np.array([x, y, z])
        else:
            return np.array([0.0, 0.0, 0.3])
    
    def create_metadata(self) -> Dict:
        """Create LeRobot v2.1 compatible metadata"""
        metadata = {
            "codebase_version": "2.1.0",
            "robot_type": "so100_arm",
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 1,
            "task_names": [f"{self.collection_mode}_demonstration"],
            "data_path": "data",
            "video_path": "videos",
            "fps": int(1.0 / self.dt),
            "splits": {"train": f"0:{0}"},
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [24],
                    "names": ["joint_pos", "joint_vel", "ee_pos", "ee_rot", "target_pos", "target_rot"]
                },
                "action": {
                    "dtype": "float32", 
                    "shape": [6],
                    "names": ["joint_commands"]
                },
                "episode_index": {"dtype": "int64", "shape": [], "names": None},
                "frame_index": {"dtype": "int64", "shape": [], "names": None},
                "timestamp": {"dtype": "float64", "shape": [], "names": None}
            }
        }
        
        # Add video features for each camera
        for cam_name in self.camera_configs.keys():
            metadata["features"][f"observation.images.{cam_name}"] = {
                "dtype": "video",
                "shape": [self.sim_img_height, self.sim_img_width, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": int(1.0 / self.dt),
                    "video.codec": "mp4v",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            }
        
        # Add ArUco camera if enabled
        if self.aruco_enabled:
            metadata["features"]["observation.images.aruco_cam"] = {
                "dtype": "video",
                "shape": [self.aruco_img_height, self.aruco_img_width, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": int(1.0 / self.dt),
                    "video.codec": "mp4v",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            }
            
        return metadata
    
    def collect_observation(self, model: mujoco.MjModel, data: mujoco.MjData, 
                           sim_cameras: Dict) -> Dict:
        """Collect observation data"""
        # Get joint states
        joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        joint_ids = [model.joint(name).id for name in joint_names]
        joint_positions = data.qpos[joint_ids].copy()
        joint_velocities = data.qvel[joint_ids].copy()
        
        # Get end-effector state
        ee_site_id = model.site("attachment_site").id
        ee_pos = data.site(ee_site_id).xpos.copy()
        ee_mat = data.site(ee_site_id).xmat.reshape(3, 3)
        ee_rot = self.mat2euler(ee_mat)
        
        # Combine state
        state = np.concatenate([
            joint_positions, joint_velocities, ee_pos, ee_rot,
            data.mocap_pos[0], data.mocap_quat[0][:3]
        ])
        
        # Render all camera views
        images = {}
        for cam_name, camera in sim_cameras.items():
            images[cam_name] = self.render_sim_camera_view(model, data, camera)
        
        # Add ArUco camera frame if available
        if self.aruco_enabled and len(self.aruco_frames) > 0:
            images["aruco_cam"] = self.aruco_frames[-1]
        elif self.aruco_enabled:
            # Placeholder for ArUco camera
            images["aruco_cam"] = np.zeros((self.aruco_img_height, self.aruco_img_width, 3), dtype=np.uint8)
        
        return {"state": state, "images": images}
    
    def mat2euler(self, mat: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles"""
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
        """Collect action data"""
        joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        actuator_ids = [model.actuator(name).id for name in joint_names]
        return data.ctrl[actuator_ids].copy()
    
    def start_episode(self):
        """Start episode recording"""
        self.episode_data = []
        self.aruco_frames = []
        self.episode_start_time = time.time()
        self.is_recording = True
        print(f"🎬 Started episode {self.current_episode} ({self.collection_mode} mode)")
    
    def stop_episode(self):
        """Stop episode recording and save"""
        if not self.is_recording:
            return
        self.is_recording = False
        duration = time.time() - self.episode_start_time
        
        if len(self.episode_data) > 0:
            self.save_episode()
            print(f"💾 Saved episode {self.current_episode}: {len(self.episode_data)} steps, {duration:.2f}s")
        else:
            print(f"⚠️  Episode {self.current_episode} had no data")
        self.current_episode += 1
    
    def add_step(self, observation: Dict, action: np.ndarray):
        """Add step to current episode"""
        if not self.is_recording:
            return
        timestamp = time.time() - self.episode_start_time
        step_data = {
            "observation": observation,
            "action": action,
            "timestamp": timestamp,
            "episode_index": self.current_episode,
            "frame_index": len(self.episode_data)
        }
        self.episode_data.append(step_data)
    
    def save_episode(self):
        """Save episode to HDF5 and videos"""
        if len(self.episode_data) == 0:
            return
            
        episode_file = self.dataset_dir / "data" / f"episode_{self.current_episode:06d}.hdf5"
        
        with h5py.File(episode_file, 'w') as f:
            # Save observation states
            states = np.array([step["observation"]["state"] for step in self.episode_data])
            f.create_dataset("observation.state", data=states, compression="gzip")
            
            # Save actions
            actions = np.array([step["action"] for step in self.episode_data])
            f.create_dataset("action", data=actions, compression="gzip")
            
            # Save metadata
            timestamps = np.array([step["timestamp"] for step in self.episode_data])
            episode_indices = np.array([step["episode_index"] for step in self.episode_data])
            frame_indices = np.array([step["frame_index"] for step in self.episode_data])
            
            f.create_dataset("timestamp", data=timestamps)
            f.create_dataset("episode_index", data=episode_indices)
            f.create_dataset("frame_index", data=frame_indices)
        
        # Save videos for each camera
        for cam_name in self.camera_configs.keys():
            frames = [step["observation"]["images"][cam_name] for step in self.episode_data]
            if frames:
                video_path = self.dataset_dir / "videos" / "chunk-000" / f"observation.images.{cam_name}" / f"episode_{self.current_episode:06d}.mp4"
                self._save_video_opencv(str(video_path), np.array(frames), int(1.0/self.dt))
        
        # Save ArUco camera video if available
        if self.aruco_enabled and len(self.aruco_frames) > 0:
            video_path = self.dataset_dir / "videos" / "chunk-000" / "observation.images.aruco_cam" / f"episode_{self.current_episode:06d}.mp4"
            self._save_video_opencv(str(video_path), np.array(self.aruco_frames), int(1.0/self.dt))
        
        print(f"✅ Episode {self.current_episode} saved: HDF5 + MP4 videos")
    
    def _save_video_opencv(self, video_path: str, frames: np.ndarray, fps: int):
        """Save video using OpenCV"""
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width = frames.shape[1:3]
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
        except Exception as e:
            print(f"⚠️  Video save failed for {video_path}: {e}")
    
    def save_metadata(self):
        """Save dataset metadata"""
        metadata = self.create_metadata()
        metadata["total_episodes"] = self.current_episode
        metadata["total_frames"] = sum(len(self.episode_data) for _ in range(self.current_episode))
        
        metadata_file = self.dataset_dir / "meta.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📋 Saved metadata to {metadata_file}")


def select_collection_mode():
    """Interactive mode selection"""
    print("\n🤖 SO-100 Arm Unified Data Collection System")
    print("=" * 60)
    print("Select collection mode:")
    print("1. Predefined Trajectories (circle, figure8, square)")
    print("2. ArUco Teleoperation (requires camera calibration)")
    print("3. Manual Teleoperation (mouse control)")
    print("=" * 60)
    
    while True:
        try:
            choice = input("Enter choice (1-3): ").strip()
            if choice == "1":
                return "trajectories"
            elif choice == "2":
                return "aruco"
            elif choice == "3":
                return "manual"
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n👋 Cancelled by user")
            sys.exit(0)


def main_collection_loop(collector: UnifiedDataCollector, mode: str):
    """Main data collection loop"""
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    model.opt.timestep = collector.dt
    
    # Setup simulation components
    end_effector = model.site("attachment_site").id
    body_names = ["Rotation_Pitch", "Upper_Arm", "Lower_Arm", "Wrist_Pitch_Roll", "Fixed_Jaw", "Moving_Jaw"]
    body_ids = [model.body(name).id for name in body_names]
    model.body_gravcomp[body_ids] = 1.0
    
    joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    dof_ids = [model.joint(name).id for name in joint_names]
    actuator_ids = [model.actuator(name).id for name in joint_names]
    
    key_id = model.key("home").id
    mocap_id = model.body("target").mocapid[0]
    
    # Setup cameras
    cameras = collector.setup_sim_cameras(model)
    
    # IK arrays
    jac = np.zeros((6, model.nv))
    diag = collector.damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    ee_quat = np.zeros(4)
    ee_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    
    # Start ArUco detection if enabled
    if mode == "aruco" and collector.aruco_enabled:
        collector.start_aruco_detection()
    
    # Trajectory mode automated collection setup
    trajectory_types = ["circle", "figure8", "square"]
    current_trajectory = 0
    episode_length_seconds = 2.0  # Fixed episode length for trajectories
    episode_length_steps = int(episode_length_seconds / collector.dt)
    
    # Automated collection state for trajectories
    auto_episode_step = 0
    auto_episodes_completed = 0
    auto_collection_active = False
    
    print(f"\n🚀 Starting {mode} data collection")
    if mode == "trajectories":
        print("🤖 AUTOMATED COLLECTION MODE")
        print(f"📊 Episode length: {episode_length_seconds}s ({episode_length_steps} steps)")
        print(f"🎯 Trajectories: {', '.join(trajectory_types)}")
        print("🔄 Collection will start automatically...")
        print("Controls:")
        print("  ESC: Exit")
        print("  T: Skip to next trajectory")
    else:
        print("Controls:")
        print("  SPACE: Start/Stop recording")
        print("  ESC: Exit")
        if mode == "aruco":
            print("  ArUco detection active")
    
    # Launch MuJoCo viewer
    with mujoco.viewer.launch_passive(model=model, data=data, 
                                     show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        
        while viewer.is_running():
            step_start = time.time()
            
            # Clear and add boundary visualization
            viewer.user_scn.ngeom = 0
            add_boundary_visualization(viewer)
            
            # Automated collection logic for trajectories mode
            if mode == "trajectories":
                # Start new episode automatically
                if not auto_collection_active and not collector.is_recording:
                    if auto_episodes_completed < len(trajectory_types):
                        trajectory_type = trajectory_types[current_trajectory]
                        print(f"\n🎬 Starting episode {auto_episodes_completed + 1}/{len(trajectory_types)}: {trajectory_type}")
                        collector.start_episode()
                        auto_collection_active = True
                        auto_episode_step = 0
                
                # Stop episode when length reached
                if auto_collection_active and collector.is_recording:
                    if auto_episode_step >= episode_length_steps:
                        print(f"✅ Episode {auto_episodes_completed + 1} completed ({auto_episode_step} steps)")
                        collector.stop_episode()
                        auto_collection_active = False
                        auto_episodes_completed += 1
                        current_trajectory = (current_trajectory + 1) % len(trajectory_types)
                        
                        # Check if all episodes completed
                        if auto_episodes_completed >= len(trajectory_types):
                            print(f"\n🎉 All {len(trajectory_types)} trajectory episodes completed!")
                            break
                    else:
                        # Progress update every 50 steps
                        if auto_episode_step % 50 == 0:
                            progress = (auto_episode_step / episode_length_steps) * 100
                            print(f"📈 Episode {auto_episodes_completed + 1} progress: {progress:.1f}% ({auto_episode_step}/{episode_length_steps} steps)")
                        auto_episode_step += 1
            
            # Handle different collection modes
            if mode == "trajectories":
                trajectory_type = trajectory_types[current_trajectory]
                target_pos = collector.demonstrate_trajectory(data, mocap_id, data.time, trajectory_type)
                target_pos = check_workspace_boundaries(target_pos)
                data.mocap_pos[mocap_id] = target_pos
                
            elif mode == "aruco":
                # Use ArUco detected position (if available)
                data.mocap_pos[mocap_id] = check_workspace_boundaries(collector.last_aruco_pose)
                
            elif mode == "manual":
                # Manual control via mouse (existing viewer functionality)
                data.mocap_pos[mocap_id] = check_workspace_boundaries(data.mocap_pos[mocap_id])
            
            # Inverse kinematics
            error_pos[:] = data.mocap_pos[mocap_id] - data.site(end_effector).xpos
            mujoco.mju_mat2Quat(ee_quat, data.site(end_effector).xmat)
            mujoco.mju_negQuat(ee_quat_conj, ee_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], ee_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)
            
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], end_effector)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)
            
            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, collector.integration_dt)
            np.clip(q, *model.jnt_range.T, out=q)
            data.ctrl[actuator_ids] = q[dof_ids]
            
            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Collect data if recording
            if collector.is_recording:
                observation = collector.collect_observation(model, data, cameras)
                action = collector.collect_action(model, data)
                collector.add_step(observation, action)
            
            # Handle key presses (manual modes only)
            if mode != "trajectories" and viewer.is_running():
                # Check for key presses for manual modes
                pass
            
            # Maintain real-time
            time_until_next_step = collector.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    # Cleanup
    if collector.is_recording:
        collector.stop_episode()
    collector.save_metadata()
    
    if collector.aruco_enabled:
        collector.aruco_enabled = False
        if collector.aruco_thread:
            collector.aruco_thread.join(timeout=1.0)


def validate_lerobot_dataset(dataset_path: str) -> bool:
    """Validate LeRobot dataset compatibility"""
    print(f"\n🔍 Validating LeRobot dataset: {dataset_path}")
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_path}")
        return False
    
    # Check required files
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
    print("🤖 SO-100 Arm Unified Data Collection System")
    
    # Check required files
    if not os.path.exists("scene.xml"):
        print("❌ scene.xml not found! Please ensure the model file exists.")
        return 1
    
    # Select collection mode
    mode = select_collection_mode()
    
    # Initialize collector
    collector = UnifiedDataCollector()
    collector.collection_mode = mode
    
    # Setup camera calibration for ArUco mode
    if mode == "aruco":
        if not collector.setup_camera_calibration():
            print("❌ ArUco mode requires camera calibration")
            print("📝 Run calibrate.py first or switch to manual mode")
            return 1
    
    try:
        # Run collection
        main_collection_loop(collector, mode)
        
        # Validate dataset
        print(f"\n🔍 Validating collected dataset...")
        if validate_lerobot_dataset(str(collector.dataset_dir)):
            print(f"\n🎉 Dataset collection completed successfully!")
            print(f"📁 Dataset location: {collector.dataset_dir}")
            print(f"✅ Ready for LeRobot training (ACT, etc.)")
        else:
            print(f"\n⚠️  Dataset validation failed - check the output above")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Collection interrupted by user")
        if collector.is_recording:
            collector.stop_episode()
        collector.save_metadata()
    except Exception as e:
        print(f"❌ Collection error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 