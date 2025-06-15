# Manual Dataset Collection with ArUco Integration

This document describes the manual dataset collection system for the SO-100 arm simulation, featuring dynamic episode control and ArUco marker teleoperation integration.

## 🎯 Overview

The manual collection system allows you to:
- **Manually control episode start/stop** - Record only when you want to
- **Use ArUco markers for teleoperation** - Real-time pose detection and control
- **Dynamic episode lengths** - Episodes can be any duration you choose
- **Real-time visual feedback** - See recording status and workspace boundaries
- **LeRobot compatibility** - Direct integration with LeRobot training pipeline

## 📁 Files Overview

### Core Scripts
- **`collect_dataset_manual.py`** - Main manual collection script with ArUco integration
- **`demo_manual_collection.py`** - Demo script showing functionality without GUI
- **`inspect_dataset.py`** - Dataset inspection and visualization tool

### Supporting Files
- **`main.py`** - Contains boundary check functions and visualization
- **`aruco_detection.py`** - ArUco detection utilities
- **`calib.npz`** - Camera calibration file (optional for ArUco)

## 🚀 Quick Start

### 1. Basic Manual Collection (Mouse Control)

```bash
# Run manual collection without ArUco
uv run python collect_dataset_manual.py
```

**Controls:**
- **SPACE** - Start/Stop episode recording
- **Mouse** - Move red target box to control arm
- **ESC** - Exit and save dataset

### 2. ArUco Teleoperation Mode

```bash
# Ensure camera calibration exists
ls calib.npz

# Run with ArUco integration
uv run python collect_dataset_manual.py
```

**Controls:**
- **SPACE** - Start/Stop episode recording
- **A** - Toggle between ArUco and Manual control
- **H** - Show help
- **Move ArUco marker** - Control arm in ArUco mode

## 🎮 Detailed Controls

### Keyboard Controls

| Key | Action |
|-----|---------|
| `SPACE` | Start/Stop episode recording |
| `A` | Toggle ArUco/Manual control mode |
| `H` | Show help and control instructions |
| `ESC` | Exit application and save dataset |

### Control Modes

#### 1. Manual Mode (Default)
- Use mouse to drag the red target box in the MuJoCo viewer
- Arm follows the target using inverse kinematics
- Real-time workspace boundary enforcement

#### 2. ArUco Mode (if enabled)
- Move ArUco marker in front of camera
- Target position updates based on marker pose
- Real-time coordinate transformation from camera to robot space

## 📊 Dataset Format

### Episode Structure
Each manually collected episode contains:

```python
episode_000000.hdf5:
├── observation/state: [N, 18]  # Joint states + end-effector pose
├── action: [N, 6]              # Joint commands
├── timestamp: [N]              # Time elapsed since episode start
└── attributes:
    ├── episode_id: 0
    ├── length: N
    ├── duration: X.X seconds
    ├── collection_mode: "manual_teleoperation"
    └── aruco_enabled: True/False
```

### State Vector (18 dimensions)
```python
state = [
    # Joint positions (6): [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]
    q1, q2, q3, q4, q5, q6,
    
    # Joint velocities (6): Same order as positions
    dq1, dq2, dq3, dq4, dq5, dq6,
    
    # End-effector position (3): [x, y, z] in meters
    ee_x, ee_y, ee_z,
    
    # End-effector orientation (3): [roll, pitch, yaw] in radians
    ee_roll, ee_pitch, ee_yaw
]
```

## 🎥 ArUco Integration Details

### Camera Setup
The system automatically detects your camera and applies calibration from `calib.npz`:

```python
# Camera calibration data structure
calib.npz:
├── mtx: [3, 3]    # Camera matrix
└── dist: [1, 5]   # Distortion coefficients
```

### Coordinate Transformation
ArUco marker coordinates are transformed to robot workspace:

```python
# Camera coordinates → Robot coordinates
robot_x = marker_pos[0]              # Left/right
robot_y = -marker_pos[2] + 0.5       # Forward/back (inverted Z)
robot_z = -marker_pos[1] + 0.3       # Up/down (inverted Y)
```

### ArUco Window Features
- **Real-time marker detection** with pose estimation
- **Recording status display** (RECORDING/STANDBY)
- **Control mode indicator** (ArUco Control/Manual Control)
- **3D position display** showing current target coordinates

## 🛡️ Workspace Safety

### Boundary Enforcement
All target positions are automatically constrained to safe workspace limits:

```python
WORKSPACE_BOUNDS = {
    'x_min': -0.3, 'x_max': 0.3,    # ±30cm lateral
    'y_min': -0.3, 'y_max': 0.3,    # ±30cm forward/back
    'z_min': 0.05, 'z_max': 0.6     # 5cm-60cm height
}
```

### Visual Feedback
- **Red wireframe box** shows workspace boundaries in 3D
- **Warning messages** when targets exceed limits
- **Automatic clamping** to nearest valid position

## 📈 Usage Workflow

### Typical Collection Session

1. **Setup**
   ```bash
   uv run python collect_dataset_manual.py
   ```

2. **Choose Control Mode**
   - Press `A` to toggle between Manual/ArUco control
   - Manual: Use mouse to control target
   - ArUco: Move marker in front of camera

3. **Collect Episodes**
   ```
   Press SPACE → Start recording
   Demonstrate task → Teleoperate the arm
   Press SPACE → Stop and save episode
   Repeat → Collect multiple demonstrations
   ```

4. **Exit and Inspect**
   ```bash
   Press ESC → Exit collection
   
   # Inspect collected data
   uv run python inspect_dataset.py ./datasets/so100_manual_dataset
   ```

### Example Session Output

```
🚀 Starting Manual Dataset Collection with ArUco Integration
✅ ArUco detection enabled with calibration from calib.npz
🎥 ArUco detection thread started - Press 'q' in ArUco window to stop

✅ Ready for data collection!
📍 Press SPACE to start recording your first episode

🎬 Started recording episode 0
🔴 Recording Episode 0: 150 steps, 5.2s | Mode: ArUco
💾 Saved episode 0: 150 steps, 5.20s duration
✅ Episode 0 saved successfully

📊 Final Dataset Summary:
   Total episodes collected: 3
   Dataset location: datasets/so100_manual_dataset
   ArUco detection: Enabled
✅ Dataset ready for use with LeRobot!
```

## 🔧 Customization

### Modify ArUco Coordinate Transformation

Edit the transformation in `collect_dataset_manual.py`:

```python
# Custom coordinate mapping
robot_x = marker_pos[0] * scale_factor
robot_y = -marker_pos[2] + y_offset
robot_z = -marker_pos[1] + z_offset
```

### Adjust Sampling Rate

Change data collection frequency:

```python
# Collect every N steps (lower = higher frequency)
if collector.is_recording and step_count % 5 == 0:  # Every 5th step
    observation = collector.collect_observation(model, data)
    action = collector.collect_action(model, data)
    collector.add_step(observation, action)
```

### Modify Workspace Bounds

Update boundaries in `main.py`:

```python
WORKSPACE_BOUNDS = {
    'x_min': -0.4, 'x_max': 0.4,  # Larger workspace
    'y_min': -0.4, 'y_max': 0.4,
    'z_min': 0.1,  'z_max': 0.8
}
```

## 🤖 LeRobot Integration

### Loading Manual Dataset

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load manually collected dataset
dataset = LeRobotDataset("so100_manual_dataset", root="./datasets")

# Access episode data
episode = dataset.get_episode(0)
observations = episode["observation"]["state"]
actions = episode["action"]
```

### Training with Manual Data

```python
from lerobot.common.policies.act.modeling_act import ACTPolicy

# Train policy on manual demonstrations
policy = ACTPolicy(
    config=ACTConfig(
        input_shapes={"observation.state": [18]},
        output_shapes={"action": [6]}
    )
)

# Train on collected demonstrations
trainer.fit(policy, datamodule=dataset)
```

## 🐛 Troubleshooting

### ArUco Detection Issues

**Problem**: No ArUco markers detected
```bash
# Check camera access
ls /dev/video*

# Verify calibration file
python -c "import numpy as np; print(np.load('calib.npz').files)"
```

**Problem**: Poor marker tracking
- Ensure good lighting conditions
- Use high-contrast markers
- Keep marker flat and visible
- Check camera focus

### Performance Issues

**Problem**: Slow recording/lag
```python
# Reduce sampling frequency in collect_dataset_manual.py
if step_count % 20 == 0:  # Sample less frequently
```

**Problem**: Large file sizes
- Reduce episode duration
- Lower sampling rate
- Use compression in HDF5 files

### Dataset Validation

```bash
# Verify dataset integrity
uv run python inspect_dataset.py ./datasets/so100_manual_dataset

# Check episode contents
h5dump datasets/so100_manual_dataset/data/episode_000000.hdf5
```

## 📝 Comparison with Automatic Collection

| Feature | Manual Collection | Automatic Collection |
|---------|-------------------|---------------------|
| Episode Length | Dynamic (user controlled) | Fixed duration |
| Control | Real-time start/stop | Predefined trajectories |
| ArUco Integration | ✅ Full support | ❌ Not integrated |
| Teleoperation | ✅ Mouse + ArUco | ❌ Trajectory only |
| Data Quality | High (human supervised) | Consistent (automated) |
| Use Case | Human demonstrations | Synthetic data |

## 🎯 Next Steps

1. **Collect Demonstrations**: Use the manual system to gather human demonstrations
2. **Train Policies**: Use LeRobot to train imitation learning policies
3. **Real Robot Integration**: Adapt for real SO-100 hardware
4. **Advanced Features**: Add force feedback, visual observations, etc.

## 📄 Files Summary

- **`collect_dataset_manual.py`** (589 lines) - Complete manual collection system
- **`demo_manual_collection.py`** (79 lines) - Functionality demonstration
- **`MANUAL_COLLECTION_README.md`** - This documentation

The manual collection system provides maximum flexibility for creating high-quality demonstration datasets while maintaining full compatibility with the LeRobot ecosystem. 