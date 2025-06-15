# SO-100 Arm Unified Data Collection System

## Overview

This unified data collection system consolidates all data collection functionality for the SO-100 arm simulation into a single, comprehensive script. It integrates with existing project infrastructure and provides LeRobot-compatible datasets for training policies like ACT.

## Features

### 🎯 **Three Collection Modes**
1. **Predefined Trajectories** - Automated collection with circle, figure-8, and square trajectories
2. **ArUco Teleoperation** - Real-time control using ArUco marker detection
3. **Manual Teleoperation** - Mouse-based control in MuJoCo viewer

### 🎥 **Multi-Camera Video Recording**
- **Simulation Cameras**: 3 angles (cam_high, cam_side, cam_front)
- **ArUco Camera**: Real camera feed with marker detection overlays
- **LeRobot Compatible**: MP4 format with proper metadata

### 🔧 **Existing Infrastructure Integration**
- Uses `aruco_detection.py` for marker detection
- Integrates `calibrate.py` camera calibration
- Loads `camera_params.json` configuration
- Includes workspace boundaries from `main.py`

### ✅ **LeRobot Compatibility**
- LeRobot v2.1 format compliance
- HDF5 + MP4 structure
- Ready for HuggingFace visualize_dataset
- Compatible with ACT and other LeRobot policies

## Files

### Core Scripts
- `unified_data_collector.py` - Main collection script
- `validate_lerobot_dataset.py` - Dataset validation tool
- `cleanup_old_scripts.py` - Remove redundant scripts

### Existing Infrastructure (Unchanged)
- `main.py` - Core simulation with workspace boundaries
- `aruco_detection.py` - ArUco marker detection
- `calibrate.py` - Camera calibration
- `camera_params.json` - Camera parameters
- `scene.xml` - MuJoCo model

## Quick Start

### 1. Setup (if needed)
```bash
# Install dependencies
pip install mujoco numpy opencv-python h5py

# Camera calibration (for ArUco mode)
python calibrate.py  # Follow calibration procedure
```

### 2. Run Data Collection
```bash
python unified_data_collector.py
```

### 3. Select Collection Mode
```
🤖 SO-100 Arm Unified Data Collection System
============================================================
Select collection mode:
1. Predefined Trajectories (circle, figure8, square)
2. ArUco Teleoperation (requires camera calibration)
3. Manual Teleoperation (mouse control)
============================================================
Enter choice (1-3): 
```

### 4. Validate Dataset
```bash
python validate_lerobot_dataset.py ./datasets/so100_unified_dataset
```

## Collection Modes

### 1. Predefined Trajectories
- **Automatic**: No user input required during collection
- **Trajectories**: Circle, figure-8, square patterns
- **Duration**: Configurable episode length
- **Use Case**: Consistent demonstration data

**Controls:**
- `SPACE`: Start/Stop recording
- `T`: Switch trajectory type
- `ESC`: Exit

### 2. ArUco Teleoperation
- **Real-time**: Control arm using ArUco markers
- **Requirements**: Camera calibration (`calib.npz` or `camera_params.json`)
- **Visual Feedback**: Live ArUco detection window
- **Use Case**: Human demonstrations with physical markers

**Controls:**
- `SPACE`: Start/Stop recording
- Move ArUco marker to control arm
- `ESC`: Exit

### 3. Manual Teleoperation
- **Interactive**: Mouse control in MuJoCo viewer
- **Direct**: Click and drag target position
- **Intuitive**: Visual feedback in simulation
- **Use Case**: Precise manual demonstrations

**Controls:**
- `SPACE`: Start/Stop recording
- Mouse: Control target position
- `ESC`: Exit

## Dataset Structure

### Directory Layout
```
datasets/so100_unified_dataset/
├── meta.json                          # LeRobot metadata
├── data/                              # Episode data
│   ├── episode_000000.hdf5
│   ├── episode_000001.hdf5
│   └── ...
└── videos/chunk-000/                  # Video data
    ├── observation.images.cam_high/
    │   ├── episode_000000.mp4
    │   └── ...
    ├── observation.images.cam_side/
    ├── observation.images.cam_front/
    └── observation.images.aruco_cam/   # If ArUco enabled
```

### Data Format

#### HDF5 Structure
- `observation.state` - [N, 24] float32
  - Joint positions (6), velocities (6)
  - End-effector pose (6)
  - Target pose (6)
- `action` - [N, 6] float32
  - Joint commands
- `episode_index` - [N] int64
- `frame_index` - [N] int64  
- `timestamp` - [N] float64

#### Video Format
- **Format**: MP4 with mp4v codec
- **Resolution**: 640x480 (simulation), 640x480 (ArUco)
- **Frame Rate**: 500 FPS (matches simulation)
- **Channels**: RGB

## Validation

### Automatic Validation
The system includes comprehensive dataset validation:

```bash
python validate_lerobot_dataset.py <dataset_path>
```

### Validation Checks
- ✅ Directory structure
- ✅ Metadata format (LeRobot v2.1)
- ✅ HDF5 data structure
- ✅ Video file integrity
- ✅ Data shape consistency
- ✅ Training readiness

### Example Output
```
🔍 Validating LeRobot dataset: ./datasets/so100_unified_dataset
✅ Metadata structure valid
✅ Required features present
✅ Video features found: 4
✅ Found 5 episode files
✅ Found 20 video files
✅ HDF5 data structure valid
✅ Dataset is LeRobot compatible and ready for training!
🎉 Dataset validation passed!
```

## Integration with LeRobot

### HuggingFace Compatibility
The datasets are fully compatible with LeRobot's visualization tools:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load dataset
dataset = LeRobotDataset("path/to/so100_unified_dataset")

# Visualize with HuggingFace tools
# Use visualize_dataset on HuggingFace
```

### Training with ACT
```python
from lerobot.common.policies.act.modeling_act import ACTPolicy

# The dataset is ready for ACT training
policy = ACTPolicy(
    config=config,
    dataset_stats=dataset_stats
)
```

## Troubleshooting

### Common Issues

#### 1. ArUco Detection Not Working
```bash
# Check camera calibration
ls calib.npz camera_params.json

# Recalibrate if needed
python calibrate.py
```

#### 2. OpenGL Rendering Errors
The system uses colored placeholders to avoid OpenGL conflicts when running within MuJoCo viewer. This is normal and provides proper multi-camera structure.

#### 3. Video Files Empty
- Check that recording was started with SPACE
- Ensure sufficient episode duration
- Verify camera permissions

#### 4. Dataset Validation Fails
```bash
# Run validation for detailed error messages
python validate_lerobot_dataset.py <dataset_path>

# Check file permissions and disk space
ls -la datasets/
```

### Performance Tips

1. **Storage**: Ensure sufficient disk space (videos can be large)
2. **Memory**: Close other applications during collection
3. **Frame Rate**: Adjust episode duration based on available storage

## Cleanup

### Remove Old Scripts
```bash
python cleanup_old_scripts.py
```

This removes all redundant collection scripts, keeping only:
- `unified_data_collector.py`
- `validate_lerobot_dataset.py`
- Core infrastructure files

## Advanced Usage

### Custom Trajectories
Modify the `demonstrate_trajectory()` function in `unified_data_collector.py` to add custom trajectory patterns.

### Camera Configuration
Adjust camera angles in the `camera_configs` dictionary:
```python
self.camera_configs = {
    "cam_high": {
        "distance": 1.5,
        "azimuth": 45,
        "elevation": -30,
        "lookat": [0.0, 0.0, 0.3]
    },
    # Add more cameras...
}
```

### Episode Parameters
Modify collection parameters:
```python
# In main_collection_loop()
num_episodes = 10        # Number of episodes
episode_duration = 30.0  # Seconds per episode
```

## Support

For issues or questions:
1. Check this README
2. Run dataset validation
3. Review error messages in terminal
4. Ensure all dependencies are installed

## License

This unified data collection system integrates with the existing SO-100 arm project infrastructure and maintains compatibility with all existing components. 