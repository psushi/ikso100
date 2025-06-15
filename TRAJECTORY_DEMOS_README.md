# SO-100 Arm Trajectory & Collection Demos

This directory contains several demo scripts to test and visualize the SO-100 arm's predefined trajectories and data collection capabilities.

## 🎯 Available Demo Scripts

### 1. **Trajectory Visualization Demos**

#### `demo_trajectories.py` - Full Trajectory Demo
- **Purpose**: Interactive demonstration of all predefined trajectories
- **Duration**: 30 seconds per trajectory (90 seconds total)
- **Features**:
  - Circle trajectory (smooth circular motion)
  - Figure-8 trajectory (complex 3D path)
  - Square trajectory (precise geometric movements)
  - Real-time inverse kinematics
  - Workspace boundary visualization
- **Controls**: 
  - `SPACE`: Switch to next trajectory
  - `ESC`: Exit demo

```bash
python demo_trajectories.py
```

#### `demo_trajectories_quick.py` - Quick Trajectory Test
- **Purpose**: Fast demonstration for testing (10 seconds each)
- **Duration**: 30 seconds total
- **Features**: Same trajectories as above, but faster switching
- **Use Case**: Quick verification that all trajectories work

```bash
python demo_trajectories_quick.py
```

### 2. **Enhanced Data Collection with Trajectories**

#### `collect_dataset_enhanced_trajectories.py` - Full Dataset Collection
- **Purpose**: Complete dataset collection with predefined trajectories
- **Duration**: ~2.5 minutes (5 episodes × 30 seconds)
- **Features**:
  - **Full dataset collection and saving**
  - Predefined trajectory execution (circle → figure8 → square)
  - Multi-camera video recording (3 angles)
  - Real-time progress tracking and readouts
  - LeRobot v2.1 compatible format
  - HuggingFace visualize_dataset ready
- **Output**: Complete dataset in `datasets/so100_enhanced_trajectories/`

```bash
python collect_dataset_enhanced_trajectories.py
```

#### `demo_quick_trajectory_collection.py` - Quick Collection Demo
- **Purpose**: Fast demonstration of trajectory collection (10 seconds each)
- **Duration**: 30 seconds total (3 episodes)
- **Features**: Same as above but faster for testing
- **Use Case**: Quick verification that collection works properly

```bash
python demo_quick_trajectory_collection.py
```

### 3. **Legacy Data Collection Demos**

#### `demo_collect_with_trajectories.py` - Original System Collection
- **Purpose**: Run full dataset collection using original `collect_dataset.py`
- **Duration**: ~2.5 minutes (5 episodes × 30 seconds)
- **Features**:
  - Uses original `collect_dataset.py` system
  - Automatic trajectory cycling: circle → figure8 → square
  - Multi-camera video recording
  - LeRobot v2.1 compatible dataset format
- **Output**: Complete dataset in `datasets/so100_sim_dataset/`

```bash
python demo_collect_with_trajectories.py
```

#### `demo_enhanced_collection.py` - Enhanced Systems Demo
- **Purpose**: Test enhanced collection systems with video capture
- **Options**:
  1. Manual Collection V2 (ArUco + Mouse teleoperation)
  2. Viewer Collection (Mouse teleoperation + screen capture)
  3. Both systems (sequential)
  4. **Trajectory Demo** (Circle, Figure-8, Square motions)
  5. **Enhanced Trajectory Collection** (Collect dataset with trajectories)
- **Features**: Dependency checking, model validation, comprehensive testing

```bash
python demo_enhanced_collection.py
```

## 🎬 Trajectory Details

### Circle Trajectory
- **Pattern**: Smooth circular motion in XY plane
- **Parameters**:
  - Radius: 0.1 meters
  - Frequency: 0.1 Hz
  - Height: 0.3 meters (fixed Z)
- **Use Case**: Testing smooth continuous motion

### Figure-8 Trajectory  
- **Pattern**: Complex 3D figure-8 with varying height
- **Parameters**:
  - X amplitude: 0.15 meters
  - Y amplitude: 0.1 meters  
  - Z variation: 0.05 meters around 0.25m base
  - Frequency: 0.05 Hz
- **Use Case**: Testing complex multi-axis coordination

### Square Trajectory
- **Pattern**: Precise geometric square path
- **Parameters**:
  - Side length: 0.2 meters
  - Period: 20 seconds per complete cycle
  - Height: 0.3 meters (fixed Z)
- **Use Case**: Testing precise positioning and sharp direction changes

## 🚀 Quick Start Guide

### For Trajectory Visualization Only:
```bash
# Quick test (30 seconds total)
python demo_trajectories_quick.py

# Full interactive demo
python demo_trajectories.py
```

### For Data Collection Testing:
```bash
# Enhanced systems with all options
python demo_enhanced_collection.py

# Original system with trajectories
python demo_collect_with_trajectories.py
```

## 📊 System Requirements

### Required Files:
- `scene.xml` - MuJoCo model with target mocap body
- `main.py` - Workspace bounds and utilities
- `collect_dataset.py` - Original collection system (for some demos)

### Dependencies:
- `mujoco` - Physics simulation
- `numpy` - Mathematical operations  
- `opencv-python` - Video recording (enhanced systems)
- `h5py` - Dataset storage

### Optional (for enhanced features):
- `calib.npz` or `camera_params.json` - ArUco camera calibration
- `aruco_detection.py` - ArUco marker detection

## 🎥 Data Collection Outputs

### Original System (`collect_dataset.py`):
```
datasets/so100_sim_dataset/
├── data/
│   ├── episode_000000.hdf5
│   ├── episode_000001.hdf5
│   └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.cam_high/
│       ├── observation.images.cam_side/
│       └── observation.images.cam_front/
└── meta.json
```

### Enhanced Systems:
```
datasets/so100_manual_dataset_v2/  (or so100_sim_dataset_with_video/)
├── data/
│   └── episode_*.hdf5
├── videos/
│   └── chunk-000/
│       ├── observation.images.cam_high/
│       ├── observation.images.cam_side/
│       ├── observation.images.cam_front/
│       └── observation.images.aruco_cam/  (or viewer_screen/)
└── meta.json
```

## 🔧 Troubleshooting

### Common Issues:

1. **Model not found**: Ensure `scene.xml` exists in current directory
2. **Import errors**: Check that all dependencies are installed
3. **Camera errors**: For ArUco features, ensure camera is connected
4. **Slow performance**: Reduce trajectory frequency or timestep

### Debug Commands:
```bash
# Check model compatibility
python -c "import mujoco; m = mujoco.MjModel.from_xml_path('scene.xml'); print('Model loaded successfully')"

# Test dependencies
python demo_enhanced_collection.py  # Will check all dependencies

# Verify workspace bounds
python -c "from main import WORKSPACE_BOUNDS; print(WORKSPACE_BOUNDS)"
```

## 📈 Performance Metrics

### Trajectory Timing:
- **Simulation rate**: 500 Hz (2ms timestep)
- **Real-time factor**: ~1.0x (matches wall clock time)
- **IK convergence**: <1ms per step
- **Memory usage**: ~50MB per episode

### Data Collection Rates:
- **Original system**: ~100 samples/second (every 5th simulation step)
- **Enhanced systems**: ~500 samples/second (every simulation step)
- **Video frame rate**: 500 FPS (matches simulation)

## 🎯 Use Cases

### Research Applications:
- **Imitation Learning**: Collect demonstration datasets
- **Policy Training**: Generate training data for VLA models
- **Trajectory Optimization**: Test and validate motion planning
- **Sim-to-Real**: Bridge simulation and physical robot control

### Development Testing:
- **System Validation**: Verify all components work correctly
- **Performance Benchmarking**: Measure collection speeds and quality
- **Integration Testing**: Test camera, simulation, and data pipeline
- **Regression Testing**: Ensure updates don't break functionality

---

*All demo scripts are designed to work out-of-the-box with proper dependencies installed. For questions or issues, check the troubleshooting section above.* 