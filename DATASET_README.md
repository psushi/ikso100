# SO-100 Arm Dataset Collection for LeRobot

This project provides a complete dataset collection system for the SO-100 robotic arm simulation, compatible with the LeRobot platform and Hugging Face visualization tools.

## 🎯 Overview

The dataset collection system captures demonstration data from MuJoCo simulations of the SO-100 arm performing various trajectories. The collected data is stored in LeRobot's HDF5 format and can be used for training imitation learning policies.

## 📁 Project Structure

```
├── main.py                           # Original simulation with boundary checks
├── collect_dataset_with_viewer.py    # Main dataset collection script
├── inspect_dataset.py               # Dataset inspection and visualization
├── test_dataset.py                  # Quick test script
├── scene.xml                        # MuJoCo scene file
├── so_arm100.xml                    # SO-100 arm model
└── datasets/
    └── so100_sim_dataset/
        ├── meta.json                 # Dataset metadata
        ├── data/
        │   ├── episode_000000.hdf5   # Episode data files
        │   ├── episode_000001.hdf5
        │   └── episode_000002.hdf5
        └── episode_000000_plot.png   # Generated plots
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
uv add h5py matplotlib
```

### 2. Collect Dataset

Run the dataset collection script with MuJoCo viewer:

```bash
uv run python collect_dataset_with_viewer.py
```

This will:
- Open MuJoCo viewer showing the SO-100 arm simulation
- Collect 3 episodes (20 seconds each) with different trajectories:
  - Episode 1: Circular trajectory
  - Episode 2: Figure-8 trajectory  
  - Episode 3: Square trajectory
- Save data in LeRobot-compatible HDF5 format
- Apply workspace boundary constraints
- Display progress in terminal

### 3. Inspect Dataset

View dataset contents and generate plots:

```bash
uv run python inspect_dataset.py
```

This will show:
- Dataset metadata and statistics
- Episode summaries with duration and step counts
- Sample data from first episode
- Generated trajectory plots

## 📊 Dataset Format

### LeRobot Compatibility

The dataset follows LeRobot's standard format:

```json
{
  "codebase_version": "1.0.0",
  "robot_type": "so100_arm",
  "total_episodes": 3,
  "total_frames": 2014,
  "fps": 500,
  "features": {
    "observation.state": {
      "dtype": "float32",
      "shape": [18],
      "names": ["joint_positions", "joint_velocities", "ee_position", "ee_orientation"]
    },
    "action": {
      "dtype": "float32",
      "shape": [6],
      "names": ["joint_commands"]
    }
  }
}
```

### Data Structure

Each episode contains:

- **Observations** (`observation/state`): 18-dimensional state vector
  - Joint positions (6): `[Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]`
  - Joint velocities (6): Same joint order
  - End-effector position (3): `[x, y, z]` in meters
  - End-effector orientation (3): Euler angles `[roll, pitch, yaw]`

- **Actions** (`action`): 6-dimensional joint commands
  - Target joint positions for each actuator

- **Timestamps** (`timestamp`): Time elapsed since episode start

### Sample Data

```python
# State vector (18 dimensions)
state = [
    # Joint positions (rad)
    -8.64e-10, -1.57, 1.57, 1.57, -1.57, 1.18e-09,
    # Joint velocities (rad/s)  
    -4.32e-07, 6.06e-03, 4.45e-03, -1.93e-06, -5.37e-10, 5.90e-07,
    # End-effector position (m)
    1.57e-18, -2.39e-01, 1.67e-01,
    # End-effector orientation (rad)
    -3.14, 4.24e-06, -1.57
]

# Action vector (6 dimensions)
action = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]  # Joint commands (rad)
```

## 🎮 Trajectory Types

The system generates three types of demonstration trajectories:

### 1. Circular Trajectory
- Radius: 0.1m
- Center: (0, 0)
- Fixed height: 0.3m
- Frequency: 0.1 Hz

### 2. Figure-8 Trajectory
- X amplitude: 0.15m
- Y amplitude: 0.1m
- Z variation: 0.25-0.3m
- Frequency: 0.05 Hz

### 3. Square Trajectory
- Side length: 0.2m
- Fixed height: 0.3m
- 20-second period per cycle

## 🛡️ Workspace Boundaries

The system enforces 3D workspace constraints:

```python
WORKSPACE_BOUNDS = {
    'x_min': -0.3, 'x_max': 0.3,    # ±30cm lateral
    'y_min': -0.3, 'y_max': 0.3,    # ±30cm forward/back  
    'z_min': 0.05, 'z_max': 0.6     # 5cm-60cm height
}
```

When trajectories exceed boundaries, the target is clamped to the nearest valid position.

## 📈 Dataset Statistics

From a typical collection run:

- **Episodes**: 3
- **Total Duration**: ~60 seconds
- **Total Steps**: ~2000
- **Sample Rate**: ~33 Hz (collected every 10th simulation step)
- **File Size**: ~270 KB total
- **Data Format**: HDF5 with gzip compression

## 🔧 Customization

### Modify Collection Parameters

Edit `collect_dataset_with_viewer.py`:

```python
# Change number of episodes and duration
collect_dataset_episodes_with_viewer(
    num_episodes=5,      # Number of episodes
    episode_duration=30.0  # Duration per episode (seconds)
)

# Modify sampling rate
if step_count % 5 == 0:  # Collect every 5th step (higher frequency)
```

### Add New Trajectories

Add custom trajectory functions in `demonstrate_trajectory()`:

```python
elif trajectory_type == "spiral":
    # Custom spiral trajectory
    r = 0.05 + 0.05 * t / 20.0  # Expanding radius
    x = r * np.cos(2 * np.pi * 0.1 * t)
    y = r * np.sin(2 * np.pi * 0.1 * t)  
    z = 0.2 + 0.1 * t / 20.0  # Rising height
    return np.array([x, y, z])
```

### Modify Workspace Bounds

Edit `WORKSPACE_BOUNDS` in `main.py`:

```python
WORKSPACE_BOUNDS = {
    'x_min': -0.4, 'x_max': 0.4,  # Larger workspace
    'y_min': -0.4, 'y_max': 0.4,
    'z_min': 0.1,  'z_max': 0.8
}
```

## 🤖 Integration with LeRobot

### Loading Dataset in LeRobot

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load dataset
dataset = LeRobotDataset("so100_sim_dataset", root="./datasets")

# Access episodes
episode_0 = dataset.get_episode(0)
observations = episode_0["observation"]
actions = episode_0["action"]
```

### Training Policies

The dataset can be used with LeRobot's policy training:

```python
from lerobot.common.policies.act.modeling_act import ACTPolicy

# Initialize policy for SO-100 arm
policy = ACTPolicy(
    config=ACTConfig(
        n_obs_steps=1,
        chunk_size=100,
        n_action_steps=1,
        input_shapes={
            "observation.state": [18],
        },
        output_shapes={
            "action": [6],
        }
    )
)
```

### Hugging Face Visualization

Upload to Hugging Face Hub for visualization:

```python
from lerobot.common.datasets.push_dataset_to_hub import push_dataset_to_hub

push_dataset_to_hub(
    dataset_path="./datasets/so100_sim_dataset",
    repo_id="your-username/so100-sim-dataset",
    tags=["robotics", "so100", "mujoco", "simulation"]
)
```

## 🐛 Troubleshooting

### OpenGL Errors
If you see `gladLoadGL error`, ensure you're running with a display:
- Use `collect_dataset_with_viewer.py` (requires GUI)
- Or run with `DISPLAY` environment variable set

### Missing Dependencies
```bash
uv add h5py matplotlib mujoco
```

### Dataset Validation
Use the inspection script to verify data integrity:
```bash
uv run python inspect_dataset.py
```

## 📝 File Descriptions

- **`collect_dataset_with_viewer.py`**: Main collection script using MuJoCo viewer
- **`inspect_dataset.py`**: Dataset inspection and plotting utilities  
- **`test_dataset.py`**: Quick test with minimal parameters
- **`main.py`**: Original simulation with boundary checking functions

## 🎯 Next Steps

1. **Extend Trajectories**: Add more complex demonstration patterns
2. **Add Vision**: Include camera observations for visual policies
3. **Real Robot**: Adapt for real SO-100 hardware data collection
4. **Policy Training**: Train imitation learning policies using LeRobot
5. **Evaluation**: Test trained policies in simulation and real hardware

## 📄 License

This project is part of the SO-100 arm simulation environment. See main project license for details. 