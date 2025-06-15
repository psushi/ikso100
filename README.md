# AprilArm 🪽 
Zero cost leader arm for the SO-100

> Why buy expensive motors just to read joint angles?

Cost-effective robotic teleoperation using **AprilTags** with a **3D printed controller** to control simulated and real SO-100 robots. Replaces expensive leader-follower setups, halving the cost for imitation learning.

### Required hardware:
- SO-100 robot follower arm
- 3D printed handheld controller (just a hollow cube with a grip)
- Webcam

## 🎬 Demo
[`1 Min Demo Video`](./347-IlliniRoboticsCrashDummies.mp4)

## 💰 Cost Comparison

| Traditional Setup | Our AprilTag System |
|------------------|-------------------|
| Leader Arm: $100+ & controller | 3D Printed Controller: $5 |
| Follower Arm: $100+ & controller | 122$ & controller |
| **Total: $230** | **Total: $127** |

## 🔧 System Architecture
**Hardware**: 3D printed controller with AprilTags, webcam, SO-100 robot  
**Software**: AprilTag detection, MuJoCo simulation, inverse kinematics, multi-threading

## 🛠 Setup

**Prerequisites**: Python 3.10+, USB webcam, 3D printer

1. **Install dependencies**:
   ```bash
   uv sync  # or pip install mujoco opencv-contrib-python scipy lerobot[feetech]
   ```

2. **3D print controller**: Print cube from `/assets/`, attach 46mm AprilTags (DICT_APRILTAG_16h5, IDs 0-5)

3. **Calibrate camera**:
   ```bash
   python calibrate.py
   ```

4. **Setup MuJoCo** (if needed):
   ```bash
   bash mjpy-init.sh
   ```

## 🚀 Usage

**Main application**:
```bash
python main.py
```

**Individual components**:
```bash
python april.py          # AprilTag detection only
python mujoco_loop.py     # MuJoCo simulation only
```

**Controls**:
- **Hand movement**: Move controller to control robot end-effector
- **w/s keys**: Open/close gripper
- **ESC/Q**: Exit

## 🧠 How It Works

**AprilTag Detection**: Multi-face detection, pose fusion, temporal smoothing, coordinate transformation

**Robot Control**: Inverse kinematics, Jacobian-based control, gravity compensation, joint limits

**Architecture**: Two parallel threads communicate via queue - AprilTag detection (`april.py`) and MuJoCo simulation (`mujoco_loop.py`) for real-time performance

## 📁 Files

- `main.py` - Main application
- `april.py` - AprilTag detection 
- `mujoco_loop.py` - MuJoCo simulation
- `calibrate.py` - Camera calibration
- `scene.xml` - MuJoCo scene
- `so_arm100.xml` - Robot model
- `assets/` - 3D printable STL files

## 🔬 Technical Details

**AprilTags**: DICT_APRILTAG_16h5 family, 46mm size, IDs 0-5  
**Camera**: Behind robot, 45° downward, auto-calibrated  
**Robot**: 6 DOF (Rotation, Pitch, Elbow, Wrist Pitch, Wrist Roll, Jaw)

## 🎓 Applications

- Robotics education
- Research data collection  
- Behavior prototyping
- Remote teleoperation

---

**Built with**: Python, OpenCV, MuJoCo, NumPy, SciPy
