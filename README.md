# AprilArm 🪽 (🏅 Winner of HuggingFace LeRobot Hackathon 2025)
## _By Susheendhar Vijay, Nidhin Ninan, and Mohammed Rashad_
### Zero-cost leader arm for the SO-100

> Why buy expensive motors just to read joint angles?

Cost-effective robotic teleoperation using **AprilTags** with a **3D printed controller** to control simulated and real SO-100 robots. Replaces expensive leader-follower setups, halving the cost for imitation learning.

### Required hardware:
- SO-100 robot follower arm
- 3D printed handheld controller (just a hollow cube with a grip)
- Webcam

## 🎬 Demo (brain rot version)
[`1 Min Demo Video`](https://github.com/user-attachments/assets/e8480acb-b422-40d0-b512-5118c11b4a0e)


## 💰 Cost Comparison

| Traditional Setup | Our AprilTag System |
|------------------|-------------------|
| Leader Arm: $100+ & controller | 3D Printed Controller: $5 |
| Follower Arm: $100+ & controller | 122$ & controller |
| **Total: $230** | **Total: $127** |

## 🔧 System Architecture
**Hardware**: 3D printed controller with AprilTags, webcam, SO-100 robot  
**Software**: AprilTag detection, MuJoCo simulation

## 🛠 Setup

**Prerequisites**: Python 3.10+, USB webcam

1. **Install dependencies**:
   ```bash
   uv sync  # or pip install mujoco opencv-contrib-python scipy lerobot[feetech]
   ```

2. **3D print controller**: Print cube from `/assets/`, attach 46mm AprilTags (DICT_APRILTAG_16h5, IDs 0-5)

3. **Calibrate camera**:
   ```bash
   python calibrate.py
   ```

4. **Setup mjpython(macOS)**( :
   ```bash
   bash mjpy-init.sh
   ```

## 🚀 Usage

**Main application**:
```bash
mjpython main.py
```

**Individual components**:
```bash
uv run april.py          # AprilTag detection only
uv run mujoco_loop.py     # MuJoCo simulation only
```

**Controls**:
- **Hand movement**: Move controller to control robot end-effector
- **u/j keys**: Open/close gripper (while focusing cursor in terminal)
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

## 🎓 Applications

- Robotics education
- Research data collection  
- Behavior prototyping
- Remote teleoperation

## 💡 Hackathon Update
We are happy to share that this project was one of the winners of the [HuggingFace LeRobot Hackathon 2025](https://huggingface.co/spaces/LeRobot-worldwide-hackathon/all-winners), where we placed **#24** from over 250+ submissions worldwide **(Top 10%)**, and won a [LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi) as a result. We thank [HuggingFace](https://huggingface.co/) and [Seeedstudio](https://www.seeedstudio.com/) for this award.

---

**Built with**: Python, OpenCV, MuJoCo, NumPy, SciPy
