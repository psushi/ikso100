# Enhanced Video Capture for SO-100 Dataset Collection

This document describes the enhanced video capture functionality added to both manual collection systems, making them fully compatible with LeRobot's visualization tools and modern robotics dataset standards.

## Overview

Both `collect_dataset_manual_v2.py` and `collect_dataset_with_viewer.py` have been enhanced with comprehensive video capture capabilities:

### 🎥 Video Features Added

#### Multi-Camera Simulation Recording
- **cam_high**: High-angle overview (distance: 1.5m, azimuth: 45°, elevation: -30°)
- **cam_side**: Side view for manipulation analysis (distance: 1.2m, azimuth: 90°, elevation: -15°)
- **cam_front**: Front view for end-effector tracking (distance: 1.0m, azimuth: 0°, elevation: -20°)

#### Additional Camera Views
- **aruco_cam**: Real camera feed with ArUco detection overlay (manual V2 only)
- **viewer_screen**: Screen capture of simulation with teleoperation overlays (viewer collection only)

#### Technical Specifications
- **Resolution**: 640x480 for simulation cameras, 640x480 for ArUco, 800x600 for viewer screen
- **Format**: MP4 videos with mp4v codec
- **Frame Rate**: Matches simulation timestep (500 FPS by default)
- **Storage**: Videos saved in `videos/chunk-000/observation.images.{camera_name}/`

## Enhanced Manual Collection V2

### Features
```python
# Camera configuration (from collect_dataset.py)
self.camera_configs = {
    "cam_high": {"distance": 1.5, "azimuth": 45, "elevation": -30, "lookat": [0.0, 0.0, 0.3]},
    "cam_side": {"distance": 1.2, "azimuth": 90, "elevation": -15, "lookat": [0.0, 0.0, 0.3]},
    "cam_front": {"distance": 1.0, "azimuth": 0, "elevation": -20, "lookat": [0.0, 0.0, 0.3]}
}
```

### ArUco Camera Integration
- Real-time camera feed with marker detection
- Pose estimation overlays
- Recording status indicators
- Synchronized frame capture during episodes

### Dataset Structure
```
so100_manual_dataset_v2/
├── data/
│   ├── episode_000000.hdf5
│   └── episode_000001.hdf5
├── videos/
│   └── chunk-000/
│       ├── observation.images.cam_high/
│       │   ├── episode_000000.mp4
│       │   └── episode_000001.mp4
│       ├── observation.images.cam_side/
│       ├── observation.images.cam_front/
│       └── observation.images.aruco_cam/
└── meta.json
```

### Usage
```bash
python collect_dataset_manual_v2.py
```

**Controls:**
- `SPACE`: Start/Stop episode recording
- `A`: Toggle ArUco/Manual control modes
- `H`: Show help
- `ESC`: Exit and save dataset

## Enhanced Viewer Collection

### Features
- Multi-camera simulation recording
- Viewer screen capture with teleoperation overlays
- Mouse-based manual teleoperation
- Automatic episode timing

### Screen Capture Overlays
The viewer screen capture includes:
- Recording status (ON/OFF)
- Current episode number
- Step count and duration
- Control instructions
- Mouse teleoperation guidance

### Dataset Structure
```
so100_sim_dataset_with_video/
├── data/
│   ├── episode_000000.hdf5
│   └── episode_000001.hdf5
├── videos/
│   └── chunk-000/
│       ├── observation.images.cam_high/
│       ├── observation.images.cam_side/
│       ├── observation.images.cam_front/
│       └── observation.images.viewer_screen/
└── meta.json
```

### Usage
```bash
python collect_dataset_with_viewer.py
```

**Controls:**
- `Mouse`: Move target (manual teleoperation)
- `SPACE`: Start/Stop episode recording
- `H`: Show help
- `ESC`: Exit and save dataset

## LeRobot Compatibility

### Metadata Format (v2.1)
```json
{
    "codebase_version": "v2.1",
    "robot_type": "so100_arm",
    "total_videos": 4,
    "features": {
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": 500,
                "video.codec": "mp4v",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        }
    }
}
```

### HuggingFace Visualization
The enhanced datasets are fully compatible with:
- **LeRobot Dataset Visualizer**: https://huggingface.co/spaces/lerobot/visualize_dataset
- **Multi-camera views**: All camera angles visible in the interface
- **Timeline scrubbing**: Synchronized video playback
- **Teleoperation analysis**: Viewer screen shows control methods

## Technical Implementation

### Video Rendering Pipeline
1. **Simulation Cameras**: Use MuJoCo's `mjr_render()` for off-screen rendering
2. **ArUco Integration**: CV2 camera capture with detection overlays
3. **Viewer Capture**: MuJoCo scene rendering with OpenCV text overlays
4. **Synchronization**: Frame-aligned capture during recording episodes

### Performance Optimizations
- Efficient memory management for frame storage
- Threaded ArUco detection to prevent blocking
- Viewport-based rendering for consistent output
- Graceful fallbacks for camera failures

### Error Handling
- Fallback frames for camera failures
- Placeholder videos for missing data
- Robust threading cleanup
- Comprehensive logging

## Installation Requirements

### Additional Dependencies
```bash
pip install opencv-python
```

**Note**: Video recording now uses OpenCV's VideoWriter instead of imageio for better reliability and compatibility.

### Camera Calibration (for ArUco)
Run calibration once for ArUco detection:
```bash
python calibrate.py
```

This generates `calib.npz` or `camera_params.json` required for marker detection.

## Demo Script

Test the enhanced functionality:
```bash
python demo_enhanced_collection.py
```

The demo includes:
- Dependency checking
- Model file validation
- Interactive system selection
- Full feature testing

## Video Analysis

### Dataset Inspection
Use the collected videos for:
- **Policy training**: Multi-view visual inputs
- **Teleoperation analysis**: Understanding human control strategies
- **Behavior verification**: Ensuring correct task execution
- **Data quality assessment**: Identifying anomalies or failures

### Integration with Training
The video data integrates seamlessly with:
- **VLA models**: Vision-Language-Action policies
- **Multi-modal learning**: Combining visual and proprioceptive data
- **Temporal modeling**: Leveraging video sequences for dynamics
- **Sim-to-real transfer**: Visual domain adaptation

## Best Practices

### Recording Tips
1. **Lighting**: Ensure consistent, neutral lighting
2. **Background**: Use clean, uncluttered backgrounds
3. **Movements**: Make smooth, deliberate movements
4. **Duration**: Keep episodes focused and task-oriented
5. **Quality**: Check video playback before proceeding

### Performance Tips
1. **Storage**: Ensure sufficient disk space for videos
2. **Memory**: Monitor RAM usage during long recordings
3. **Processing**: Consider video compression for large datasets
4. **Cleanup**: Remove failed episodes to maintain quality

## Future Enhancements

### Planned Features
- **Depth cameras**: Add depth information to video streams
- **Higher resolutions**: Support for 1080p recording
- **Stereo vision**: Dual-camera setups for depth perception
- **Real-time compression**: Reduce storage requirements
- **Cloud integration**: Direct upload to HuggingFace Hub

### Integration Opportunities
- **ROS2 compatibility**: Bridge with robotic middleware
- **Real robot support**: Extend to physical SO-100 arms
- **Multi-robot**: Coordinate multiple arms simultaneously
- **VR/AR interfaces**: Immersive teleoperation methods

## Troubleshooting

### Common Issues
1. **Missing videos**: Check camera initialization and recording status
2. **Sync problems**: Ensure consistent frame rates across cameras
3. **Storage errors**: Verify disk space and write permissions
4. **Performance**: Reduce video resolution or frame rate if needed

### Debug Commands
```bash
# Check video files
ls -la datasets/*/videos/chunk-000/*/

# Verify metadata
cat datasets/*/meta.json | jq '.features'

# Test video playback
python -c "import cv2; cap = cv2.VideoCapture('path/to/video.mp4'); ret, frame = cap.read(); print(frame.shape if ret else 'Failed to read'); cap.release()"
```

## Conclusion

The enhanced video capture functionality transforms the SO-100 dataset collection systems into comprehensive, modern robotics data gathering tools. With multi-camera recording, ArUco integration, and full LeRobot compatibility, these systems support advanced research in:

- **Visual imitation learning**
- **Multi-modal policy training**
- **Teleoperation analysis**
- **Sim-to-real transfer**

The resulting datasets are immediately compatible with state-of-the-art visualization tools and training pipelines, accelerating robotics research and development. 