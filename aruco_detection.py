import json
import sys
from typing import Sequence

import cv2
import numpy as np
import numpy.typing as npt
from cv2.aruco import estimatePoseSingleMarkers
from cv2.typing import MatLike
from numpy.matlib import mean
from scipy.spatial.transform import Rotation as R

NAME_TO_ID = {
    "top": 0,
    "bot": 1,
    "front": 2,
    "back": 3,
    "left": 4,
    "right": 5,
}

# Temporary, will be replaced a real cube later.
cube_height = 0.03025
cube_width = 0.0340
cube_depth = 0.03025
half_cube_height = cube_height / 2
half_cube_width = cube_width / 2
half_cube_depth = cube_depth / 2


trans_offset = {
    0: np.array([0.0, 0.0, -half_cube_height]),
    1: np.array([0.0, 0.0, -half_cube_height]),
    2: np.array([0.0, 0.0, -half_cube_depth]),
    3: np.array([0.0, 0.0, -half_cube_depth]),
    4: np.array([0.0, 0.0, -half_cube_width]),
    5: np.array([0.0, 0.0, -half_cube_width]),
}

rot_offset = {
    0: np.eye(3),
    1: np.eye(3),
    2: R.from_euler("x", -90, degrees=True).as_matrix()
    @ R.from_euler("z", 180, degrees=True).as_matrix(),
    3: R.from_euler("x", -90, degrees=True).as_matrix(),
    4: R.from_euler("x", -90, degrees=True).as_matrix()
    @ R.from_euler("z", 90, degrees=True).as_matrix(),
    5: R.from_euler("x", -90, degrees=True).as_matrix()
    @ R.from_euler("z", -90, degrees=True).as_matrix(),
}

MARKER_SIZE = 0.02


def get_com_pose(
    corners: Sequence[MatLike],
    ids: MatLike,
    camera_matrix: npt.NDArray[np.float64],
    distortion_coeff: npt.NDArray[np.float64],
):
    com_rot = []
    com_pos = []

    for corner, id in zip(corners, ids):
        # Estimate the pose of the marker
        rot, pos, _ = cv2.aruco.estimatePoseSingleMarkers(  # pyright: ignore[reportCallIssue]
            corner,  # pyright: ignore[reportArgumentType]
            MARKER_SIZE,
            camera_matrix,
            distortion_coeff,
        )

        if id[0] not in [0, 1, 2, 3, 4, 5]:
            continue

        # Center of mass
        rot_mat = cv2.Rodrigues(rot)[0]
        pos_offset = rot_mat @ trans_offset[id[0]]
        pos_com = pos + pos_offset
        rot_offset_mat = rot_mat @ rot_offset[id[0]]

        com_rot.append(cv2.Rodrigues(rot_offset_mat)[0])
        com_pos.append(pos_com)

    return com_pos, com_rot[0]


def detect_pose(
    frame: MatLike,
    camera_matrix: npt.NDArray[np.float64],
    distortion_coeff: npt.NDArray[np.float64],
):
    """Detect ArUco markers in an image frame.

    Args:
        frame (np.ndarray): The image frame in rgb format with shape HxWx3.
        camera_matrix (np.ndarray): The camera matrix with shape 3x3.
        distortion_coeff (np.ndarray): The distortion coefficients with shape 1x5.
    """
    assert len(frame.shape) == 3 and frame.shape[2] == 3, "image must be rgb"

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    det_params = cv2.aruco.DetectorParameters()
    DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    detector = cv2.aruco.ArucoDetector(DICT, det_params)

    # undistort
    gray = cv2.undistort(gray, camera_matrix, distortion_coeff)

    # Detect the markers
    corners, ids, _ = detector.detectMarkers(gray)

    print(f"Detected {len(corners)} markers")

    frame_vis = frame

    if len(corners) == 0:
        return frame_vis

    com_pos, com_rot = get_com_pose(corners, ids, camera_matrix, distortion_coeff)

    frame_vis = cv2.drawFrameAxes(
        frame,
        camera_matrix,
        distortion_coeff,
        com_rot,
        com_pos,
        MARKER_SIZE * 0.5,
    )

    # Draw the detected markers
    frame_vis = cv2.aruco.drawDetectedMarkers(frame_vis, corners)

    return frame_vis


def aruco_detection(calib_path: str):
    data = np.load(calib_path)
    mtx = data["mtx"]
    dist = data["dist"]
    print("[i] Loaded calibration from", sys.argv[1])

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # last arg only matters on macOS
    if not cap.isOpened():
        sys.exit("Couldn’t open webcam")

    camera_params = {}
    # load json file
    with open("camera_params.json", "r") as f:
        camera_params = json.load(f)

    print(camera_params)

    print("Press q / ESC to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_pose(frame, mtx, dist)

        cv2.imshow("Aruco live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    aruco_detection("calib.npz")
