import json
import sys
from typing import Sequence

import cv2
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike
from scipy.spatial.transform import Rotation as R

# ---------- constants ------------------------------------------------------
TAG_FAMILY = cv2.aruco.DICT_APRILTAG_16h5
MARKER_SIZE = 0.046  # 30 mm
CAM_ID = 0

# ---------- cube geometry --------------------------------------------------
cube_length = 0.05
cube_half = cube_length / 2

trans_offset = {  # tag frame → cube-COM translation (metres)
    0: np.array([0, 0, -cube_half]),
    1: np.array([0, 0, -cube_half]),
    2: np.array([0, 0, -cube_half]),
    3: np.array([0, 0, -cube_half]),
    4: np.array([0, 0, -cube_half]),
    5: np.array([0, 0, -cube_half]),
}
rot_offset = {  # tag frame → cube frame rotation
    0: np.eye(3),
    1: np.eye(3),
    2: R.from_euler("x", 90, degrees=True).as_matrix()
    @ R.from_euler("z", 180, degrees=True).as_matrix(),
    3: R.from_euler("x", 90, degrees=True).as_matrix(),
    4: R.from_euler("x", 90, degrees=True).as_matrix()
    @ R.from_euler("z", 90, degrees=True).as_matrix(),
    5: R.from_euler("x", 90, degrees=True).as_matrix(),
}


# ---------- pose smoother --------------------------------------------------
class PoseLPF:
    def __init__(self, a_pos=0, a_rot=0):
        self.a_pos = a_pos
        self.a_rot = a_rot
        self.pos = None
        self.quat = None

    def update(self, t: np.ndarray, r: np.ndarray):
        t = t.reshape(3)
        q = R.from_rotvec(r.reshape(3)).as_quat()
        if self.pos is None:
            self.pos = t
        else:
            self.pos = self.a_pos * self.pos + (1 - self.a_pos) * t
        if self.quat is None:
            self.quat = q
        else:
            if np.dot(self.quat, q) < 0:
                q = -q
            self.quat = self.a_rot * self.quat + (1 - self.a_rot) * q
            self.quat /= np.linalg.norm(self.quat)
        return self.pos.copy(), R.from_quat(self.quat).as_rotvec().reshape(1, 1, 3)


cube_filter = PoseLPF(0.7, 0.7)  # smooth final COM pose
tag_filters: dict[int, PoseLPF] = {}  # per-tag smoothing


# ---------- helpers --------------------------------------------------------
def average_rot(mats: list[np.ndarray]) -> np.ndarray:
    """Return the orthonormal mean of rotation matrices."""
    M = sum(mats) / len(mats)
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def get_tag_filter(tid: int) -> PoseLPF:
    if tid not in tag_filters:
        tag_filters[tid] = PoseLPF(0.5, 0.5)
    return tag_filters[tid]


# ---------- main detection -------------------------------------------------
def detect_and_draw(frame: MatLike, K, D) -> MatLike:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(TAG_FAMILY), cv2.aruco.DetectorParameters()
    )
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return frame

    com_positions = []
    com_rotations = []
    for c, id_arr in zip(corners, ids):
        tid = int(id_arr[0])
        offT = trans_offset.get(tid)
        offR = rot_offset.get(tid)
        if offT is None:
            continue
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(c, MARKER_SIZE, K, D)

        # smooth per-tag pose
        tfilt = get_tag_filter(tid)
        t_s, r_s = tfilt.update(tvec[0], rvec[0])

        # tag frame → cube COM
        R_tag, _ = cv2.Rodrigues(r_s)
        com_pos = t_s + R_tag @ offT
        com_rot = R_tag @ offR
        com_positions.append(com_pos)
        com_rotations.append(com_rot)

        # draw tag axes
        cv2.drawFrameAxes(frame, K, D, r_s, t_s, MARKER_SIZE * 0.4)

    # fuse COM if at least one face visible
    if com_positions:
        pos_avg = np.mean(np.vstack(com_positions), axis=0)
        rot_avg = average_rot(com_rotations)
        pos_sm, rvec_sm = cube_filter.update(pos_avg, cv2.Rodrigues(rot_avg)[0])

        cv2.drawFrameAxes(frame, K, D, rvec_sm, pos_sm, MARKER_SIZE * 0.6)

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    return frame


# ---------- live loop ------------------------------------------------------
def live_demo(calib="calib.npz"):
    data = np.load(calib)
    K, D = data["mtx"], data["dist"]
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        sys.exit("No webcam")
    while True:
        ok, frame = cap.read()
        frame = detect_and_draw(frame, K, D)
        cv2.imshow("AprilTag COM demo", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_demo()
