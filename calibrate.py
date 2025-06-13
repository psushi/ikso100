import glob
import json
import os

import cv2
import numpy as np

# Defining the dimensions of checkerboard
CHECKERBOARD = (8, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob("./photos/*.jpg")
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_shape = gray.shape
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        None,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        # cv2.imshow("img", img)
        # Wait for a keypress, and exit on 'q' or ESC
        # key = cv2.waitKey(0) & 0xFF
        # if key == ord("q") or key == 27:  # 27 is ESC
        # cv2.destroyAllWindows()
        # continue
    else:
        print("Corners not detected")


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None
)

# save the camera calibration matrix


np.savez("calib.npz", mtx=mtx, dist=dist)

# params = {
#     "camera_matrix": [m.tolist() for m in mtx],
#     "dist_coeffs": [d.tolist() for d in dist],
#     "rvecs": [rvec.tolist() for rvec in rvecs],
#     "tvecs": [tvec.tolist() for tvec in tvecs],
# }
#
# print(params)
#
#
# with open("camera_params.json", "w") as f:
#     json.dump(params, f, indent=4)
