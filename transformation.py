import numpy as np
import cv2
import glob
import helpers

warping_from = np.float32([[200, 720], [604, 450], [696, 450], [1120, 720]])
warping_to = np.float32([[200, 720], [200, 0], [1120, 0], [1120, 720]])


def calibrate():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    images = glob.glob('camera_cal/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret is True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist


def undistort(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    return cv2.resize(undistorted[y:y+h, x:x+w], (1280, 720))


def warp(img):
    M = cv2.getPerspectiveTransform(warping_from, warping_to)
    return cv2.warpPerspective(img, M, (1280, 720)), M
