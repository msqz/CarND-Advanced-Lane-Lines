#!/usr/bin/python3
import numpy as np
import glob
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time

mtx = None
dist = None


def calibrate_camera():
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


def undistort(img):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    return undistorted[y:y+h, x:x+w]


def warp(img):
    resized = cv2.resize(img, (1280, 720))
    src = np.float32([[250, 720], [608, 450], [690, 450], [1070, 720]])
    dst = np.float32([[250, 720], [250, 0], [1070, 0], [1070, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(resized, M, (1280, 720))


def binarize(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

    binary = np.zeros_like(sx_binary)
    binary[(sx_binary == 1) | (s_binary == 1)] = 1

    return binary


def detect_lanes(img):
    histogram = np.sum(img[img.shape])


def pipeline(img):
    binary = binarize(img)
    undistorted = undistort(binary)
    warped = warp(undistorted)
    lanes = detect_lanes(warped)
    plt.imshow(lanes)
    plt.show()
#     determine_curvature()
#     determine_position()
#     warp_lanes()
#     draw_stats()
#     save_output()


mtx, dist = calibrate_camera()

image = mpimg.imread('test_images/test5.jpg')
pipeline(image)
