#!/usr/bin/python3
import numpy as np
import glob
import cv2
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import skvideo.io
import time
import sys

import threshold
import lanes
import convolution
import helpers

mtx = None
dist = None
frame_no = 0
left_fit = None
right_fit = None


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
    return cv2.resize(undistorted[y:y+h, x:x+w], (1280, 720))


def warp(img):
    src = np.float32([[233, 720], [605, 450], [691, 450], [1087, 720]])
    dst = np.float32([[233, 720], [233, 0], [1087, 0], [1087, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (1280, 720)), M


def draw_lane(img, left_fitx, right_fitx, ploty, orig, M):
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([
        np.flipud(np.transpose(np.vstack([right_fitx, ploty])))
    ])

    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    Minv = np.linalg.inv(M)
    newwarp = cv2.warpPerspective(
        color_warp, Minv, (orig.shape[1], orig.shape[0]))
    return cv2.addWeighted(orig, 1, newwarp, 0.3, 0)


def pipeline(img):
    global left_fit
    global right_fit
    undistorted = undistort(img)
    binary = threshold.to_binary(undistorted)
    warped, M = warp(binary)
    #left_fitx, right_fitx, ploty = convolution.detect_lanes(warped)
    if (frame_no % 5 == 0):
        left_fitx, right_fitx, ploty, l_fit, r_fit = lanes.fit_polynomial(
            warped)
    else:
        left_fitx, right_fitx, ploty, l_fit, r_fit = lanes.search_around_poly(
            warped, left_fit, right_fit)

    left_fit = l_fit
    right_fit = r_fit
#     determine_curvature()
#     determine_position()
    lane = draw_lane(warped, left_fitx, right_fitx, ploty, undistorted, M)
    return lane, warped * 255, binary * 255
#     draw_stats()
#     save_output()


mtx, dist = calibrate_camera()

if len(sys.argv) != 2:
    raise Exception('missing path')

if sys.argv[1][-4:] == ".mp4":
    reader = skvideo.io.FFmpegReader(sys.argv[1])
    writer_lane = skvideo.io.FFmpegWriter("/home/m/Videos/output-lane.mp4")
    writer_warped = skvideo.io.FFmpegWriter("/home/m/Videos/output-warped.mp4")
    writed_combined = skvideo.io.FFmpegWriter("/home/m/Videos/output-combined.mp4")

    for frame in reader.nextFrame():
        lane, warped, binary = pipeline(frame)
        combined = np.copy(lane)
        
        warped_sm = cv2.resize(warped, (320, 180))
        combined[:180, -320:] = np.dstack((warped_sm, warped_sm, warped_sm))

        binary_sm = cv2.resize(binary, (320, 180))
        combined[:180, -320*2:-320] = np.dstack((binary_sm, binary_sm, binary_sm))

        writer_lane.writeFrame(lane)
        writer_warped.writeFrame(np.dstack((warped, warped, warped)))
        writed_combined.writeFrame(combined)
        frame_no += 1

    reader.close()
    writer_lane.close()
    writer_warped.close()
else:
    image = np.asarray(Image.open(sys.argv[1]))
    helpers.show(pipeline(image))
