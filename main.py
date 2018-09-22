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
import transformation
import drawer

profile = False
t = time.clock()

mtx = None
dist = None
frame_no = 0
left_fit = None
right_fit = None
ploty = None

warping_from = np.float32([[200, 720], [604, 450], [696, 450], [1120, 720]])
warping_to = np.float32([[200, 720], [200, 0], [1120, 0], [1120, 720]])


def click_stopwatch(action):
    global t
    measured = round(time.clock() - t, 2) * 1000
    print('{}: {}'.format(action, measured))
    t = time.clock()


def warp(img):
    M = cv2.getPerspectiveTransform(warping_from, warping_to)
    return cv2.warpPerspective(img, M, (1280, 720)), M


def draw_lane(img, left_fitx, right_fitx, ploty, orig, M):
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([
        np.flipud(np.transpose(np.vstack([right_fitx, ploty])))
    ])

    pts = np.hstack((pts_left, pts_right))
    pts_ext = np.copy(pts)
    pts_ext[0, :, 0] += 1280
    pts_ext[0, :, 1] += 720
    color_warp_ext = np.zeros((720*3, 1280*3, 3), dtype=np.uint8)
    cv2.fillPoly(color_warp_ext, np.int_([pts_ext]), (0, 255, 0))
    lines_ext = np.zeros((color_warp_ext.shape[0], color_warp_ext.shape[1]))
    cv2.polylines(lines_ext, np.int_([pts_ext]), False, (255, 0, 0), 15)
    lines_ext = lines_ext[720:-720, 1100:-1100]
    # src = np.float32([[233, 720], [233, 0], [1087, 0], [1087, 720]])
    # src = np.float32([[200, 720], [200, 0], [1120, 0], [1120, 720]])
    src = np.copy(warping_to)
    src[:, 0] += 1280
    src[:, 1] += 720
    dst = np.float32([[200, 720], [604, 450], [696, 450], [1120, 720]])
    dst = np.copy(warping_from)
    dst[:, 0] += 1280
    dst[:, 1] += 720
    M_ext = cv2.getPerspectiveTransform(src, dst)

    new_warp_ext = cv2.warpPerspective(
        color_warp_ext, M_ext, (1280*3, 720*3))
    new_warp_ext = new_warp_ext[720:-720, 1280:-1280]
    weighted = cv2.addWeighted(orig, 1, new_warp_ext, 0.3, 0)
    return weighted, lines_ext


def pipeline(img):
    global left_fit
    global right_fit
    undistorted = transformation.undistort(img, mtx, dist)
    binary = threshold.to_binary(undistorted)
    warped, M = warp(binary)
    left_fitx, right_fitx, p, l_fit, r_fit = lanes.fit_polynomial(warped)
    left_fit = l_fit
    right_fit = r_fit
    ploty = p
    left_curverad, right_curverad = geometry.determine_curvature(
        left_fitx, right_fitx, ploty)
    position = geometry.determine_position(left_fitx, right_fitx)
    lane, lines = draw_lane(
        warped, left_fitx, right_fitx, ploty, undistorted, M)
    return lane, lines, binary * 255, left_curverad, right_curverad, position


mtx, dist = transformation.calibrate()

if len(sys.argv) != 2:
    raise Exception('missing path')

if sys.argv[1][-4:] == ".mp4":
    reader = skvideo.io.FFmpegReader(sys.argv[1])
    writer = skvideo.io.FFmpegWriter("/home/m/Videos/output.mp4")
    for frame in reader.nextFrame():
        profile and click_stopwatch('Frame started')
        combined = drawer.combine(*pipeline(frame))
        writer.writeFrame(combined)
        frame_no += 1
        profile and click_stopwatch('Frame ended')

    reader.close()
    writer.close()
else:
    image = np.asarray(Image.open(sys.argv[1]))[:, :, :3]
    combined = drawer.combine(*pipeline(image))
    helpers.show(combined)
