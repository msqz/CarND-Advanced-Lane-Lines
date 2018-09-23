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
import polynomial
import convolution
import helpers
import transformation
import drawer
from line import Line

mtx, dist = transformation.calibrate()
frame_no = 0
sync_each = 5

left = Line()
right = Line()


def pipeline(img):

    undistorted = transformation.undistort(img, mtx, dist)

    binary = threshold.to_binary(undistorted)

    warped, M = transformation.warp(binary)

    if frame_no % 5 == 0:
        polynomial.fit_polynomial(warped, left, right)
    else:
        polynomial.search_around_poly(warped, left, right)

    left_curverad = 10  # left.get_curverad()
    right_curverad = 10  # right.get_curverad()

    position = 10  # lane.get_position()

    lane, lines = drawer.draw_lane(left, right, undistorted)

    return lane, warped * 255, lines, binary * 255, left_curverad, right_curverad, position


def photo():
    image = np.asarray(Image.open(sys.argv[1]))[:, :, :3]
    combined = drawer.combine(*pipeline(image))
    helpers.show(combined)


def video():
    global frame_no
    reader = skvideo.io.FFmpegReader(sys.argv[1])
    writer = skvideo.io.FFmpegWriter("/home/m/Videos/output.mp4")
    for frame in reader.nextFrame():
        combined = drawer.combine(*pipeline(frame))
        writer.writeFrame(combined)
        frame_no += 1

    reader.close()
    writer.close()


if len(sys.argv) != 2:
    raise Exception('missing path')

if sys.argv[1][-4:] == ".mp4":
    video()
else:
    photo()
