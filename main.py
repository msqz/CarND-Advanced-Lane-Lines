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
from lane import Lane

mtx, dist = transformation.calibrate()
frame_no = 0


def pipeline(img):

    undistorted = transformation.undistort(img, mtx, dist)

    binary = threshold.to_binary(undistorted)

    warped, M = transformation.warp(binary)

    lane = Lane(lambda: lanes.fit_polynomial(warped))

    curverad = lane.get_curverad()

    position = lane.get_position()

    lane, lines = drawer.draw_lane(lane, undistorted)

    return lane, lines, binary * 255, curverad, position


def photo():
    image = np.asarray(Image.open(sys.argv[1]))[:, :, :3]
    combined = drawer.combine(*pipeline(image))
    helpers.show(combined)


def video():
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
