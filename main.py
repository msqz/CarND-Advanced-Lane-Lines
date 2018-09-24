#!/usr/bin/python3
import numpy as np
import glob
import cv2
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import skvideo.io
import sys

import threshold
import polynomial
import helpers
import transformation
import drawer
from line import Line

mtx, dist = transformation.calibrate()
frame_no = 0
sync_each = 5

left = Line()
right = Line()


def is_valid(left, right):
    # Undetected line forces recalculating from scratch
    if not left.detected or not right.detected:
        return False

    # Checking if the lanes are roughly parallel
    if left.bestx[0] == right.bestx[0] or left.bestx[-1] == right.bestx[-1]:
        return False

    return True


def illustrate_steps(img):
    '''That's only for illustration purpose, not a part of process'''
    undistorted = transformation.undistort(img, mtx, dist)
    a = np.copy(undistorted)
    cv2.polylines(a, np.int_(
        [transformation.warping_from]), False, [255, 0, 0], 3)

    b, c = transformation.warp(undistorted)
    cv2.polylines(b, np.int_(
        [transformation.warping_to]), False, [255, 0, 0], 3)

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(a)
    fig.add_subplot(1, 2, 2)
    plt.imshow(b)
    plt.show()


def pipeline(img):
    # Illustration purpose only
    illustrate_steps(img)

    '''The main process of detecting lane'''
    # 1. Removing distortion from the input image
    undistorted = transformation.undistort(img, mtx, dist)

    # 2. Applying filtering on image to produce binary image of the road
    binary = threshold.to_binary(undistorted)

    # 3. Creating a birds-eye view of the binary image
    warped, M = transformation.warp(binary)

    # 4. Detecting lane pixels.
    # If that's a first frame or the lines are broken,
    # then full window search is performed.
    # Otherwise the search is performed using previous results.
    if frame_no == 0 or not is_valid(left, right):
        polynomial.fit_polynomial(warped, left, right, refit=True)
    else:
        polynomial.search_around_poly(warped, left, right)

    # 5. Calculating curvature
    left_curverad = left.get_curverad()
    right_curverad = right.get_curverad()

    # 6. Calculation position on lane (offset to the left or right)
    position = round(left.get_line_base_pos() - right.get_line_base_pos(), 2)

    # Only for illustration purpose - it pauses the process
    # by displaying image
    #
    # drawer.illustrate(undistorted, left, right)

    # 7. Detected lane is put on top of the road image.
    lane, lines = drawer.draw_lane(undistorted, left, right)

    return lane, warped * 255, lines, binary * 255, left_curverad, right_curverad, position


def photo():
    '''Processing of single image. It's displayed at the end'''
    image = np.asarray(Image.open(sys.argv[1]))[:, :, :3]
    combined = drawer.combine(*pipeline(image))
    helpers.show(combined)


def video():
    '''
    Processing a video. It runs the pipeline for each frame
    and generates an output video file
    '''
    global frame_no
    reader = skvideo.io.FFmpegReader(sys.argv[1])
    writer = skvideo.io.FFmpegWriter("output_videos/project_video.mp4")
    shape = reader.getShape()
    for frame in reader.nextFrame():
        combined = drawer.combine(*pipeline(frame))
        writer.writeFrame(combined)
        frame_no += 1
        print("\r{}%".format(int(frame_no/shape[0] * 100)), end=' ')
        if frame_no == shape[0]:
            print("\r\n")

    reader.close()
    writer.close()


'''
Program works in two modes:

1. main.py path_to_video
Only mp4 extension is supported.
Output video is generated to "output_videos/output.mp4" file

2. main.py path_to_image
It shows output screen at the end of the processing.
'''
if len(sys.argv) != 2:
    raise Exception('missing path')

if sys.argv[1][-4:] == ".mp4":
    video()
else:
    photo()
