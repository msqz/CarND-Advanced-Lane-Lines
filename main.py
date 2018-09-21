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
ploty = None


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
    # src = np.float32([[200, 720], [572, 450], [734, 450], [1120, 720]])
    # dst = np.float32([[200, 720], [200, 0], [1120, 0], [1120, 720]])
    src = np.float32([[233, 720], [605, 450], [691, 450], [1087, 720]])
    dst = np.float32([[233, 720], [233, 0], [1087, 0], [1087, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (1280, 720)), M


def draw_lane(img, left_fitx, right_fitx, ploty, orig, M):
    # warp_zero = np.zeros_like(img).astype(np.uint8)
    # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([
        np.flipud(np.transpose(np.vstack([right_fitx, ploty])))
    ])

    pts = np.hstack((pts_left, pts_right))

    # cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # lines = np.copy(img)
    # cv2.polylines(lines, np.int_([pts]), False, (255, 0, 0), 5)
    # Minv = np.linalg.inv(M)

    # newwarp = cv2.warpPerspective(
    #     color_warp, Minv, (orig.shape[1], orig.shape[0]))

    # extending start
    pts_ext = np.copy(pts)
    pts_ext[0, :, 0] += 1280
    pts_ext[0, :, 1] += 720

    color_warp_ext = np.zeros((720*3, 1280*3)).astype(np.uint8)
    color_warp_ext = np.dstack(
        (color_warp_ext, color_warp_ext, color_warp_ext))
    cv2.fillPoly(color_warp_ext, np.int_([pts_ext]), (0, 255, 0))

    lines_ext = np.zeros(img.shape)
    lines_ext = np.hstack((lines_ext, lines_ext, lines_ext))
    lines_ext = np.vstack((lines_ext, lines_ext, lines_ext))
    cv2.polylines(lines_ext, np.int_([pts_ext]), False, (255, 0, 0), 5)

    src = np.float32([[233, 720], [233, 0], [1087, 0], [1087, 720]])
    src[:, 0] += 1280
    src[:, 1] += 720
    dst = np.float32([[233, 720], [605, 450], [691, 450], [1087, 720]])
    dst[:, 0] += 1280
    dst[:, 1] += 720
    M_ext = cv2.getPerspectiveTransform(src, dst)

    new_warp_ext = cv2.warpPerspective(
        color_warp_ext, M_ext, (1280*3, 720*3))

    new_warp_ext = new_warp_ext[720:-720, 1280:-1280]
    # extending end

    return cv2.addWeighted(orig, 1, new_warp_ext, 0.3, 0), lines


def determine_curvature(left_fitx, right_fitx, ploty):
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700

    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

    y_eval = np.max(ploty)
    # Implement the calculation of the left line here
    left_curverad = (
        (1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**(3/2)) / abs(2*left_fit_cr[0])
    # Implement the calculation of the right line here
    right_curverad = (
        (1 + (2*right_fit_cr[0]*y_eval*xm_per_pix + right_fit_cr[1])**2)**(3/2)) / abs(2*right_fit_cr[0])

    return round(left_curverad, 2), round(right_curverad, 2)


def determine_position(left_fitx, right_fitx):
    xm_per_pix = 3.7/700
    left_offset = 640 - left_fitx[-1]
    right_offset = right_fitx[-1] - 640
    return round((left_offset - right_offset) * xm_per_pix, 2)


t_total = []


def pipeline(img):
    global left_fit
    global right_fit
    undistorted = undistort(img)
    binary = threshold.to_binary(undistorted)
    warped, M = warp(binary)
    # left_fitx, right_fitx, ploty = convolution.detect_lanes(warped)
    t = time.clock()
    # if (frame_no % 5 == 0):
    left_fitx, right_fitx, p, l_fit, r_fit = lanes.fit_polynomial(warped)
    # else:
    # left_fitx, right_fitx, p, l_fit, r_fit = lanes.search_around_poly(
    #     warped, left_fit, right_fit)
    # left_fitx, right_fitx, p, l_fit, r_fit = convolution.detect_lanes(warped)
    t_total.append((time.clock() - t) * 1000)
    left_fit = l_fit
    right_fit = r_fit
    ploty = p

    left_curverad, right_curverad = determine_curvature(
        left_fitx, right_fitx, ploty)
    position = determine_position(left_fitx, right_fitx)
    lane, lines = draw_lane(
        warped, left_fitx, right_fitx, ploty, undistorted, M)
    return lane, lines, binary * 255, left_curverad, right_curverad, position


def draw_text(img, lines):
    top_offset = 30
    left_offset = 20

    for i, line in enumerate(lines):
        cv2.putText(img,
                    line,
                    (20, top_offset + top_offset*i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0,),
                    2,
                    cv2.LINE_AA)


def combine(lane, warped, binary, left_curverad, right_curverad, position):
    warped_sm = cv2.resize(warped, (320, 180))
    lane[5:185, -325:-5] = np.dstack((warped_sm, warped_sm, warped_sm))
    binary_sm = cv2.resize(binary, (320, 180))
    lane[5:185, -650:-330] = np.dstack((binary_sm, binary_sm, binary_sm))

    draw_text(lane, [
        'Radius left: {}'.format(left_curverad),
        'Radius right: {}'.format(right_curverad),
        'Position: {}'.format(position),
    ])

    return lane


mtx, dist = calibrate_camera()

if len(sys.argv) != 2:
    raise Exception('missing path')

if sys.argv[1][-4:] == ".mp4":
    reader = skvideo.io.FFmpegReader(sys.argv[1])
    writer = skvideo.io.FFmpegWriter("/home/m/Videos/output.mp4")

    for frame in reader.nextFrame():
        lane, warped, binary, left_curverad, right_curverad, position = pipeline(
            frame)
        combined = combine(lane, warped, binary,
                           left_curverad, right_curverad, position)
        writer.writeFrame(combined)
        frame_no += 1

    reader.close()
    writer.close()
    print(sum(t_total) / len(t_total))
else:
    image = np.asarray(Image.open(sys.argv[1]))
    lane, warped, binary, left_curverad, right_curverad, position = pipeline(
        image)
    combined = combine(lane, warped, binary, left_curverad,
                       right_curverad, position)
    helpers.show(combined)
