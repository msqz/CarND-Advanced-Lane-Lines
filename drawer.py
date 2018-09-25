import matplotlib.pyplot as plt
import numpy as np
import cv2
import transformation
import helpers


def draw_text(img, paragraphs):
    top_offset = 30
    left_offset = 20

    for i, p in enumerate(paragraphs):
        cv2.putText(img,
                    p,
                    (20, top_offset + top_offset*i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0,),
                    2,
                    cv2.LINE_AA)


def expand(pts):
    '''
    Matrix gets extended to allow drawing full lane edges (polynomial graphs).
    Otherwise any f(x) : (x > edge) would be cropped
    '''
    lane = np.zeros((720*3, 1280*3, 3), dtype=np.uint8)

    pts_expanded = np.copy(pts)
    pts_expanded[0, :, 0] += 1280
    pts_expanded[0, :, 1] += 720
    cv2.fillPoly(lane, np.int_([pts_expanded]), (0, 255, 0))

    lines = np.zeros((lane.shape[0], lane.shape[1]))
    cv2.polylines(lines, np.int_([pts_expanded]), False, (255, 0, 0), 15)
    lines = lines[720:-720, 1100:-1100]

    return lane, lines


def unwarp(extended):
    '''Extended matrix is now transformed back to be in perspective'''
    src = np.copy(transformation.warping_to)
    src[:, 0] += 1280
    src[:, 1] += 720
    dst = np.copy(transformation.warping_from)
    dst[:, 0] += 1280
    dst[:, 1] += 720
    M_inv = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(
        extended, M_inv, (1280*3, 720*3))


def crop(new_warp_ext, h, w):
    '''Matrix central area gets extracted'''
    return new_warp_ext[h:-h, w:-w]


def build_points(left, right):
    ploty = np.linspace(0, 719-1, 720)
    pts_left = np.array(
        [np.transpose(np.vstack([left.bestx, ploty]))])
    pts_right = np.array([
        np.flipud(np.transpose(np.vstack([right.bestx, ploty])))
    ])
    return np.hstack((pts_left, pts_right))


def draw_lane(img, left, right):
    if not left.detected or not right.detected:
        return img, np.zeros(img.shape[:2])

    pts = build_points(left, right)
    expanded, lines = expand(pts)
    unwarped = unwarp(expanded)
    cropped = crop(unwarped, img.shape[0], img.shape[1])
    weighted = cv2.addWeighted(img, 1, cropped, 0.3, 0)

    return weighted, lines


def combine(img, warped, lines, binary, left_curverad, right_curverad, position):
    '''The partial results are merged into one frame with the output image'''
    lines_sm = cv2.resize(lines, (240, 180))
    img[5:185, -245:-5] = np.dstack((lines_sm, lines_sm, lines_sm))
    binary_sm = cv2.resize(binary, (240, 180))
    img[5:185, -490:-250] = np.dstack((binary_sm, binary_sm, binary_sm))
    warped_sm = cv2.resize(warped, (240, 180))
    img[5:185, -735:-495] = np.dstack((warped_sm, warped_sm, warped_sm))

    draw_text(img, [
        'Radius left: {}m'.format(left_curverad),
        'Radius right: {}m'.format(right_curverad),
        'Position: {}m'.format(position),
    ])

    return img


def illustrate(undistorted, left, right):
    _, lines = draw_lane(undistorted, left, right)

    offset = (lines.shape[1] - undistorted.shape[1]) // 2
    print(offset)
    out_img = np.zeros((lines.shape[0], lines.shape[1], 3))
    out_img[left.ally, left.allx + offset] = [255, 0, 0]
    out_img[right.ally, right.allx + offset] = [0, 0, 255]
    ploty = np.linspace(0, lines.shape[0] - 1, lines.shape[0])

    fig, ax = plt.subplots()
    ax.imshow(out_img)
    ax.plot(left.bestx + offset, ploty, color="yellow")
    ax.plot(right.bestx + offset, ploty, color="yellow")
    plt.show()
