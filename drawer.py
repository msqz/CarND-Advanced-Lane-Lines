import numpy as np
import cv2
import transformation


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


def extend(pts):
    '''Matrix gets extended to allow drawing full lane edges (they are cropped to the size of the image)'''
    # --- 58 ms ---
    pts_ext = np.copy(pts)
    pts_ext[0, :, 0] += 1280
    pts_ext[0, :, 1] += 720
    color_warp_ext = np.zeros((720*3, 1280*3, 3), dtype=np.uint8)
    cv2.fillPoly(color_warp_ext, np.int_([pts_ext]), (0, 255, 0))
    lines_ext = np.zeros((color_warp_ext.shape[0], color_warp_ext.shape[1]))
    cv2.polylines(lines_ext, np.int_([pts_ext]), False, (255, 0, 0), 15)
    lines_ext = lines_ext[720:-720, 1100:-1100]
    return color_warp_ext, lines_ext


def unwarp(extended):
    '''Extended matrix is now transformed back to be in perspective'''
    # --- 116 ms ---
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


def draw_lane(left, right, orig):
    ploty = np.linspace(0, 719-1, 720)
    pts_left = np.array(
        [np.transpose(np.vstack([left.bestx, ploty]))])
    pts_right = np.array([
        np.flipud(np.transpose(np.vstack([right.bestx, ploty])))
    ])
    pts = np.hstack((pts_left, pts_right))

    extended, lines_ext = extend(pts)

    unwarped = unwarp(extended)

    cropped = crop(unwarped, orig.shape[0], orig.shape[1])

    weighted = cv2.addWeighted(orig, 1, cropped, 0.3, 0)

    return weighted, lines_ext


def combine(img, warped, lines, binary, left_curverad, right_curverad, position):
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
