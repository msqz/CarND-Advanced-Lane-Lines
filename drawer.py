import numpy as np
import cv2


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


def combine(img, warped, binary, left_curverad, right_curverad, position):
    warped_sm = cv2.resize(warped, (320, 180))
    img[5:185, -325:-5] = np.dstack((warped_sm, warped_sm, warped_sm))
    binary_sm = cv2.resize(binary, (320, 180))
    img[5:185, -650:-330] = np.dstack((binary_sm, binary_sm, binary_sm))

    draw_text(img, [
        'Radius left: {}'.format(left_curverad),
        'Radius right: {}'.format(right_curverad),
        'Position: {}'.format(position),
    ])

    return img
