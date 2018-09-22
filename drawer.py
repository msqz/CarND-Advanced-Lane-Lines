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


def extend_canvas(pts):
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


def unwarp_extended(color_warp_ext):
    ''' Extended matrix is now transformed back to be in perspective
    '''
    # --- 116 ms ---
    src = np.copy(transformation.warping_to)
    src[:, 0] += 1280
    src[:, 1] += 720
    dst = np.copy(transformation.warping_from)
    dst[:, 0] += 1280
    dst[:, 1] += 720
    M_ext = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(
        color_warp_ext, M_ext, (1280*3, 720*3))


def crop_to_original(new_warp_ext):
    return new_warp_ext[720:-720, 1280:-1280]


def draw_lane(lane, orig):
    pts_left = np.array(
        [np.transpose(np.vstack([lane.left_fitx, lane.ploty]))])
    pts_right = np.array([
        np.flipud(np.transpose(np.vstack([lane.right_fitx, lane.ploty])))
    ])
    pts = np.hstack((pts_left, pts_right))

    color_warp_ext, lines_ext = extend_canvas(pts)

    new_warp_ext = unwarp_extended(color_warp_ext)

    new_warp_ext = crop_to_original(new_warp_ext)

    weighted = cv2.addWeighted(orig, 1, new_warp_ext, 0.3, 0)

    return weighted, lines_ext


def combine(img, warped, binary, curverad, position):
    warped_sm = cv2.resize(warped, (320, 180))
    img[5:185, -325:-5] = np.dstack((warped_sm, warped_sm, warped_sm))
    binary_sm = cv2.resize(binary, (320, 180))
    img[5:185, -650:-330] = np.dstack((binary_sm, binary_sm, binary_sm))

    draw_text(img, [
        'Radius: {}m'.format(curverad),
        'Position: {}m'.format(position),
    ])

    return img
