import numpy as np

ym_per_pix = 30/720
xm_per_pix = 3.7/700


class Lane:
    def __init__(self, initializer):
        a, b, c, d, e = initializer()

        self.left_fitx = a
        self.right_fitx = b
        self.ploty = c
        self.l_fit = d
        self.r_fit = e

    def get_curverad(self):
        left_fit_cr = np.polyfit(
            self.ploty*ym_per_pix, self.left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(
            self.ploty*ym_per_pix, self.right_fitx*xm_per_pix, 2)

        y_eval = np.max(self.ploty)
        left_curverad = (
            (1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**(3/2)) / abs(2*left_fit_cr[0])
        right_curverad = (
            (1 + (2*right_fit_cr[0]*y_eval*xm_per_pix + right_fit_cr[1])**2)**(3/2)) / abs(2*right_fit_cr[0])

        return round((left_curverad + right_curverad) / 2, 2)

    def get_position(self):
        xm_per_pix = 3.7/700
        left_offset = 640 - self.left_fitx[-1]
        right_offset = self.right_fitx[-1] - 640
        return round((left_offset - right_offset) * xm_per_pix, 2)
