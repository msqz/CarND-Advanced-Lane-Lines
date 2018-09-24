import numpy as np

ym_per_pix = 30/720
xm_per_pix = 3.7/700
# Define a class to receive the characteristics of each line detection


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.recent_fit = []
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = []
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def get_curverad(self):
        if not self.detected:
            return 0

        ploty = np.linspace(0, 719-1, 720)
        left_fit_cr = np.polyfit(ploty*ym_per_pix, self.bestx*xm_per_pix, 2)

        y_eval = np.max(ploty)
        curverad = (
            (1 + (2*self.best_fit[0]*y_eval*ym_per_pix + self.best_fit[1])**2)**(3/2)) / abs(2*self.best_fit[0])

        return int(curverad)

    def get_line_base_pos(self):
        if not self.detected:
            return 0

        offset = abs(640 - self.bestx[-1])
        return round(offset * xm_per_pix, 2)
