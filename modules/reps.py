import numpy as np
import cv2

def det_start_end(self, img_list: list[np.ndarray]) -> tuple[int, int]:
    '''
    Predicts the start and end of a single lift. Does not take into account the number of reps, it only captures a single one for the time being.
    '''
    start = 0
    end = 0
    # if we want to crop the start and end of the lift
    if (self.analysis_conf.start_end):
        # calculate the difference in bar position at every frame
        bar_diffs = []
        print("number of points when determining start and end: ", len(self.bar_pt_list))
        for i in range(len(self.bar_pt_list) - 1):
            bar_diffs.append(self.bar_pt_list[i + 1][1] - self.bar_pt_list[i][1])
            # annotate the bar diff onto the image
        # start is declared when there are 3 consecutive frames of upward bar movement
        for i in range(len(bar_diffs) - 2):
            if (bar_diffs[i] < 0 and bar_diffs[i + 1] < 0 and bar_diffs[i + 2] < 0):
                start = i
                break
        # mark the as the frame where the bar reaches its highest point
        # remember that the smaller the y is, the higher in the image it is
        highest_bar_y = min(np.array(self.bar_pt_list)[:, 1])
        for i in range(len(self.bar_pt_list)):
            if (self.bar_pt_list[i][1] <= highest_bar_y):
                end = i
                break
    if end == 0: end = len(img_list)-1
    # only output between the start and end frames
    return start, end
