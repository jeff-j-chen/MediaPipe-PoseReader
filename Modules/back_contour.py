import math
import cv2
import numpy as np
from Modules.drawer import text
import Modules.colors as colors

# evaluation and drawing of the back contour
def analyze(self) -> None:
    '''
    Performs all necessary analysis for the back contour, and writes the result to the screen.
    '''

    # INITIAL HIP-SHOULDER BOUNDING BOX
    # draw a rectangle around the point from the hip to shoulder
    cv2.rectangle(
        img=self.img, thickness=1, lineType=cv2.LINE_AA,
        pt1=(int(self.hip[0]), int(self.hip[1])),
        pt2=(int(self.shoulder[0]), int(self.shoulder[1])),
        color=colors.light_orange,
    )
    # continue only if the horizontal distance of the extension is > self.contour_bbox_min
    if abs(self.shoulder[0] - self.hip[0]) < self.back_conf.contour_bbox_min:
        _write_contour_text(self, -999)
        return

    # LARGE BOUNDING BOX THRESHOLDING (TO DETERMINE BACK INTERSECTION)
    # extend the bounding box upwards and leftwards, based on the differences of distance
    ext_bbox_bottom_left: tuple[int, int] = ( 0, int(self.hip[1]) )
    ext_bbox_top_right: tuple[int, int] = ( int(self.shoulder[0]), 0 )

    # threshold the extended bounding box, red for contrast, blurred to smooth
    image_b, image_g, image_r = cv2.split(self.img_orig)
    region: np.ndarray = image_g[
        np.clip(ext_bbox_top_right[1], 0, 600):abs(ext_bbox_bottom_left[1]), # y1:y2
        np.clip(ext_bbox_bottom_left[0], 0, 600):abs(ext_bbox_top_right[0]) # x1:x2
    ]
    blurred: np.ndarray = cv2.GaussianBlur(region,(5,5),0)
    ret, thresh = cv2.threshold(
        src=blurred,
        thresh=self.back_conf.thresh_val_init,
        maxval=255,
        type=cv2.THRESH_BINARY
    )

    # erode, then dilate to remove any holes
    kernel: np.ndarray = np.ones((5,5),np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=self.back_conf.erosion_steps)
    thresh = cv2.dilate(thresh, kernel, iterations=self.back_conf.erosion_steps+1)
    # find only the largest contour (assumed to be user's back)
    contours, hierarchy = cv2.findContours(
        image=thresh,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_NONE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) < 1:
        return
    # offset the contour to start at the top left of the extended bounding box


    # DETERMINING THE ACTUAL CONTOUR OF THE BACK
    # split the spine line into thirds
    thirds: np.ndarray = np.linspace(0, 1, num=4)[1:3]
    det_points: list[np.ndarray] = [self.hip + (np.array(self.shoulder) - np.array(self.hip))*t for t in thirds]
    # annotate det_points
    for d in det_points:
        cv2.circle(
            img=self.img,
            center=(int(d[0]), int(d[1])),
            radius=5,
            color=colors.light_red,
            thickness=1,
            lineType=cv2.LINE_AA
        )
    # matches = [0 for i in det_points]

    # for i in range(len(contours)):
    #     # Check if the contour contains the test point
    #     for pt in det_points:
    #         result = cv2.pointPolygonTest(contours[i], pt, False)
    #         # print(result)
    #         # if (result == 0 or result == 1):
    #         #     matches[i] += 1

    # # print(matches)
    # return


    contour = contours[0] + np.array([ext_bbox_bottom_left[0], np.clip(ext_bbox_top_right[1], 0, 600)])
    # overlay the threshold onto the exact same region of the original image
    # img[
    #     np.clip(ext_bbox_top_right[1], 0, 600):abs(ext_bbox_bottom_left[1]), # y1:y2
    #     np.clip(ext_bbox_bottom_left[0], 0, 600):abs(ext_bbox_top_right[0]) # x1:x2
    # ] = np.repeat(thresh[..., np.newaxis], 3, axis=2)

# LOWER LUMBAR DIAGONAL ESTIMATION
################################################################################################
    # -3/4 slope to the right side of the box seems to work reasonably well
    chosen_y_pos: int = int(self.hip[1]) - int(self.hip[0] * 0.75)
    cv2.line(
        img=self.img, thickness=1, lineType=cv2.LINE_AA,
        pt1=(int(self.hip[0]), int(self.hip[1])),
        pt2=(
            ext_bbox_bottom_left[0],
            chosen_y_pos
        ),
        color=colors.light_blue
    )
    bottom_left_intersection = _det_intersect(
        [
            (int(self.hip[0]), int(self.hip[1])),
            (ext_bbox_bottom_left[0], chosen_y_pos)
        ],
        contour[:, 0, :]
    )
    if (bottom_left_intersection is None):
        return

    # DRAWING AND FILTERING LARGE CONTOUR
    # for point in contour:
    #     cv2.circle(img=self.img, center=(point[0][0], point[0][1]), radius=1, color=colors.black, thickness=-1)
    # first pass removes all points left of the intersection and right of the shoulder
    pass_1: list[np.ndarray] = []
    rightmost_point: np.ndarray = np.array([0, 0])
    for point in contour:
        if point[0][0] > bottom_left_intersection[0] and point[0][0] < int(self.shoulder[0] - self.back_conf.contour_cutoff):
            if (point[0][1] > 50 and int(self.shoulder[0]) > 50):
                pass_1.append(point[0])
                if point[0][0] > rightmost_point[0]:
                    rightmost_point = point[0]
    # calculate the middle point of a line from the bottom left to the top right
    middle_x: int = int((bottom_left_intersection[0] + rightmost_point[0]) / 2)
    middle_y: int = int((bottom_left_intersection[1] + rightmost_point[1]) / 2)
    # find the closest point on the curve to the middle point of the line, this gives the upper estimation for the lumbar spine
    nearest_point: np.ndarray = np.array([0, 0])
    dist: float = 600.0
    for point in pass_1:
        cur_dist: float = math.sqrt((point[0] - middle_x)**2 + (point[1] - middle_y)**2)
        if cur_dist < dist:
            dist = cur_dist
            nearest_point = point
    # discard all points above the estimation of the lumbar spine
    actual_contour_pts: list[np.ndarray] = []
    for point in pass_1:
        if (point[1] > nearest_point[1]):
            actual_contour_pts.append(point)


    # DRAWING SPINE VS TRIANGLE
    top_right_intersection = nearest_point
    # fill the user's back contour with red
    lumbar_spine_contour = actual_contour_pts
    lumbar_spine_contour.append(
        np.array([top_right_intersection[0], bottom_left_intersection[1]])
    )
    lumbar_spine_contour = np.array(lumbar_spine_contour)
    cv2.fillPoly(self.img, [lumbar_spine_contour], colors.light_red)

    # overlay the 'ideal' triangle on their back with green, only the red peeks through
    triangle_pts = [
        [bottom_left_intersection[0], bottom_left_intersection[1]],
        [top_right_intersection[0], top_right_intersection[1]],
        [top_right_intersection[0], bottom_left_intersection[1]],
    ]
    triangle_pts = np.array(triangle_pts)
    cv2.fillPoly(img=self.img, pts=[triangle_pts], color=colors.light_aqua)


    # COMPARISON WITH 'IDEAL' STRAIGHT BACK
    # compare aeras
    measured_area: float = cv2.contourArea(lumbar_spine_contour)
    perfect_area: float = cv2.contourArea(triangle_pts)
    if (perfect_area == 0):
        perfect_area = 1
    coeff: float = round(measured_area / perfect_area - 1, 2)
    self.eval_list.append(coeff)
    self.successful_eval_count += 1
    # simple running average to (partially) negate outliers, estimation occasionally will bug out
    avg: float = 0.0
    if (len(self.eval_list) < self.back_conf.running_avg_amount):
        avg = round(sum(self.eval_list) / len(self.eval_list), 2)
    else:
        avg = round(sum(self.eval_list[self.successful_eval_count - self.back_conf.running_avg_amount:self.successful_eval_count]) / self.back_conf.running_avg_amount, 2)
    # print(f"measured area: {measured_area}, perfect area: {perfect_area}")
    # print(f"coeff: {coeff}, avg: {avg}")
###################################################################################################

    _write_contour_text(self, avg)

def _write_contour_text(self, avg: float) -> None:
    '''
    Helper function, writes the result of contour evaluation on the screen.
    '''
    cur_eval: str = "straight"
    cur_color: tuple[int, int, int] = colors.light_aqua
    if (avg == -999):
        cur_eval = "none"
        cur_color = colors.light_blue
    else:
        if (avg > self.back_conf.round_thresh):
            cur_eval = "rounded"
            cur_color = colors.light_red
    text(
        self, text=f"back {cur_eval} ({avg:.2f})",
        org=(15, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=cur_color, thickness=2,
        lineType=cv2.LINE_AA
    )

# determine the intersection between a line (defined by two points) and contour (defined by a list of points)
def _det_intersect(line_endpoints: list[tuple[int, int]], contour_pts: np.ndarray):
    '''
    Helper function, determines intersections between a line and contour, returns the one intersection closest to the bottom right.
    '''
    slope: float = (line_endpoints[1][1] - line_endpoints[0][1]) / (line_endpoints[1][0] - line_endpoints[0][0])
    intercept: float = line_endpoints[0][1] - slope * line_endpoints[0][0]
    # define y and m
    intersections: list[np.ndarray] = []
    # look through every contour point, if it lies close enough to the line, add it to the list of intersections
    for point in contour_pts:
        if abs(point[1] - (slope * point[0] + intercept)) < 3:
            intersections.append(point)
    # if there is an intersection, get the ones closer to the bottom right
    # this assumes that the user is facing right, this makes sure the intersection lies on the user
    # as opposed to being some random point on the contour caused by thresholding
    if len(intersections) > 0:
        closest_intersect: np.ndarray = intersections[0]
        for point in intersections:
            if point[0] > closest_intersect[0] and point[1] > closest_intersect[1]:
                closest_intersect = point
        return closest_intersect
    return None
