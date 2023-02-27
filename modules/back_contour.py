import math
import cv2
import numpy as np
from modules.drawer import text
import modules.colors as colors

# evaluation and drawing of the back contour
def analyze(self) -> None:
    '''
    Performs all necessary analysis for the back contour, and writes the result to the screen.
    '''

    # -3/4 slope to the right side of the box seems to work reasonably well
    chosen_y_pos: int = int(self.hip[1]) - int(self.hip[0] * 0.75)
    cv2.line(
        img=self.img, thickness=1, lineType=cv2.LINE_AA,
        pt1=(int(self.hip[0]), int(self.hip[1])),
        pt2=(0, chosen_y_pos),
        color=colors.light_blue
    )

    for i in range(int(self.hip[0])):
        x = int(self.hip[0]) - i
        y = int(self.hip[1]) - i * 1.33
        y_c = math.ceil(y)
        y_f = math.floor(y)
        # print(f"({x}, {y})")
        if (y < 0 or x < 0):
            break

        if (self.seg_mask[y_c][x] == False or self.seg_mask[y_f][x] == False):
            print(f" found at ({x}, {y})")
            cv2.circle(img=self.img, center=(x, y_c), radius=5, color=colors.red, thickness=-1)
            break

    # draw the perpendicular bisector to the line that connects the hip and shoulder
    cv2.line(
        img=self.img,
        pt1=(int(self.hip[0]), int(self.hip[1])),
        pt2=(int(self.shoulder[0]), int(self.shoulder[1])),
        color=colors.light_red,
        thickness=2,
        lineType=cv2.LINE_AA
    )


#     # DRAWING SPINE VS TRIANGLE
#     top_right_intersection = nearest_point
#     # fill the user's back contour with red
#     lumbar_spine_contour = actual_contour_pts
#     lumbar_spine_contour.append(
#         np.array([top_right_intersection[0], bottom_left_intersection[1]])
#     )
#     lumbar_spine_contour = np.array(lumbar_spine_contour)
#     cv2.fillPoly(self.img, [lumbar_spine_contour], colors.light_red)

#     # overlay the 'ideal' triangle on their back with green, only the red peeks through
#     triangle_pts = [
#         [bottom_left_intersection[0], bottom_left_intersection[1]],
#         [top_right_intersection[0], top_right_intersection[1]],
#         [top_right_intersection[0], bottom_left_intersection[1]],
#     ]
#     triangle_pts = np.array(triangle_pts)
#     cv2.fillPoly(img=self.img, pts=[triangle_pts], color=colors.light_aqua)


#     # COMPARISON WITH 'IDEAL' STRAIGHT BACK
#     # compare aeras
#     measured_area: float = cv2.contourArea(lumbar_spine_contour)
#     perfect_area: float = cv2.contourArea(triangle_pts)
#     if (perfect_area == 0):
#         perfect_area = 1
#     coeff: float = round(measured_area / perfect_area - 1, 2)
#     self.eval_list.append(coeff)
#     self.successful_eval_count += 1
#     # simple running average to (partially) negate outliers, estimation occasionally will bug out
#     avg: float = 0.0
#     if (len(self.eval_list) < self.back_conf.running_avg_amount):
#         avg = round(sum(self.eval_list) / len(self.eval_list), 2)
#     else:
#         avg = round(sum(self.eval_list[self.successful_eval_count - self.back_conf.running_avg_amount:self.successful_eval_count]) / self.back_conf.running_avg_amount, 2)
#     # print(f"measured area: {measured_area}, perfect area: {perfect_area}")
#     # print(f"coeff: {coeff}, avg: {avg}")
# ###################################################################################################

#     _write_contour_text(self, avg)

# def _write_contour_text(self, avg: float) -> None:
#     '''
#     Helper function, writes the result of contour evaluation on the screen.
#     '''
#     cur_eval: str = "straight"
#     cur_color: tuple[int, int, int] = colors.light_aqua
#     if (avg == -999):
#         cur_eval = "none"
#         cur_color = colors.light_blue
#     else:
#         if (avg > self.back_conf.round_thresh):
#             cur_eval = "rounded"
#             cur_color = colors.light_red
#     text(
#         self, text=f"back {cur_eval} ({avg:.2f})",
#         org=(15, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=cur_color, thickness=2,
#         lineType=cv2.LINE_AA
#     )