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

    center_point: tuple[int, int] = (int((self.hip[0] + self.shoulder[0]) / 2), int((self.hip[1] + self.shoulder[1]) / 2))
    rise: int = self.shoulder[1] - self.hip[1]
    run: int = self.shoulder[0] - self.hip[0]
    perp_slope: float = -run / rise
    ls_offset: float = 0.1
    offset_pt: tuple[int, int] = (
        int(self.hip[0] + ls_offset * run),
        int(self.hip[1] + ls_offset * rise)
    )

    cv2.line(
        img=self.img, thickness=1, lineType=cv2.LINE_AA,
        pt1=offset_pt,
        pt2=(0, int(offset_pt[1] - offset_pt[0] * perp_slope)),
        color=colors.black
    )
    cv2.line(
        img=self.img, thickness=1, lineType=cv2.LINE_AA,
        pt1=center_point,
        pt2=(0, int(center_point[1] - center_point[0] * perp_slope)),
        color=colors.black
    )

    lower_pt: tuple[int, int] = _det_int(self, offset_pt, perp_slope)
    upper_pt: tuple[int, int] = _det_int(self, center_point, perp_slope)
    print(lower_pt)
    print(upper_pt)
    merged = cv2.merge([
        self.seg_mask[upper_pt[1]:lower_pt[1], lower_pt[0]:upper_pt[0]] * 255,
        self.seg_mask[upper_pt[1]:lower_pt[1], lower_pt[0]:upper_pt[0]] * 255,
        self.seg_mask[upper_pt[1]:lower_pt[1], lower_pt[0]:upper_pt[0]] * 255,
    ])
    # fill the center region of self.img with red
    self.img[upper_pt[1]:lower_pt[1], lower_pt[0]:upper_pt[0]] = merged

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

def _det_int(self, pt: tuple[int, int], mult: float) -> tuple[int, int]:
    for i in range(int(pt[0])):
        x = int(pt[0]) - i
        y = int(pt[1]) - i * mult
        y_c = math.ceil(y)
        y_f = math.floor(y)
        # print(f"({x}, {y})")
        if (y < 0 or x < 0):
            break
        cv2.circle(img=self.img, center=(x, y_c), radius=2, color=colors.blue, thickness=-1)
        if (self.seg_mask[y_c][x] == False or self.seg_mask[y_f][x] == False):
            cv2.circle(img=self.img, center=(x, y_c), radius=5, color=colors.red, thickness=1)
            return (x, y_c)

    return (0, 0)
