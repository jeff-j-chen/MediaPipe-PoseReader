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

    # center point of the spine (hip-shoulder)
    center_point: tuple[int, int] = (int((self.hip[0] + self.shoulder[0]) / 2), int((self.hip[1] + self.shoulder[1]) / 2))
    # for calclating the perpendicular slope
    rise: int = self.shoulder[1] - self.hip[1]
    run: int = self.shoulder[0] - self.hip[0]
    perp_slope: float = -run / rise

    # how much 'upwards' to offset the lower lumbar spine line along the hip-shoulder line
    ls_offset: float = 0.1
    offset_pt: tuple[int, int] = (
        int(self.hip[0] + ls_offset * run),
        int(self.hip[1] + ls_offset * rise)
    )

    # draw the upper and lower lumbar spine estimation lines
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

    # lower point and upper point of the lumbar spine contour
    l_pt: tuple[int, int] = _det_int(self, offset_pt, perp_slope)
    u_pt: tuple[int, int] = _det_int(self, center_point, perp_slope)

    # make sure that the x points aren't reserved
    if (u_pt[0] - l_pt[0] < 0):
        return

    # draw the lumbar spine contour on the image in red, using a stacked true/false
    self.img[u_pt[1]:l_pt[1], l_pt[0]:u_pt[0]] = np.where(
        self.stacked[u_pt[1]:l_pt[1],l_pt[0]:u_pt[0]],
        self.red_stacked[u_pt[1]:l_pt[1],l_pt[0]:u_pt[0]],
        self.img[u_pt[1]:l_pt[1],l_pt[0]:u_pt[0]]
    )

    # draw the 'perfect' lumbar spine (a triangle) onto the screen in bright green, covering the red contour, so that the bad sections 'stick out'
    triangle_pts: list[tuple[int, int]] = [
        (l_pt[0] + self.back_conf.realign, l_pt[1]),
        (u_pt[0], u_pt[1] + self.back_conf.realign),
        (u_pt[0], l_pt[1]),
    ]
    triangle_pts = np.array(triangle_pts) # type: ignore
    overlay = self.img.copy()
    cv2.fillPoly(img=overlay, pts=[triangle_pts], color=colors.bright_green)
    self.img = cv2.addWeighted(src1=overlay, alpha=0.5, src2=self.img, beta=0.5, gamma=0)


    # COMPARISON WITH 'IDEAL' STRAIGHT BACK
    to_count = self.seg_mask[u_pt[1]:l_pt[1], l_pt[0]:u_pt[0]]
    measured_area: int = np.count_nonzero(to_count)
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
# ###################################################################################################

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

def _det_int(self, pt: tuple[int, int], mult: float) -> tuple[int, int]:
    for i in range(int(pt[0])):
        x = int(pt[0]) - i
        y = int(pt[1]) - i * mult
        y_c = math.ceil(y)
        y_f = math.floor(y)
        # print(f"({x}, {y})")
        if (y < 0 or x < 0):
            break
        cv2.circle(img=self.img, center=(x, y_c), radius=2, color=colors.aqua, thickness=-1)
        if (self.seg_mask[y_c][x] == False or self.seg_mask[y_f][x] == False):
            cv2.circle(img=self.img, center=(x, y_c), radius=5, color=colors.red, thickness=1)
            return (x, y_c)

    print("no intersection found")
    return (0, 0)
