import Modules.drawer as drawer
import Modules.colors as colors
import math
import cv2
import numpy as np

def analyze_initial(self) -> None:
    '''
    Calculates elbow angle and heel angle and draws them. Also analyzes heel angle and adds it to lists, but does not draw analysis.
    '''
    elbow_angle = drawer.annotate_angle(
        self, color=colors.green, l_or_r="r",
        pt1=[int(self.shoulder[0]), int(self.shoulder[1])],
        pt2=[int(self.elbow[0]), int(self.elbow[1])],
        pt3=[int(self.wrist[0]), int(self.wrist[1])],
        radius=self.angle_conf.annotation_radius,
    )
    # elbow check here, knee check is in yolo_annotate due to needed the bar position
    if (elbow_angle < self.angle_conf.elbow_angle_threshold):
        self.failed_elbow_check = True
    # toe angle is drawn in 2nd pass because we first need to evaluate if the foot is occluded by the weight or not
    # calculate and append to lists here
    heel_angle = drawer.annotate_angle(
        self, color=colors.light_purple, l_or_r="l",
        pt1=[int(self.heel[0]), int(self.heel[1])],
        pt2=[int(self.toe[0]), int(self.toe[1])],
        pt3=[int(self.heel[0]), int(self.toe[1])],
        radius=self.angle_conf.annotation_radius,
        draw=False,
    )
    self.heel_angle_list.append(heel_angle)
    self.heel_pos_list.append(self.heel)
    self.toe_pos_list.append(self.toe)

# foot is inconsistently detected when occluded by the weight, so only calculate foot angle when not occluded
# occlusion is determined by checking if the foot is close to its "regular" position, if occluded the position signficiantly deviates from the median
def analyze_secondary(self) -> tuple[bool, list[float]]:
    '''
    Second round analysis of angles, for heel specifically.
    If applicable, draws heel angle and returns whether the heel angle ever exceeded the threshold.
    '''
    failed_heel_check: bool = False
    median_foot_point: list[float] = self.toe_pos_list[len(self.toe_pos_list) // 2]
    for i in range(len(self.heel_angle_list)):
        angle: float = self.heel_angle_list[i]
        toe: list[float] = self.toe_pos_list[i]
        if ((math.sqrt((median_foot_point[0] - toe[0]) ** 2 + (median_foot_point[1] - toe[1]) ** 2) < self.angle_conf.toe_radius)):
            if (angle > self.angle_conf.heel_angle_threshold):
                failed_heel_check: bool = True
                break
    return failed_heel_check, median_foot_point


# circle around the median foot point, makes it visually clear when and why the angle is being calculated
def heel_annotation(self, i: int, median_foot_point: list[float]) -> None:
    '''
    Draw the circle around the median foot point to make it visually clear when the angle is being calculated. Calculates and annotates the angle if the foot is in position (i.e. not occluded)
    '''
    cv2.circle(
        img=self.img,
        center=(int(median_foot_point[0]), int(median_foot_point[1])),
        radius=self.angle_conf.toe_radius,
        color=colors.light_red,
        thickness=2,
        lineType=cv2.LINE_AA
    )
    # only draw the angle if foot is not occluded
    toe: list[float] = self.toe_pos_list[i]
    heel: list[float] = self.heel_pos_list[i]
    if (math.sqrt((median_foot_point[0] - toe[0]) ** 2 + (median_foot_point[1] - toe[1]) ** 2) < self.angle_conf.toe_radius):
        # horizontal line from heel to toe
        cv2.line(
            img=self.img,
            pt1=(int(toe[0]), int(toe[1])),
            pt2=(int(heel[0]), int(toe[1])),
            color=colors.red,
            thickness=2,
            lineType=cv2.LINE_AA
        )
        # angle annotation
        drawer.annotate_angle(
            self, color=colors.bright_red, l_or_r="l",
            pt1=[int(heel[0]), int(heel[1])],
            pt2=[int(toe[0]), int(toe[1])],
            pt3=[int(heel[0]), int(toe[1])],
            radius=self.angle_conf.annotation_radius,
            extra_offset=True
        )
    # heel text
    if (self.failed_heel_check):
        drawer.text(
            self,
            text="heel was lifted",
            org=(15, 125),
            color=colors.light_red,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            thickness=2,
            lineType=cv2.LINE_AA
        )
    else:
        drawer.text(
            self,
            text="heel was acceptable",
            org=(15, 125),
            color=colors.light_aqua,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            thickness=2,
            lineType=cv2.LINE_AA
        )

def elbow_annotation(self) -> None:
    '''
    Place the elbow evaluation text on the image, depending on `self.failed_elbow_check`.
    '''
    if (self.failed_elbow_check):
        drawer.text(
            self,
            text="arm was pulling",
            org=(15, 105),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.85,
            color=colors.light_red,
            thickness=2,
            lineType=cv2.LINE_AA
        )
    else:
        drawer.text(
            self,
            text="arm was acceptable",
            org=(15, 105),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.85,
            color=colors.light_aqua,
            thickness=2,
            lineType=cv2.LINE_AA
        )

def knee_annotation(self) -> None:
    if (self.failed_knee_check):
        drawer.text(
            self,
            text="knee was straightened too early",
            org=(15, 85),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.85,
            color=colors.light_red,
            thickness=2,
            lineType=cv2.LINE_AA
        )
    else:
        drawer.text(
            self,
            text="knee was acceptable",
            org=(15, 85),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.85,
            color=colors.light_aqua,
            thickness=2,
            lineType=cv2.LINE_AA
        )
