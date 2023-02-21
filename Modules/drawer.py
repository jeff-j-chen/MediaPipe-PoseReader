import cv2
import math
import numpy as np

import Modules.colors as colors
from main import PoseAnalyzer

# draw lines between keypoints, e.g. shoulder-elbow...
def draw_lines(self: PoseAnalyzer, img: np.ndarray) -> None:
    '''
    Draws the main skeleotn lines between each detected keypoint.
    '''
    cv2.line(
        img=img, thickness=3, lineType=cv2.LINE_AA,
        pt1=(int(self.shoulder[0]), int(self.shoulder[1])),
        pt2=(int(self.right_ear[0]), int(self.right_ear[1])),
        color=colors.light_aqua,
    )
    cv2.line( # SHOULDER - HIP
        img=img, thickness=3, lineType=cv2.LINE_AA,
        pt1=(int(self.shoulder[0]), int(self.shoulder[1])),
        pt2=(int(self.hip[0]), int(self.hip[1])),
        color=colors.light_aqua,
    )
    cv2.line( # HIP - KNEE
        img=img, thickness=3, lineType=cv2.LINE_AA,
        pt1=(int(self.hip[0]), int(self.hip[1])),
        pt2=(int(self.knee[0]), int(self.knee[1])),
        color=colors.light_blue,
    )
    cv2.line( # KNEE - ANKLE
        img=img, thickness=3, lineType=cv2.LINE_AA,
        pt1=(int(self.knee[0]), int(self.knee[1])),
        pt2=(int(self.ankle[0]), int(self.ankle[1])),
        color=colors.light_blue,
    )
    cv2.line( # ANKLE - HEEL
        img=img, thickness=3, lineType=cv2.LINE_AA,
        pt1=(int(self.ankle[0]), int(self.ankle[1])),
        pt2=(int(self.heel[0]), int(self.heel[1])),
        color=colors.light_purple,
    )
    cv2.line( # ANKLE - TOE
        img=img, thickness=3, lineType=cv2.LINE_AA,
        pt1=(int(self.ankle[0]), int(self.ankle[1])),
        pt2=(int(self.toe[0]), int(self.toe[1])),
        color=colors.light_purple,
    )
    cv2.line( # HEEL - TOE
        img=img, thickness=3, lineType=cv2.LINE_AA,
        pt1=(int(self.heel[0]), int(self.heel[1])),
        pt2=(int(self.toe[0]), int(self.toe[1])),
        color=colors.light_purple,
    )
    cv2.line( # SHOULDER - ELBOW
        img=img, thickness=3, lineType=cv2.LINE_AA,
        pt1=(int(self.shoulder[0]), int(self.shoulder[1])),
        pt2=(int(self.elbow[0]), int(self.elbow[1])),
        color=colors.light_green,
    )
    cv2.line( # ELBOW - WRIST
        img=img, thickness=3, lineType=cv2.LINE_AA,
        pt1=(int(self.elbow[0]), int(self.elbow[1])),
        pt2=(int(self.wrist[0]), int(self.wrist[1])),
        color=colors.light_green,
    )

# draw points on keypoints, e.g. nose, shoulder...
def draw_points(self: PoseAnalyzer, img: np.ndarray) -> None:
    '''
    Draws each detected keypoint as a circle.
    '''
    cv2.circle( # EAR
        img=img, thickness=-1, lineType=cv2.LINE_AA,
        center=(int(self.right_ear[0]), int(self.right_ear[1])),
        radius=5, color=colors.aqua,
    )
    cv2.circle( # SHOULDER
        img=img, thickness=-1, lineType=cv2.LINE_AA,
        center=(int(self.shoulder[0]), int(self.shoulder[1])),
        radius=5, color=colors.aqua,
    )
    cv2.circle( # HIP
        img=img, thickness=-1, lineType=cv2.LINE_AA,
        center=(int(self.hip[0]), int(self.hip[1])),
        radius=5, color=colors.blue,
    )
    cv2.circle( # KNEE
        img=img, thickness=-1, lineType=cv2.LINE_AA,
        center=(int(self.knee[0]), int(self.knee[1])),
        radius=5, color=colors.blue,
    )
    cv2.circle( # ANKLE
        img=img, thickness=-1, lineType=cv2.LINE_AA,
        center=(int(self.ankle[0]), int(self.ankle[1])),
        radius=5, color=colors.purple,
    )
    cv2.circle( # HEEL
        img=img, thickness=-1, lineType=cv2.LINE_AA,
        center=(int(self.heel[0]), int(self.heel[1])),
        radius=5, color=colors.purple,
    )
    cv2.circle( # TOE
        img=img, thickness=-1, lineType=cv2.LINE_AA,
        center=(int(self.toe[0]), int(self.toe[1])),
        radius=5, color=colors.purple,
    )
    cv2.circle( # ELBOW
        img=img, thickness=-1, lineType=cv2.LINE_AA,
        center=(int(self.elbow[0]), int(self.elbow[1])),
        radius=5, color=colors.green,
    )
    cv2.circle( # WRIST
        img=img, thickness=-1, lineType=cv2.LINE_AA,
        center=(int(self.wrist[0]), int(self.wrist[1])),
        radius=5, color=colors.green,
    )

# label an angle with the arc and the degrees, pt2 is the center point
def annotate_angle(img, pt1, pt2, pt3, radius, color, l_or_r, draw=True, extra_offset=False):
    '''
    
    '''
    # define angles
    center = pt2
    angle1 = int(math.degrees(math.atan2(pt1[1] - center[1], pt1[0] - center[0])))
    angle2 = int(math.degrees(math.atan2(pt3[1] - center[1], pt3[0] - center[0])))
    # both require adjusting in different ways so that the arc is drawn correctly
    if (l_or_r == 'r'):
        angle1, angle2 = angle2, angle1
    else:
        angle1 += 360
    # calculate the angle with some trig
    angle = math.degrees(math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) - math.atan2(pt3[1] - pt2[1], pt3[0] - pt2[0]))
    # adjust the text draw point, then write it to the screen
    x, y = pt2
    draw_x = x
    draw_y = y
    # extra offset specifically for the heel angle
    if (l_or_r == 'l'):
        angle += 180
        draw_x -= 75
        if (extra_offset):
            draw_x -= 25
    else:
        angle = 180 - angle
        draw_x += 27
    draw_y += 7
    # don't draw the arc if it will do a full loop around
    draw_arc = True
    if (angle <= 0):
        draw_arc = False
    if (draw):
        if (draw_arc):
            cv2.ellipse(
                img=img,
                center=center,
                axes=(radius, radius),
                angle=0,
                startAngle=angle2,
                endAngle=angle1,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA
            )
        text(
            img=img,
            text=f"{round(angle)}",
            org=(draw_x, draw_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA
        )
    return angle

# custom text drawing function that allows for outline
def text(img, text, org, color, fontFace, fontScale, thickness, lineType, outline_thickness=2, outline_color=(0, 0, 0)):
    if (outline_thickness > 0):
        if (thickness == 1):
            outline_thickness = 1
        cv2.putText(
            img=img,
            text=text,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=outline_color,
            thickness=thickness + outline_thickness,
            lineType=lineType
        )
    cv2.putText(
        img=img,
        text=text,
        org=org,
        fontFace=fontFace,
        fontScale=fontScale,
        color=color,
        thickness=thickness,
        lineType=lineType
    )
