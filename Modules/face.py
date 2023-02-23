import torch
import torch.nn as nn
import cv2
import numpy as np
import math

import Modules.colors as colors
from Modules.drawer import text

class FaceDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self._body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
        )
        self._head = nn.Sequential(
            nn.Linear(in_features=576, out_features=576),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(576),

            nn.Linear(in_features=576, out_features=576),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(576),

            nn.Linear(in_features=576, out_features=576),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(576),

            nn.Linear(in_features=576, out_features=576),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(576),

            nn.Linear(in_features=576, out_features=288),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(288),

            nn.Linear(in_features=288, out_features=144),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(144),

            nn.Linear(in_features=144, out_features=72),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(72),

            nn.Linear(in_features=72, out_features=4),
        )

    def forward(self, x):
        x = self._body(x)
        x = x.view(x.size()[0], -1)
        x = self._head(x)
        return x

# draw the face bounding box, calculate the looking direction, and draw it
def analyze(self) -> None:
    '''
    Draw the face bounding box, calculate the looking direction, and draw it.
    '''
    # draw the face region, which is centered at the ear
    if (self.right_ear is not None):
        cv2.rectangle(
            img=self.img,
            pt1=(int(self.right_ear[0] - self.face_conf.face_bound), int(self.right_ear[1] - self.face_conf.face_bound)),
            pt2=(int(self.right_ear[0] + self.face_conf.face_bound), int(self.right_ear[1] + self.face_conf.face_bound)),
            color=colors.light_red,
            thickness=2,
            lineType=cv2.LINE_AA
        )
    # crop the region to be evaluateed by the neural network
    face_region = self.img_orig[
        int(self.right_ear[1] - self.face_conf.face_bound):int(self.right_ear[1] + self.face_conf.face_bound),
        int(self.right_ear[0] - self.face_conf.face_bound):int(self.right_ear[0] + self.face_conf.face_bound)
    ]
    # necessary setup
    conv = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
    tensor = self.face_conf.pt_trans(conv).unsqueeze(0) # pyright: ignore[reportGeneralTypeIssues]
    output = self.model(tensor)
    probs = torch.softmax(output, dim=1)
    # use softmax to predict the angle: either 0 (straight forward), 30, 60, or 90 (straight down)
    angle = self.face_conf.class_angle_dict[torch.argmax(probs).item()]
    self.face_angles.append(angle)

    # fancy stuff to draw the arrow
    tip_length = 15
    tip_x = self.right_ear[0] + 100 * math.cos(math.radians(angle))
    tip_y = self.right_ear[1] + 100 * math.sin(math.radians(angle))
    line1_x1 = tip_x - tip_length * math.cos(math.radians(angle + 25))
    line1_y1 = tip_y - tip_length * math.sin(math.radians(angle + 25))
    line2_x1 = tip_x - tip_length * math.cos(math.radians(angle - 25))
    line2_y1 = tip_y - tip_length * math.sin(math.radians(angle - 25))
    cv2.line(
        img=self.img,
        pt1=(int(self.right_ear[0]), int(self.right_ear[1])),
        pt2=(int(tip_x), int(tip_y)),
        color=colors.red,
        thickness=2,
        lineType=cv2.LINE_AA
    )
    cv2.line(
        img=self.img,
        pt1=(int(line1_x1), int(line1_y1)),
        pt2=(int(tip_x), int(tip_y)),
        color=colors.red,
        thickness=2,
        lineType=cv2.LINE_AA
    )
    cv2.line(
        img=self.img,
        pt1=(int(line2_x1), int(line2_y1)),
        pt2=(int(tip_x), int(tip_y)),
        color=colors.red,
        thickness=2,
        lineType=cv2.LINE_AA
    )

def write_res(self, avg: float, sixty_count: int):
    '''
    Write the result of the face analysis onto the image.
    '''
    if (avg > 20 or sixty_count > 3):
        text(
            self,
            text="don't look down",
            org=(15, 145),
            color=colors.light_red,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            thickness=2,
            lineType=cv2.LINE_AA
        )
    else:
        text(
            self,
            text="look direction acceptable",
            org=(15, 145),
            color=colors.light_aqua,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            thickness=2,
            lineType=cv2.LINE_AA
        )
