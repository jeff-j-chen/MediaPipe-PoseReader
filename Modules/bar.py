import cv2
import numpy as np

import Modules.colors as colors
from drawer import annotate_angle

# pre-processing for yolo, from their github
def _preprocess(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup: r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto: dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

# drawing the detected bounding boxes every frame, and the line to the knee/him
def analyze_initial(self, img, img_orig):
    # more yolo preprocessing
    image, ratio, dwdh = _preprocess(img_orig, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    im = image.astype(np.float32)
    im /= 255
    outname = [i.name for i in self.yolo_session.get_outputs()]
    inname = [i.name for i in self.yolo_session.get_inputs()]
    inp = {inname[0]:im}
    outputs = self.yolo_session.run(outname, inp)[0]
    # actual annotation
    bar_count = 0
    weight_count = 0
    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        center_pt = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]
        if (cls_id) == 0:
            self.weight_pt_list.append(center_pt)
            weight_count += 1
        else:
            self.bar_pt_list.append(center_pt)
            bar_count += 1
        cv2.rectangle(
            img=img,
            pt1=box[:2],
            pt2=box[2:],
            color=self.cls_colors[self.cls_names[cls_id]],
            thickness=2
        )
        # draw a line from the center of the bounding box to the knee or hip, whichever is closer in the y axis
        # flag if the knee is straightened and the weight is closer to the knee than the hip, meaning that lifter straighted out too early and are placing excess strain on the lumbar spine
        diff_hip = (box[1] + box[3]) // 2 - self.hip[1]
        diff_knee = (box[1] + box[3]) // 2 - self.knee[1]
        if (abs(diff_hip) < abs(diff_knee)):
            cv2.line(
                img=img,
                pt1=center_pt,
                pt2=(int(self.hip[0]), int(self.hip[1])),
                color=colors.light_purple,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.line(
                img=img,
                pt1=center_pt,
                pt2=(int(self.knee[0]), int(self.knee[1])),
                color=colors.light_purple,
                thickness=2,
                lineType=cv2.LINE_AA
            )

            knee_angle = annotate_angle(
                img, color=colors.blue, l_or_r="l",
                pt1=[int(self.hip[0]), int(self.hip[1])],
                pt2=[int(self.knee[0]), int(self.knee[1])],
                pt3=[int(self.ankle[0]), int(self.ankle[1])],
                radius=self.annotation_radius,
            )

            if (knee_angle > self.knee_angle_threshold):
                self.failed_knee_check = True
    if (bar_count == 0):
        self.bar_pt_list.append([600, 600])
    if (weight_count == 0):
        self.weight_pt_list.append([600, 600])

def analyze_secondary(self):
    # process the points so that the bar path is good
    filtered_bar_pts = _remove_close_points(self.bar_pt_list, self.distance_threshold)
    if (len(filtered_bar_pts) > 0):
        filtered2_bar_pts = _remove_outliers(filtered_bar_pts, 50)
        s_f_bar_pts = _smooth_horizontally(filtered2_bar_pts, self.bar_window_size)
        np_s_f_bar_pts = np.array(s_f_bar_pts)
        x_coords = np_s_f_bar_pts[:, 0]
        # draw a straight line upwards through the median, representing a 'good' bar path
        median_x = sorted(x_coords)[len(x_coords) // 2]
        min_y = min(np_s_f_bar_pts[:, 1])
        max_y = max(np_s_f_bar_pts[:, 1])
        # calculate the stddev of the actual bar path compared to the 'good' bar path
        stddev = round(np.std(x_coords - median_x), 1)
        # color the text based on bar path
        text = "good bar path " if stddev < self.bar_stddev_threshold else "bad bar path "
        text_color = colors.light_red if stddev > self.bar_stddev_threshold else colors.light_aqua

    return np_s_f_bar_pts, median_x, min_y, max_y, stddev, text, text_color

def draw_bar_path(self, img, i, np_s_f_bar_pts, median_x, min_y, max_y, stddev, text, text_color):
    # bar path
    cv2.polylines(
        img=img,
        pts=[np_s_f_bar_pts],
        isClosed=False,
        color=colors.red,
        thickness=5,
        lineType=cv2.LINE_AA
    )
    # ideal path
    cv2.line(
        img=img,
        pt1=(median_x, min_y),
        pt2=(median_x, max_y),
        color=self.bright_green,
        thickness=1,
        lineType=cv2.LINE_AA
    )
    # current location
    cv2.circle(
        img=img,
        center=self.bar_pt_list[i],
        radius=5,
        color=colors.bright_red,
        thickness=-1,
        lineType=cv2.LINE_AA
    )
    # bar text
    text(
        img=img,
        text=f"{text} ({stddev:.2f})",
        org=(15, 60),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=text_color,
        thickness=2,
        lineType=cv2.LINE_AA
    )

# helper for path drawing, remove all points that are too close to each other in the y axis
# eliminates bar staying still and stationary weights
def _remove_close_points(points, threshold):
    filtered_points = []
    for i, pt1 in enumerate(points):
        should_add_point = True
        for j, pt2 in enumerate(points):
            if i == j:
                continue
            # don't add points that are too close together
            if abs(pt1[1] - pt2[1]) < threshold:
                should_add_point = False
                break
        if should_add_point:
            filtered_points.append(pt1)
    return filtered_points

# helper for path drawing, remove all points that are too far from the median in the x axis
def _remove_outliers(points, threshold):
    x_coords = [p[0] for p in points]
    median_x = sorted(x_coords)[len(x_coords) // 2]
    # remove points too far away from the x median
    filtered_points = [p for p in points if abs(p[0] - median_x) < threshold]
    return filtered_points

# helper for path drawing, interpolate the bar path, but only horizontally
def _smooth_horizontally(points, window_size):
    smoothed_points = []
    for i in range(len(points)):
        # compute the average of the x-coordinates within the window
        average_x = 0
        num_neighbors = 0
        for j in range(-window_size, window_size + 1):
            if i + j >= 0 and i + j < len(points):
                average_x += points[i + j][0]
                num_neighbors += 1
        average_x /= num_neighbors
        # y is kept the same
        y = points[i][1]
        smoothed_points.append((int(points[i][0]), y))
    return smoothed_points
