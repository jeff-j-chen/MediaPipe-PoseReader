import cv2
import math
from tqdm import tqdm
import mediapipe as mp
import numpy as np
import onnxruntime as ort

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.8
)

class PoseAnalyzer:
    def __init__(self, in_path, out_path, session, distance_threshold, bar_window_size, contour_bbox_min, thresh_val_init, erosion_steps, contour_cutoff, thresh_val_refine, estimation_extension, round_thresh, running_avg_amount, static_image_mode, model_complexity, enable_segmentation, min_detection_confidence, min_tracking_confidence):
        self.light_red = (204, 36, 29)
        self.light_green = (152, 151, 26)
        self.light_yellow = (215, 153, 33)
        self.light_blue = (69, 133, 136)
        self.light_purple = (177, 98, 134)
        self.light_aqua = (104, 157, 106)
        self.light_orange = (214, 93, 14)
        self.red = (157, 0, 6)
        self.green = (121, 116, 14)
        self.yellow = (181, 118, 20)
        self.blue = (7, 102, 120)
        self.purple = (143, 63, 113)
        self.aqua = (66, 123, 88)
        self.orange = (175, 58, 3)
        self.bright_red = (255, 0, 0)
        self.bright_green = (0, 255, 0)
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)

        # video setup
        self.in_path = in_path
        self.out_path = out_path

        # parameters for back contour detection
        self.contour_bbox_min = contour_bbox_min
        self.thresh_val_init = thresh_val_init
        self.erosion_steps = erosion_steps
        self.contour_cutoff = contour_cutoff
        self.thresh_val_refine = thresh_val_refine
        self.estimation_extension = estimation_extension
        self.round_thresh = round_thresh
        self.running_avg_amount = running_avg_amount
        self.successful_eval_count = 0
        self.eval_list = []

        self.cap = cv2.VideoCapture(self.in_path)
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_output = cv2.VideoWriter(
            filename=self.out_path,
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=int(self.cap.get(cv2.CAP_PROP_FPS)),
            frameSize=(self.w, self.h)
        )
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # mediapipe settings
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.enable_segmentation = enable_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # initialize mediapipe
        self.mp_pose = mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            enable_segmentation=self.enable_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.shoulder = self.elbow = self.wrist = self.hip = self.knee = self.ankle = self.heel = self.toe = self.nose = self.eye = None

        # yolo
        self.session = session
        self.cls_names = ["weight", "bar"]
        self.cls_colors = {
            self.cls_names[0]: self.light_red,
            self.cls_names[1]: self.red
        }
        self.bar_pt_list = []
        self.weight_pt_list = []
        self.distance_threshold = distance_threshold
        self.bar_window_size = bar_window_size

    def fully_analyze(self):
        print("initial analysis...")
        img_list = []
        for i in tqdm(range(self.video_length)):
            # IMPORT
            # read frame and convert to RGB
            ret, img = self.cap.read()
            img_orig = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # KEYPOINT ANALYSIS
            # get the coords of keypoints
            keypoints = pose.process(img)
            # defines keypoints (self.shoulder, self.elbow, ...)
            lm = keypoints.pose_landmarks
            lm_pose = mp_pose.PoseLandmark
            self.read_keypoints(lm, lm_pose)

            # ANNOTATIONS
            self.draw_lines(img)
            self.draw_points(img)

            # ANALYSIS
            # draw the contour of the back and evaluate
            self.back_contour(img, img_orig)
            # evalate position of bar and weight using yolo
            self.yolo_annotate(img, img_orig)

            # EXPORT
            # write the frame # bottom left of the video
            cv2.putText(img, f"Frame: {i}", (20, self.h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, self.white, 1, cv2.LINE_AA)
            # convert back to BGR and save the frame
            img_list.append(img)

        print("refining...")
        # yolo similar keypoint rejection
        filtered_weight_pts = self.remove_close_points(self.weight_pt_list, self.distance_threshold)
        filtered_bar_pts = self.remove_close_points(self.bar_pt_list, self.distance_threshold)
        # s_f_weight_pts = self.smooth_points(filtered_weight_pts, self.bar_window_size)
        # s_f_bar_pts = self.smooth_points(filtered_bar_pts, self.bar_window_size)
        for i in tqdm(range(len(img_list))):
            img = img_list[i]
            cv2.polylines(
                img=img,
                pts=[np.array(filtered_weight_pts)],
                isClosed=False,
                color=self.light_red,
                thickness=5,
                lineType=cv2.LINE_AA
            )
            cv2.polylines(
                img=img,
                pts=[np.array(filtered_bar_pts)],
                isClosed=False,
                color=self.red,
                thickness=5,
                lineType=cv2.LINE_AA
            )
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.video_output.write(img)

        self.cap.release()
        self.video_output.release()

    def read_keypoints(self, lm, lmPose):
        if (lm is None):
            print('fail')
            return
        self.shoulder = [lm.landmark[lmPose.RIGHT_SHOULDER].x   * self.w, lm.landmark[lmPose.RIGHT_SHOULDER].y   * self.h]
        self.elbow    = [lm.landmark[lmPose.RIGHT_ELBOW].x      * self.w, lm.landmark[lmPose.RIGHT_ELBOW].y      * self.h]
        self.wrist    = [lm.landmark[lmPose.RIGHT_WRIST].x      * self.w, lm.landmark[lmPose.RIGHT_WRIST].y      * self.h]
        self.hip      = [lm.landmark[lmPose.RIGHT_HIP].x        * self.w, lm.landmark[lmPose.RIGHT_HIP].y        * self.h]
        self.knee     = [lm.landmark[lmPose.RIGHT_KNEE].x       * self.w, lm.landmark[lmPose.RIGHT_KNEE].y       * self.h]
        self.ankle    = [lm.landmark[lmPose.RIGHT_ANKLE].x      * self.w, lm.landmark[lmPose.RIGHT_ANKLE].y      * self.h]
        self.heel     = [lm.landmark[lmPose.RIGHT_HEEL].x       * self.w, lm.landmark[lmPose.RIGHT_HEEL].y       * self.h]
        self.toe      = [lm.landmark[lmPose.RIGHT_FOOT_INDEX].x * self.w, lm.landmark[lmPose.RIGHT_FOOT_INDEX].y * self.h]
        self.nose     = [lm.landmark[lmPose.NOSE].x             * self.w, lm.landmark[lmPose.NOSE].y             * self.h]
        self.eye      = [lm.landmark[lmPose.RIGHT_EYE].x        * self.w, lm.landmark[lmPose.RIGHT_EYE].y        * self.h]
        self.right_ear = [lm.landmark[lmPose.RIGHT_EAR].x       * self.w, lm.landmark[lmPose.RIGHT_EAR].y       * self.h]

    def draw_lines(self, img):
        cv2.line( # NOSE - EYE
            img=img, thickness=3, lineType=cv2.LINE_AA,
            pt1=(int(self.nose[0]), int(self.nose[1])),
            pt2=(int(self.eye[0]), int(self.eye[1])),
            color=(self.light_aqua),
        )
        cv2.line( # NOSE - SHOULDER
            img=img, thickness=3, lineType=cv2.LINE_AA,
            pt1=(int(self.shoulder[0]), int(self.shoulder[1])),
            pt2=(int(self.eye[0]), int(self.eye[1])),
            color=self.light_aqua,
        )
        cv2.line( # SHOULDER - HIP
            img=img, thickness=3, lineType=cv2.LINE_AA,
            pt1=(int(self.shoulder[0]), int(self.shoulder[1])),
            pt2=(int(self.hip[0]), int(self.hip[1])),
            color=self.light_aqua,
        )
        cv2.line( # HIP - KNEE
            img=img, thickness=3, lineType=cv2.LINE_AA,
            pt1=(int(self.hip[0]), int(self.hip[1])),
            pt2=(int(self.knee[0]), int(self.knee[1])),
            color=self.light_blue,
        )
        cv2.line( # KNEE - ANKLE
            img=img, thickness=3, lineType=cv2.LINE_AA,
            pt1=(int(self.knee[0]), int(self.knee[1])),
            pt2=(int(self.ankle[0]), int(self.ankle[1])),
            color=self.light_blue,
        )
        cv2.line( # ANKLE - HEEL
            img=img, thickness=3, lineType=cv2.LINE_AA,
            pt1=(int(self.ankle[0]), int(self.ankle[1])),
            pt2=(int(self.heel[0]), int(self.heel[1])),
            color=self.light_purple,
        )
        cv2.line( # ANKLE - TOE
            img=img, thickness=3, lineType=cv2.LINE_AA,
            pt1=(int(self.ankle[0]), int(self.ankle[1])),
            pt2=(int(self.toe[0]), int(self.toe[1])),
            color=self.light_purple,
        )
        cv2.line( # HEEL - TOE
            img=img, thickness=3, lineType=cv2.LINE_AA,
            pt1=(int(self.heel[0]), int(self.heel[1])),
            pt2=(int(self.toe[0]), int(self.toe[1])),
            color=self.light_purple,
        )
        cv2.line( # SHOULDER - ELBOW
            img=img, thickness=3, lineType=cv2.LINE_AA,
            pt1=(int(self.shoulder[0]), int(self.shoulder[1])),
            pt2=(int(self.elbow[0]), int(self.elbow[1])),
            color=self.light_green,
        )
        cv2.line( # ELBOW - WRIST
            img=img, thickness=3, lineType=cv2.LINE_AA,
            pt1=(int(self.elbow[0]), int(self.elbow[1])),
            pt2=(int(self.wrist[0]), int(self.wrist[1])),
            color=self.light_green,
        )

    def draw_points(self, img):
        cv2.circle( # NOSE
            img=img, thickness=-1, lineType=cv2.LINE_AA,
            center=(int(self.nose[0]), int(self.nose[1])),
            radius=5, color=self.aqua,
        )
        cv2.circle( # EYE
            img=img, thickness=-1, lineType=cv2.LINE_AA,
            center=(int(self.eye[0]), int(self.eye[1])),
            radius=5, color=self.aqua,
        )
        cv2.circle( # SHOULDER
            img=img, thickness=-1, lineType=cv2.LINE_AA,
            center=(int(self.shoulder[0]), int(self.shoulder[1])), 
            radius=5, color=self.aqua,
        )
        cv2.circle( # HIP
            img=img, thickness=-1, lineType=cv2.LINE_AA,
            center=(int(self.hip[0]), int(self.hip[1])),
            radius=5, color=self.blue,
        )
        cv2.circle( # KNEE
            img=img, thickness=-1, lineType=cv2.LINE_AA,
            center=(int(self.knee[0]), int(self.knee[1])),
            radius=5, color=self.blue,
        )
        cv2.circle( # ANKLE
            img=img, thickness=-1, lineType=cv2.LINE_AA,
            center=(int(self.ankle[0]), int(self.ankle[1])),
            radius=5, color=self.purple,
        )
        cv2.circle( # HEEL
            img=img, thickness=-1, lineType=cv2.LINE_AA,
            center=(int(self.heel[0]), int(self.heel[1])),
            radius=5, color=self.purple,
        )
        cv2.circle( # TOE
            img=img, thickness=-1, lineType=cv2.LINE_AA,
            center=(int(self.toe[0]), int(self.toe[1])),
            radius=5, color=self.purple,
        )
        cv2.circle( # ELBOW
            img=img, thickness=-1, lineType=cv2.LINE_AA,
            center=(int(self.elbow[0]), int(self.elbow[1])),
            radius=5, color=self.green,
        )
        cv2.circle( # WRIST
            img=img, thickness=-1, lineType=cv2.LINE_AA,
            center=(int(self.wrist[0]), int(self.wrist[1])),
            radius=5, color=self.green,
        )

    def back_contour(self, img, img_orig):

# INITIAL HIP-SHOULDER BOUNDING BOX
################################################################################################
        # draw a rectangle around the point from the hip to shoulder
        cv2.rectangle(
            img=img, thickness=1, lineType=cv2.LINE_AA,
            pt1=(int(self.hip[0]), int(self.hip[1])),
            pt2=(int(self.shoulder[0]), int(self.shoulder[1])),
            color=self.light_orange,
        )
        # continue only if the horizontal distance of the extension is > self.contour_bbox_min
        if abs(self.shoulder[0] - self.hip[0]) < self.contour_bbox_min:
            return
################################################################################################


# LARGE BOUNDING BOX THRESHOLDING (TO DETERMINE BACK INTERSECTION)
################################################################################################
        # extend the bounding box upwards and leftwards, based on the differences of distance
        ext_bbox_bottom_left = ( 0, int(self.hip[1]) )
        ext_bbox_top_right = ( int(self.shoulder[0]), 0 )
        
        # threshold the extended bounding box, red for contrast, blurred to smooth
        image_b, image_g, image_r = cv2.split(img_orig)
        region = image_g[
            np.clip(ext_bbox_top_right[1], 0, 600):abs(ext_bbox_bottom_left[1]), # y1:y2
            np.clip(ext_bbox_bottom_left[0], 0, 600):abs(ext_bbox_top_right[0]) # x1:x2
        ]
        blurred = cv2.GaussianBlur(region,(5,5),0)
        ret, thresh = cv2.threshold(blurred, self.thresh_val_init, 255, cv2.THRESH_BINARY)

        # erode, then dilate to remove any holes
        kernel = np.ones((5,5),np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=self.erosion_steps)
        thresh = cv2.dilate(thresh, kernel, iterations=self.erosion_steps+1)
        # find only the largest contour (assumed to be user's back)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours) < 1:
            return
        # offset the contour to start at the top left of the extended bounding box
        contour = contours[0] + np.array([ext_bbox_bottom_left[0], np.clip(ext_bbox_top_right[1], 0, 600)])


# LOWER LUMBAR DIAGONAL ESTIMATION
################################################################################################
        # -3/4 slope to the right side of the box seems to work reasonably well
        chosen_y_pos = int(self.hip[1]) - int(self.hip[0] * 0.75)
        cv2.line(
            img=img, thickness=1, lineType=cv2.LINE_AA,
            pt1=(int(self.hip[0]), int(self.hip[1])),
            pt2=(
                ext_bbox_bottom_left[0],
                chosen_y_pos
            ),
            color=self.light_blue
        )
        bottom_left_intersection = self.determine_intersect(
            [
                (int(self.hip[0]), int(self.hip[1])),
                (ext_bbox_bottom_left[0], chosen_y_pos)
            ],
            contour[:, 0, :]
        )
################################################################################################

# DRAWING AND FILTERING LARGE CONTOUR
################################################################################################
        # for point in contour:
        #     cv2.circle(img=img, center=(point[0][0], point[0][1]), radius=1, color=self.black, thickness=-1)
        # first pass removes all points left of the intersection and right of the shoulder
        pass_1 = []
        rightmost_point = (0, 0)
        for point in contour:
            if point[0][0] > bottom_left_intersection[0] and point[0][0] < int(self.shoulder[0] - self.contour_cutoff):
                if (point[0][1] > 50 and int(self.shoulder[0]) > 50):
                    pass_1.append(point[0])
                    if point[0][0] > rightmost_point[0]:
                        rightmost_point = point[0]
        # calculate the middle point of a line from the bottom left to the top right
        middle_x = int((bottom_left_intersection[0] + rightmost_point[0]) / 2)
        middle_y = int((bottom_left_intersection[1] + rightmost_point[1]) / 2)
        # find the closest point on the curve to the middle point of the line, this gives the upper estimation for the lumbar spine
        nearest_point = (0, 0)
        dist = 600
        for point in pass_1:
            cur_dist = math.sqrt((point[0] - middle_x)**2 + (point[1] - middle_y)**2)
            if cur_dist < dist:
                dist = cur_dist
                nearest_point = point
        # discard all points above the estimation of the lumbar spine
        actual_contour_pts = []
        for point in pass_1:
            if (point[1] > nearest_point[1]):
                actual_contour_pts.append(point)
################################################################################################


# DRAWING SPINE VS TRIANGLE
###################################################################################################
        top_right_intersection = nearest_point
        # fill the user's back contour with red
        lumbar_spine_contour = actual_contour_pts
        lumbar_spine_contour.append([top_right_intersection[0], bottom_left_intersection[1]])
        lumbar_spine_contour = np.array(lumbar_spine_contour)
        cv2.fillPoly(img, [lumbar_spine_contour], self.bright_red)

        # overlay the 'ideal' triangle on their back with green, only the red peeks through
        triangle_pts = [
            [bottom_left_intersection[0], bottom_left_intersection[1]],
            [top_right_intersection[0], top_right_intersection[1]],
            [top_right_intersection[0], bottom_left_intersection[1]],
        ]
        triangle_pts = np.array(triangle_pts)
        cv2.fillPoly(img=img, pts=[triangle_pts], color=self.bright_green)
###################################################################################################


# COMPARISON WITH 'IDEAL' STRAIGHT BACK
###################################################################################################
        # compare aeras
        measured_area = cv2.contourArea(lumbar_spine_contour)
        perfect_area = cv2.contourArea(triangle_pts)
        coeff = round(measured_area / perfect_area - 1, 2)
        self.eval_list.append(coeff)
        self.successful_eval_count += 1
        # simple running average to (partially) negate outliers, estimation occasionally will bug out
        avg = 0
        if (len(self.eval_list) < self.running_avg_amount):
            avg = round(sum(self.eval_list) / len(self.eval_list), 2)
        else:
            avg = round(sum(self.eval_list[self.successful_eval_count - self.running_avg_amount:self.successful_eval_count]) / self.running_avg_amount, 2)
        # print(f"measured area: {measured_area}, perfect area: {perfect_area}")
        # print(f"coeff: {coeff}, avg: {avg}")
        cur_eval = "straight"
        cur_color = self.light_aqua
        if (avg > self.round_thresh):
            cur_eval = "rounded"
            cur_color = self.light_red
###################################################################################################


# DISPLAYING RESULTS
###################################################################################################
        cv2.putText(
            img=img, text=f"RA Score: {avg}",
            org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=self.light_purple, thickness=2,
            lineType=cv2.LINE_AA
        )
        # put "lower is better"
        cv2.putText(
            img=img, text="(lower is better)",
            org=(20, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=self.light_red, thickness=1,
            lineType=cv2.LINE_AA
        )
        cv2.putText(
            img=img, text=f"Evaluation: {cur_eval}",
            org=(275, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=cur_color, thickness=2,
            lineType=cv2.LINE_AA
        )

    # determine the intersection between a line (defined by two points) and contour (defined by alist of points)
    def determine_intersect(self, line_endpoints, contour_pts):
        slope = (line_endpoints[1][1] - line_endpoints[0][1]) / (line_endpoints[1][0] - line_endpoints[0][0])
        intercept = line_endpoints[0][1] - slope * line_endpoints[0][0]
        intersections = []
        for point in contour_pts:
            if abs(point[1] - (slope * point[0] + intercept)) < 3:
                intersections.append(point)
        if len(intersections) > 0:
            closest_intersect = intersections[0]
            for point in intersections:
                if point[0] > closest_intersect[0] and point[1] > closest_intersect[1]:
                    closest_intersect = point
            return closest_intersect
        return None

    # pre-processing for yolo, from their github
    def yolo_pre(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
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

    def yolo_annotate(self, img, img_orig):
        image, ratio, dwdh = self.yolo_pre(img_orig, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255
        outname = [i.name for i in self.session.get_outputs()]
        inname = [i.name for i in self.session.get_inputs()]
        inp = {inname[0]:im}
        outputs = self.session.run(outname, inp)[0]
        ori_images = [img_orig]
        for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            if (cls_id) == 0:
                self.weight_pt_list.append(((box[0] + box[2]) // 2, (box[1] + box[3]) // 2))
            else:
                self.bar_pt_list.append(((box[0] + box[2]) // 2, (box[1] + box[3]) // 2))
            cv2.rectangle(
                img=img,
                pt1=box[:2],
                pt2=box[2:],
                color=self.cls_colors[self.cls_names[cls_id]],
                thickness=3
            )

    def remove_close_points(self, points, threshold):
        filtered_points = []
        for i, p1 in enumerate(points):
            should_add_point = True
            for j, p2 in enumerate(points):
                if i == j:
                    continue
                if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < threshold:
                    should_add_point = False
                    break
            if should_add_point:
                filtered_points.append(p1)
        return filtered_points

    def smooth_points(self, points, window_size):
        smoothed_points = []
        for i in range(len(points)):
            # Compute the average of the point and its neighbors
            average = points[0]
            num_neighbors = 1
            for j in range(1, window_size + 1):
                if i - j >= 0:
                    average = tuple(map(sum, zip(average, points[i - j])))
                    num_neighbors += 1
                if i + j < len(points):
                    average = tuple(map(sum, zip(average, points[i + j])))
                    num_neighbors += 1
            average = tuple(x / num_neighbors for x in average)
            smoothed_points.append(average)
        return np.array(smoothed_points, np.int32)


# todo:
# contour analysis along back
# bar path tracker (concentric circles, neural network??)
# bar distance tracker (dist formula)
# heel lift (angles, contour analysis)
# arm pull (angles)
# butt tracker, shoulder tracker (keypoints)


providers = ['CUDAExecutionProvider']
session = ort.InferenceSession("./models/yolov7_weight.onnx", providers=providers)

pose_analyzer = PoseAnalyzer(
    in_path='./videos/deadlift.mp4',
    out_path='./videos/deadlift_out.mp4',

    session=session,
    distance_threshold=2,
    bar_window_size=2,

    contour_bbox_min=20,
    thresh_val_init=200,
    thresh_val_refine=210,
    erosion_steps=4,
    contour_cutoff=10,
    estimation_extension=1,
    round_thresh=0.16,
    running_avg_amount=2,

    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.8
)
pose_analyzer.fully_analyze()
