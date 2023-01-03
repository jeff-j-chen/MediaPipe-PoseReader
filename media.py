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
    def __init__(self, video_folder, in_path, session, distance_threshold, bar_window_size, bar_stddev_threshold, annotation_radius, knee_angle_threshold, elbow_angle_threshold, heel_angle_threshold, toe_radius, contour_bbox_min, thresh_val_init, erosion_steps, contour_cutoff, round_thresh, running_avg_amount, static_image_mode, model_complexity, enable_segmentation, min_detection_confidence, min_tracking_confidence):
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
        self.cap = cv2.VideoCapture(video_folder + in_path)
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_output = cv2.VideoWriter(
            filename=video_folder + in_path[:-4] + "_out.mp4",
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
            # whether to reavluate on every single image
            static_image_mode=self.static_image_mode,
            # how complex, bigger is better but slower
            model_complexity=self.model_complexity,
            # segmentation (not used)
            enable_segmentation=self.enable_segmentation,
            # for body to be considered detected
            min_detection_confidence=self.min_detection_confidence,
            # for tracking to occur, rather than re-evaluation, makes it smoother
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.shoulder = self.elbow = self.wrist = self.hip = self.knee = self.ankle = self.heel = self.toe = self.right_ear = None

        # parameters for back contour detection
        # the minimum size of the bounding box of the contour for evaluation to occur
        self.contour_bbox_min = contour_bbox_min
        # thresholding value
        self.thresh_val_init = thresh_val_init
        # morphological operations to remove holes, requires step #
        self.erosion_steps = erosion_steps
        # how far to cut off from the top and left
        self.contour_cutoff = contour_cutoff
        # cutoff for back to be considered rounded
        self.round_thresh = round_thresh
        # window size for running average
        self.running_avg_amount = running_avg_amount
        # helper for running average
        self.successful_eval_count = 0
        # list of back evaluations
        self.eval_list = []

        # yolo
        # required for evaluations
        self.session = session
        # 0: weight, 1: bar
        self.cls_names = ["weight", "bar"]
        # colors for the classes
        self.cls_colors = {
            self.cls_names[0]: self.light_red,
            self.cls_names[1]: self.red
        }
        # lists to store detection points
        self.bar_pt_list = []
        self.weight_pt_list = []
        # minimum distance between points, any closer and they are removed
        self.distance_threshold = distance_threshold
        # window size for the smoothing for the bar path
        self.bar_window_size = bar_window_size
        # cutoff for 'good' vs 'bad' bar path
        self.bar_stddev_threshold = bar_stddev_threshold

        # knee and elbow
        # how large to draw the arc for the angle of 3 points
        self.annotation_radius = annotation_radius
        # checks, if broken then annotate accordingly in second pass
        self.knee_angle_threshold = knee_angle_threshold
        self.failed_knee_check = False
        self.elbow_angle_threshold = elbow_angle_threshold
        self.failed_elbow_check = False
        self.heel_angle_threshold = heel_angle_threshold
        # required due to how heel is evaluated (occlusion messes things up!)
        self.heel_angle_list = []
        self.toe_pos_list = []
        self.heel_pos_list = []
        self.toe_radius = toe_radius

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
            # calculation of angles
            knee_angle = self.annotate_angle(
                img, color=self.blue, l_or_r="l",
                pt1=[int(self.hip[0]), int(self.hip[1])],
                pt2=[int(self.knee[0]), int(self.knee[1])],
                pt3=[int(self.ankle[0]), int(self.ankle[1])],
                radius=self.annotation_radius,
            )
            elbow_angle = self.annotate_angle(
                img, color=self.green, l_or_r="r",
                pt1=[int(self.shoulder[0]), int(self.shoulder[1])],
                pt2=[int(self.elbow[0]), int(self.elbow[1])],
                pt3=[int(self.wrist[0]), int(self.wrist[1])],
                radius=self.annotation_radius,
            )
            # elbow check here, knee check is in yolo_annotate due to needed the bar position
            if (elbow_angle < self.elbow_angle_threshold):
                self.failed_elbow_check = True
            # toe angle is drawn in 2nd pass because we first need to evaluate if the foot is occluded by the weight or not
            # calculate and append to lists here
            heel_angle = self.annotate_angle(
                img, color=self.light_purple, l_or_r="l",
                pt1=[int(self.heel[0]), int(self.heel[1])],
                pt2=[int(self.toe[0]), int(self.toe[1])],
                pt3=[int(self.heel[0]), int(self.toe[1])],
                radius=self.annotation_radius,
                draw=False,
            )
            self.heel_angle_list.append(heel_angle)
            self.heel_pos_list.append(self.heel)
            self.toe_pos_list.append(self.toe)

            # evalate position of bar and weight using yolo
            self.yolo_annotate(img, img_orig, knee_angle)

            # EXPORT
            # write the frame # bottom left of the video
            self.text( img=img, text=f"Frame: {i}", org=(40, self.h - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=self.white, thickness=1, lineType=cv2.LINE_AA)
            # save the frame to a list for further processing
            img_list.append(img)

        print("refining...")
        self.second_pass(img_list)

    # define keypoints from mediapose with variable names
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
        self.right_ear = [lm.landmark[lmPose.RIGHT_EAR].x       * self.w, lm.landmark[lmPose.RIGHT_EAR].y        * self.h]

    # draw lines between keypoints, e.g. shoulder-elbow...
    def draw_lines(self, img):
        cv2.line( 
            img=img, thickness=3, lineType=cv2.LINE_AA,
            pt1=(int(self.shoulder[0]), int(self.shoulder[1])),
            pt2=(int(self.right_ear[0]), int(self.right_ear[1])),
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

    # draw points on keypoints, e.g. nose, shoulder...
    def draw_points(self, img):
        cv2.circle( # EAR
            img=img, thickness=-1, lineType=cv2.LINE_AA,
            center=(int(self.right_ear[0]), int(self.right_ear[1])),
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

    # evaluation and drawing of the back contour
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
            self.write_contour_text(img, "None")
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
        cv2.fillPoly(img, [lumbar_spine_contour], self.light_red)

        # overlay the 'ideal' triangle on their back with green, only the red peeks through
        triangle_pts = [
            [bottom_left_intersection[0], bottom_left_intersection[1]],
            [top_right_intersection[0], top_right_intersection[1]],
            [top_right_intersection[0], bottom_left_intersection[1]],
        ]
        triangle_pts = np.array(triangle_pts)
        cv2.fillPoly(img=img, pts=[triangle_pts], color=self.light_aqua)
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
###################################################################################################

        self.write_contour_text(img, avg)

    # helper for back contour, writes evaluation to the screen
    def write_contour_text(self, img, avg):
        cur_eval = "straight"
        cur_color = self.light_aqua
        if (avg == "None"):
            cur_eval = "none"
            cur_color = self.light_blue
        else:
            if (avg > self.round_thresh):
                cur_eval = "rounded"
                cur_color = self.light_red
        self.text(
            img=img, text=f"back {cur_eval} ({avg:.2f})",
            org=(15, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=cur_color, thickness=2,
            lineType=cv2.LINE_AA
        )

    # helper for back contour
    # determine the intersection between a line (defined by two points) and contour (defined by a list of points)
    def determine_intersect(self, line_endpoints, contour_pts):
        slope = (line_endpoints[1][1] - line_endpoints[0][1]) / (line_endpoints[1][0] - line_endpoints[0][0])
        intercept = line_endpoints[0][1] - slope * line_endpoints[0][0]
        # define y and m
        intersections = []
        # look through every contour point, if it lies close enough to the line, add it to the list of intersections
        for point in contour_pts:
            if abs(point[1] - (slope * point[0] + intercept)) < 3:
                intersections.append(point)
        # if there is an intersection, get the ones closer to the bottom right
        # this assumes that the user is facing right, this makes sure the intersection lies on the user
        # as opposed to being some random point on the contour caused by thresholding
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

    # drawing the detected bounding boxes every frame, and the line to the knee/him
    def yolo_annotate(self, img, img_orig, knee_angle):
        # more yolo preprocessing
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
        # actual annotation
        for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            center_pt = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]
            if (cls_id) == 0: self.weight_pt_list.append(center_pt)
            else: self.bar_pt_list.append(center_pt)
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
                    color=self.light_purple,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
            else:
                cv2.line(
                    img=img,
                    pt1=center_pt,
                    pt2=(int(self.knee[0]), int(self.knee[1])),
                    color=self.light_purple,
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
                if (knee_angle > self.knee_angle_threshold):
                    self.failed_knee_check = True

    # yolo, remove overlapping points, draw paths, calculate straightness, annotate second round stuff, write to screen
    def second_pass(self, img_list):
        # process the points so that the bar path is good
        filtered_bar_pts = self.remove_close_points(self.bar_pt_list, self.distance_threshold)
        if (len(filtered_bar_pts) == 0):
            return
        filtered2_bar_pts = self.remove_outliers(filtered_bar_pts, 50)
        s_f_bar_pts = self.smooth_horizontally(filtered2_bar_pts, self.bar_window_size)
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
        text_color = self.light_red if stddev > self.bar_stddev_threshold else self.light_aqua

        # foot is inconsistently detected when occluded by the weight, so only calculate foot angle when not occluded
        # occlusion is determined by checking if the foot is close to its "regular" position, if occluded the position signficiantly deviates from the median
        failed_heel_check = False
        median_foot_point = self.toe_pos_list[len(self.toe_pos_list) // 2]
        for i in range(len(self.heel_angle_list)):
            angle = self.heel_angle_list[i]
            toe = self.toe_pos_list[i]
            if ((math.sqrt((median_foot_point[0] - toe[0]) ** 2 + (median_foot_point[1] - toe[1]) ** 2) < self.toe_radius)):
                if (angle > self.heel_angle_threshold):
                    failed_heel_check = True
                    break

        for i in tqdm(range(len(img_list))):
            img = img_list[i]

            # bar text
            self.text(
                img=img,
                text=f"{text} ({stddev:.2f})",
                org=(15, 60),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=text_color,
                thickness=2,
                lineType=cv2.LINE_AA
            )
            # knee text
            if (self.failed_knee_check):
                self.text(
                    img=img,
                    text="knee was straightened too early",
                    org=(15, 85),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.85,
                    color=self.light_red,
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
            else:
                self.text(
                    img=img,
                    text="knee was acceptable",
                    org=(15, 85),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.85,
                    color=self.light_aqua,
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
            # elbow text
            if (self.failed_elbow_check):
                self.text(
                    img=img,
                    text="arm was pulling",
                    org=(15, 105),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.85,
                    color=self.light_red,
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
            else:
                self.text(
                    img=img,
                    text="arm was acceptable",
                    org=(15, 105),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.85,
                    color=self.light_aqua,
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
            # circle around the median foot point, makes it visually clear when and why the angle is being calculated
            cv2.circle(
                img=img,
                center=(int(median_foot_point[0]), int(median_foot_point[1])),
                radius=self.toe_radius,
                color=self.light_red,
                thickness=2,
                lineType=cv2.LINE_AA
            )
            # only draw the angle if foot is not occluded
            toe = self.toe_pos_list[i]
            heel = self.heel_pos_list[i]
            if (math.sqrt((median_foot_point[0] - toe[0]) ** 2 + (median_foot_point[1] - toe[1]) ** 2) < self.toe_radius):
                self.annotate_angle(
                    img, color=self.bright_red, l_or_r="l",
                    pt1=[int(heel[0]), int(heel[1])],
                    pt2=[int(toe[0]), int(toe[1])],
                    pt3=[int(heel[0]), int(toe[1])],
                    radius=self.annotation_radius,
                    extra_offset=True
                )
            # heel text
            if (failed_heel_check):
                self.text(
                    img=img,
                    text="heel was lifted",
                    org=(15, 125),
                    color=self.light_red,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75,
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
            else:
                self.text(
                    img=img,
                    text="heel was acceptable",
                    org=(15, 125),
                    color=self.light_aqua,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75,
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
            # bar path
            cv2.polylines(
                img=img,
                pts=[np_s_f_bar_pts],
                isClosed=False,
                color=self.red,
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
                color=self.bright_red,
                thickness=-1,
                lineType=cv2.LINE_AA
            )

            # convert to bgr for writing to video
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.video_output.write(img)

        self.cap.release()
        self.video_output.release()

    # helper for path drawing, remove all points that are too close to each other in the y axis
    # eliminates bar staying still and stationary weights
    def remove_close_points(self, points, threshold):
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
    def remove_outliers(self, points, threshold):
        x_coords = [p[0] for p in points]
        median_x = sorted(x_coords)[len(x_coords) // 2]
        # remove points too far away from the x median
        filtered_points = [p for p in points if abs(p[0] - median_x) < threshold]
        return filtered_points

    # helper for path drawing, interpolate the bar path, but only horizontally
    def smooth_horizontally(self, points, window_size):
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

    # label an angle with the arc and the degrees, pt2 is the center point
    def annotate_angle(self, img, pt1, pt2, pt3, radius, color, l_or_r, draw=True, extra_offset=False):
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
        if (angle < 0):
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
            self.text(
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
    def text(self, img, text, org, color, fontFace, fontScale, thickness, lineType, outline_thickness=2, outline_color=(0, 0, 0)):
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


# todo:
# detect when lift starts and finishes
# face angle tracker, after ^


providers = ['CUDAExecutionProvider']
session = ort.InferenceSession("./models/yolov7_weight.onnx", providers=providers)

pose_analyzer = PoseAnalyzer(
    video_folder='./videos/',
    in_path='heellift.mp4',

    session=session,
    distance_threshold=2,
    bar_window_size=2,
    bar_stddev_threshold=5,

    contour_bbox_min=20,
    thresh_val_init=200,
    erosion_steps=4,
    contour_cutoff=10,
    round_thresh=0.16,
    running_avg_amount=2,

    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.8,

    annotation_radius=20,
    knee_angle_threshold=160,
    elbow_angle_threshold=160,
    heel_angle_threshold=10,
    toe_radius=11,
)
pose_analyzer.fully_analyze()
