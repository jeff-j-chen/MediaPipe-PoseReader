import cv2
import math
from tqdm import tqdm
import mediapipe as mp

def dist_formula(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def findAngle(x1, y1, x2, y2):
    # find angle using law of cosines
    theta = math.acos(
        (y2 - y1) * (-y1) /
        (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1)
    )
    return int(180/math.pi)*theta


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.8
)

class PoseAnalyzer:
    def __init__(self, in_path, out_path, static_image_mode, model_complexity, enable_segmentation, min_detection_confidence, min_tracking_confidence):
        # video setup
        self.in_path = in_path
        self.out_path = out_path
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

        # colors, from gruvbox light theme
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

    def fully_analyze(self):
        for i in tqdm(range(self.video_length)):
            ret, img = self.cap.read()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            keypoints = pose.process(img)

            lm = keypoints.pose_landmarks
            lmPose = mp_pose.PoseLandmark

            self.read_keypoints(lm, lmPose)

            self.draw_lines(img)

            self.draw_points(img)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            self.video_output.write(img)

        self.cap.release()
        self.video_output.release()

    def read_keypoints(self, lm, lmPose):
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
        # todo: redo colors, make them more cohesive (+ some of the color codes are incorrect)


pose_analyzer = PoseAnalyzer(
    in_path='deadlift.mp4',
    out_path='deadlift_out.mp4',

    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.8
)
pose_analyzer.fully_analyze()