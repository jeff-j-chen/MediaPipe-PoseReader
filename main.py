import cv2
import math
from tqdm import tqdm
import mediapipe as mp
import argparse

import torch
import onnxruntime as ort

import Modules.face as face
import Modules.drawer as drawer
import Modules.back_contour as back_contour
import Modules.bar as bar
import Modules.angles as angles
import Modules.reps as reps
import Modules.colors as colors
import Modules.configuration as configuration

class PoseAnalyzer:
    def __init__(self, video_folder, in_path):
        self.analysis_conf = configuration.AnalysisConfig()
        self.mp_conf = configuration.MediaPipeConfig()
        self.back_conf = configuration.BackContourConfig()
        self.bar_conf = configuration.BarPathConfig()
        self.angle_conf = configuration.AngleDetConfig()
        self.face_conf = configuration.FaceDirectionConfig()

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

        # POSE DETECTION
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mp_conf.static_image_mode,
            model_complexity=self.mp_conf.model_complexity,
            enable_segmentation=self.mp_conf.enable_segmentation,
            min_detection_confidence=self.mp_conf.min_detection_confidence,
            min_tracking_confidence=self.mp_conf.min_tracking_confidence
        )
        self.shoulder = self.elbow = self.wrist = self.hip = self.knee = self.ankle = self.heel = self.toe = self.right_ear = [0.0, 0.0]

        # BACK CONTOUR
        # helper for running average
        self.successful_eval_count: int = 0
        # list of back evaluations
        self.eval_list: list[float] = []

        # BAR PATH
        self.yolo_session = ort.InferenceSession(
            path_or_bytes=self.bar_conf.model_path,
            providers=self.bar_conf.providers
        )
        # lists to store detection points
        self.bar_pt_list: list[list[int]] = []
        self.weight_pt_list: list[list[int]] = []
        # minimum distance between points, any closer and they are removed

        # ANGLES
        # checks, if broken then annotate accordingly in second pass
        self.failed_knee_check: bool = False
        self.failed_elbow_check: bool = False
        self.failed_heel_check: bool = False
        self.heel_angle_list: list[float] = []
        self.toe_pos_list: list[list[float]] = []
        self.heel_pos_list: list[list[float]] = []

        # FACE
        self.model = face.FaceDetector()
        checkpoint = torch.load(self.face_conf.model_path)
        state_dict = self.model.state_dict()
        for k1, k2 in zip(state_dict.keys(), checkpoint.keys()):
            state_dict[k1] = checkpoint[k2]
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.face_angles: list[int] = []

    def fully_analyze(self):
        print("initial analysis...")
        img_list = []
        for i in tqdm(range(self.video_length)):
            # IMPORT
            # read frame and convert to RGB
            ret, img = self.cap.read()
            img_orig = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            read_success: bool = self.read_keypoints(img)
            if (not read_success):
                continue

            drawer.draw_lines(self, img)
            drawer.draw_points(self, img)

            if (self.analysis_conf.back_contour):
                back_contour.analyze(self, img, img_orig)

            if (self.analysis_conf.angles):
                angles.analyze_initial(self, img)

            if (self.analysis_conf.bar_path):
                bar.analyze_initial(self, img, img_orig)

            if (self.analysis_conf.face):
                face.analyze(self, img, img_orig)

            drawer.text(img=img, text=f"Frame: {i}", org=(40, self.h - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors.white, thickness=1, lineType=cv2.LINE_AA)
            img_list.append(img)

        print("refining...")
        self.second_pass(img_list)

    # yolo, remove overlapping points, draw paths, calculate straightness, annotate second round stuff, write to screen
    def second_pass(self, img_list):

        if (self.analysis_conf.bar_path):
            np_s_f_bar_pts, median_x, min_y, max_y, stddev, text, text_color = bar.analyze_secondary(self)

        if (self.analysis_conf.angles):
            self.failed_heel_check, median_foot_point = angles.analyze_secondary(self)

        start, end = reps.det_start_end(self, img_list)

        if (self.analysis_conf.face):
            avg = sum(self.face_angles[start:end+1]) / (end - start + 1)
            sixty_count = len([x for x in self.face_angles if x == 60])
            # if average angle is greater than 20 or if there are more than 3x 60 degree angles, fail later

        for i in tqdm(range(start, end+1)):
            img = img_list[i]

            if (self.analysis_conf.angles):
                angles.knee_annotation(self, img)
                angles.elbow_annotation(self, img)
                angles.heel_annotation(self, img, i, median_foot_point)

            if (self.analysis_conf.bar_path and len(self.bar_pt_list) > 0):
                bar.draw_bar_path(self, img, i, np_s_f_bar_pts, median_x, min_y, max_y, stddev, text, text_color)

            if (self.analysis_conf.face):
                face.write_res(self, img, avg, sixty_count)

            # convert to bgr for writing to video
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.video_output.write(img)

        self.cap.release()
        self.video_output.release()

    # define keypoints from mediapose with variable names
    def read_keypoints(self, img):
        keypoints = self.pose.process(img)
        lm = keypoints.pose_landmarks
        if (lm is None):
            return False
        lm_pose = self.mp_pose.PoseLandmark
        self.shoulder = [lm.landmark[lm_pose.RIGHT_SHOULDER].x   * self.w, lm.landmark[lm_pose.RIGHT_SHOULDER].y   * self.h]
        self.elbow    = [lm.landmark[lm_pose.RIGHT_ELBOW].x      * self.w, lm.landmark[lm_pose.RIGHT_ELBOW].y      * self.h]
        self.wrist    = [lm.landmark[lm_pose.RIGHT_WRIST].x      * self.w, lm.landmark[lm_pose.RIGHT_WRIST].y      * self.h]
        self.hip      = [lm.landmark[lm_pose.RIGHT_HIP].x        * self.w, lm.landmark[lm_pose.RIGHT_HIP].y        * self.h]
        self.knee     = [lm.landmark[lm_pose.RIGHT_KNEE].x       * self.w, lm.landmark[lm_pose.RIGHT_KNEE].y       * self.h]
        self.ankle    = [lm.landmark[lm_pose.RIGHT_ANKLE].x      * self.w, lm.landmark[lm_pose.RIGHT_ANKLE].y      * self.h]
        self.heel     = [lm.landmark[lm_pose.RIGHT_HEEL].x       * self.w, lm.landmark[lm_pose.RIGHT_HEEL].y       * self.h]
        self.toe      = [lm.landmark[lm_pose.RIGHT_FOOT_INDEX].x * self.w, lm.landmark[lm_pose.RIGHT_FOOT_INDEX].y * self.h]
        self.right_ear = [lm.landmark[lm_pose.RIGHT_EAR].x       * self.w, lm.landmark[lm_pose.RIGHT_EAR].y        * self.h]
        return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-V', '--video', dest='video_file', type=str, help='Path to the video file')
    args = parser.parse_args()
    video_file_path = args.video_file

    pose_analyzer = PoseAnalyzer(
        video_folder='./videos/',
        in_path=video_file_path,
    )

    pose_analyzer.fully_analyze()
