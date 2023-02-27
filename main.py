import cv2
import numpy as np
import math
from tqdm import trange
import mediapipe as mp
import argparse

import torch
import onnxruntime as ort

import modules.face as face
import modules.drawer as drawer
import modules.back_contour as back_contour
import modules.bar as bar
import modules.angles as angles
import modules.reps as reps
import modules.colors as colors
import modules.configuration as configuration

class PoseAnalyzer:
    def __init__(self, video_folder, in_path):
        self.analysis_conf = configuration.AnalysisConfig()
        self.mp_conf = configuration.MediaPipeConfig()
        self.back_conf = configuration.BackContourConfig()
        self.bar_conf = configuration.BarPathConfig(
            providers=["CUDAExecutionProvider"],
            cls_names=["weight", "bar"],
            cls_colors={
                "weight": colors.light_red,
                "bar": colors.red
            }
        )
        self.angle_conf = configuration.AngleDetConfig()
        self.face_conf = configuration.FaceDirectionConfig(
            class_angle_dict={
                0: 0,
                1: 30,
                2: 60,
                3: 90,
            }
        )

        # VIDEO AND IMAGES
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
        self.img: np.ndarray = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.img_orig: np.ndarray = np.zeros((self.h, self.w), dtype=np.uint8)

        # POSE DETECTION
        self.mp_pose = mp.solutions.pose # pyright: ignore[reportGeneralTypeIssues]
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mp_conf.static_image_mode,
            model_complexity=self.mp_conf.model_complexity,
            enable_segmentation=self.mp_conf.enable_segmentation,
            min_detection_confidence=self.mp_conf.min_detection_confidence,
            min_tracking_confidence=self.mp_conf.min_tracking_confidence
        )
        self.shoulder = self.elbow = self.wrist = self.hip = self.knee = self.ankle = self.heel = self.toe = self.right_ear = [0.0, 0.0]
        self.seg_mask: np.ndarray = np.zeros((self.h, self.w), dtype=np.uint8)

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

    def fully_analyze(self) -> None:
        '''
        First round analysis, determining the back contour, agnles, bar path, and face direction.
        '''
        print("initial analysis...")
        img_list = []
        for i in trange(self.video_length):
            # IMPORT
            # read frame and convert to RGB
            ret, self.img = self.cap.read()
            self.img_orig = self.img.copy()
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

            read_success: bool = self.read_mediapipe()
            if (not read_success):
                continue

            drawer.draw_lines(self)
            drawer.draw_points(self)

            if (self.analysis_conf.back_contour):
                back_contour.analyze(self)

            if (self.analysis_conf.angles):
                angles.analyze_initial(self)

            if (self.analysis_conf.bar_path):
                bar.analyze_initial(self)

            if (self.analysis_conf.face):
                face.analyze(self)

            drawer.text(self, text=f"Frame: {i}", org=(40, self.h - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors.white, thickness=1, lineType=cv2.LINE_AA)
            img_list.append(self.img)

        print("refining...")
        self.second_pass(img_list)

    # yolo, remove overlapping points, draw paths, calculate straightness, annotate second round stuff, write to screen
    def second_pass(self, img_list: list[np.ndarray]) -> None:
        '''
        Second round analysis, refining the bar path, determining heel angle, and averaging out looking direction.
        '''
        if (self.analysis_conf.bar_path):
            np_s_f_bar_pts, median_x, min_y, max_y, stddev, text, text_color = bar.analyze_secondary(self) # pyright: ignore[reportGeneralTypeIssues]

        if (self.analysis_conf.angles):
            self.failed_heel_check, median_foot_point = angles.analyze_secondary(self)

        start, end = reps.det_start_end(self, img_list)

        if (self.analysis_conf.face):
            avg = sum(self.face_angles[start:end+1]) / (end - start + 1)
            sixty_count = len([x for x in self.face_angles if x == 60])
            # if average angle is greater than 20 or if there are more than 3x 60 degree angles, fail later

        for i in trange(start, end+1):
            img = img_list[i]

            if (self.analysis_conf.angles):
                angles.knee_annotation(self)
                angles.elbow_annotation(self)
                angles.heel_annotation(self, i, median_foot_point) # pyright: ignore[reportUnboundVariable]

            if (self.analysis_conf.bar_path and len(self.bar_pt_list) > 0):
                bar.draw_bar_path(self, i, np_s_f_bar_pts, median_x, min_y, max_y, float(stddev), text, text_color) # pyright: ignore[reportUnboundVariable]

            if (self.analysis_conf.face):
                face.write_res(self, avg, sixty_count) # pyright: ignore[reportUnboundVariable]

            # convert to bgr for writing to video
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.video_output.write(img)

        self.cap.release()
        self.video_output.release()

    # define keypoints from mediapose with variable names
    def read_mediapipe(self) -> bool:
        '''
        Reads the keypoints and segmentation mask from mediapipe and stores them in the class.
        Returns `False` if no keypoints are found.
        '''
        results = self.pose.process(self.img)
        lm = results.pose_landmarks
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

        # write the segmentation mask over the original image
        self.seg_mask = results.segmentation_mask > self.back_conf.seg_thresh
        stacked = np.stack((self.seg_mask,) * 3, axis=-1)
        img_dark = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        img_dark[:, :, 1] = img_dark[:, :, 1] * self.back_conf.desat # pyright: ignore[reportGeneralTypeIssues]
        img_dark[:, :, 2] = img_dark[:, :, 2] * self.back_conf.darken # pyright: ignore[reportGeneralTypeIssues]
        img_dark = cv2.cvtColor(img_dark, cv2.COLOR_HSV2BGR)
        self.img = np.where(stacked, self.img, img_dark)
        return True


if (__name__ == '__main__'):
    # use -V or --video to specify name of video file to analyze
    parser = argparse.ArgumentParser()
    parser.add_argument('-V', '--video', dest='video_file', type=str, help='Path to the video file')
    args = parser.parse_args()
    video_file_path = args.video_file

    pose_analyzer = PoseAnalyzer(
        video_folder='./videos/',
        in_path=video_file_path,
        # all other configurations are set to default within __init__
    )

    pose_analyzer.fully_analyze()
