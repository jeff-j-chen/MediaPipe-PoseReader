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

file_name = 'deadlift.mp4'
cap = cv2.VideoCapture(file_name)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 3D
print('\nGenerating 3D pose...')
for i in tqdm(range(video_length)):
    ret, img = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = img.shape[:2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    keypoints = pose.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    lm = keypoints.pose_landmarks
    lmPose = mp_pose.PoseLandmark

    # get the coordinates of all right side blazepose keypoints
    shoulder = [lm.landmark[lmPose.RIGHT_SHOULDER].x*w, lm.landmark[lmPose.RIGHT_SHOULDER].y*h]
    elbow = [lm.landmark[lmPose.RIGHT_ELBOW].x*w, lm.landmark[lmPose.RIGHT_ELBOW].y*h]
    wrist = [lm.landmark[lmPose.RIGHT_WRIST].x*w, lm.landmark[lmPose.RIGHT_WRIST].y*h]
    hip = [lm.landmark[lmPose.RIGHT_HIP].x*w, lm.landmark[lmPose.RIGHT_HIP].y*h]
    knee = [lm.landmark[lmPose.RIGHT_KNEE].x*w, lm.landmark[lmPose.RIGHT_KNEE].y*h]
    ankle = [lm.landmark[lmPose.RIGHT_ANKLE].x*w, lm.landmark[lmPose.RIGHT_ANKLE].y*h]
    heel = [lm.landmark[lmPose.RIGHT_HEEL].x*w, lm.landmark[lmPose.RIGHT_HEEL].y*h]
    toe = [lm.landmark[lmPose.RIGHT_FOOT_INDEX].x*w, lm.landmark[lmPose.RIGHT_FOOT_INDEX].y*h]
    nose = [lm.landmark[lmPose.NOSE].x*w, lm.landmark[lmPose.NOSE].y*h]
    eye = [lm.landmark[lmPose.RIGHT_EYE].x*w, lm.landmark[lmPose.RIGHT_EYE].y*h]

    # draw lines between keypoints
    cv2.line(img, (int(nose[0]), int(nose[1])), (int(eye[0]), int(eye[1])), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(img, (int(shoulder[0]), int(shoulder[1])), (int(eye[0]), int(eye[1])), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(img, (int(shoulder[0]), int(shoulder[1])), (int(hip[0]), int(hip[1])), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(img, (int(shoulder[0]), int(shoulder[1])), (int(elbow[0]), int(elbow[1])), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(img, (int(elbow[0]), int(elbow[1])), (int(wrist[0]), int(wrist[1])), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(img, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(img, (int(knee[0]), int(knee[1])), (int(ankle[0]), int(ankle[1])), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(img, (int(ankle[0]), int(ankle[1])), (int(heel[0]), int(heel[1])), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(img, (int(ankle[0]), int(ankle[1])), (int(toe[0]), int(toe[1])), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(img, (int(heel[0]), int(heel[1])), (int(toe[0]), int(toe[1])), (0, 255, 0), 3, cv2.LINE_AA)

    cv2.circle(img, (int(shoulder[0]), int(shoulder[1])), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (int(elbow[0]), int(elbow[1])), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (int(wrist[0]), int(wrist[1])), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (int(hip[0]), int(hip[1])), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (int(knee[0]), int(knee[1])), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (int(ankle[0]), int(ankle[1])), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (int(heel[0]), int(heel[1])), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (int(toe[0]), int(toe[1])), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (int(nose[0]), int(nose[1])), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (int(eye[0]), int(eye[1])), 5, (0, 0, 255), -1, cv2.LINE_AA)

    video_output.write(img)


print('Finished.')
cap.release()
video_output.release()


# todo: colrs, analysis, make into class (cleanup in generaL)
