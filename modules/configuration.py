from dataclasses import dataclass, field
from torchvision import transforms
import modules.colors as colors

@dataclass
class AnalysisConfig:
    '''
    Base configuration options for analysis
    '''
    # find back straightness
    back_contour: bool = False
    # find various angles
    angles: bool = False
    # find bar path
    bar_path: bool = False
    # find face direction
    face: bool = False
    # cut off lift at start and end
    start_end: bool = False

@dataclass
class MediaPipeConfig:
    '''
    Configuration for MediaPipe (Google's pose detection library)
    '''
    # whether to reavluate on every single image
    static_image_mode: bool = False
    # how complex, bigger is better but slower
    model_complexity: int = 2
    # segmentation (not used)
    enable_segmentation: bool = True
    # for body to be considered detected
    min_detection_confidence: float = 0.6
    # for tracking to occur, rather than re-evaluation, makes it smoother
    min_tracking_confidence: float = 0.8

@dataclass
class BackContourConfig:
    '''
    Configuration for back contour detection and analysis
    '''
    # the minimum size of the bounding box of the contour for evaluation to occur
    contour_bbox_min: int = 20
    # thresholding value
    thresh_val_init: int = 200
    # morphological operations to remove holes, requires step #
    erosion_steps: int = 2
    # how far to cut off from the top and left
    contour_cutoff: int = 10
    # cutoff for back to be considered rounded
    round_thresh: float = 0.16
    # window size for running average
    running_avg_amount: int = 2
    # minimum threshold for segmenting body vs. background
    seg_thresh: float = 0.45
    # how much to change desaturation and value for the darkened bg to be
    desat: float = 0.8
    darken: float = 0.8
    # how much to adjust the triangle in further to increase accuracy
    realign: int = 5

@dataclass
class BarPathConfig:
    '''
    Configuration for bar detection and path analysis
    '''
    # yolov7 weights
    model_path: str = "./models/yolov7_weight.onnx"
    # providers (to create session)
    providers: list[str] = field(default_factory=list)
    # 0: weight, 1: bar
    cls_names: list[str] = field(default_factory=list)
    # colors for each class
    cls_colors: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    # minimum distance between points, any closer and they are removed
    distance_threshold: int = 1
    # window size for the smoothing for the bar path
    bar_window_size: int = 2
    # cutoff for 'good' vs 'bad' bar path
    bar_stddev_threshold: int = 5

@dataclass
class AngleDetConfig:
    '''
    Configuration for angles of the knee, elbow, and heel
    '''
    # how large to draw the arc for the angle of 3 points
    annotation_radius: int = 20
    # if bar is closer to hip than knee and angle is less than this, it is flagged
    knee_angle_threshold: int = 160
    # if elbow ever goes above this angle, it is flagged
    elbow_angle_threshold: int = 160
    # if heel ever goes above this angle, it is flagged
    heel_angle_threshold: int = 10
    # in what radius to consider the toe angle
    toe_radius: int = 11

@dataclass
class FaceDirectionConfig:
    '''
    Configuration for face direction detection
    '''
    # pytorch weights
    model_path: str = "./models/face_direction.pt"
    # how large of a bound to draw around the face
    face_bound: int = 45
    # go from index to angle
    class_angle_dict: dict[int, int] = field(default_factory=dict)
    # necessary pre pytorch transforms
    pt_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4450, ), (0.3000, )),
        transforms.Resize([64, 64])
    ])
