# MediaPipe-PoseReader
This is the official implementation of the approaches described in the paper:

An exploration of computer vision and machine learning approaches in the evaluation of weightlifting technique

Jan 5, 2023

![image](https://github.com/jeff-j-chen/MediaPipe-PoseReader/assets/46868596/581dc040-d109-4cac-a9fc-35496ba82cd6)
![image](https://github.com/jeff-j-chen/MediaPipe-PoseReader/assets/46868596/972c9e20-43e4-4f1c-8d18-9942546406bd)
![image](https://github.com/jeff-j-chen/MediaPipe-PoseReader/assets/46868596/8bd8ba08-831c-4b1c-b146-be90813287fb)
![image](https://github.com/jeff-j-chen/MediaPipe-PoseReader/assets/46868596/1f946696-eb51-4b72-a076-0953d6edc9b0)



## Demo
![](https://github.com/4a454646/MediaPipe-PoseReader/blob/main/DEMO/output.gif)<br>
Analyses of varying lifting forms (found in the DEMO folder)

## Usage
1. Change the `in_path` argument when creating the `PoseAnalyzer` class in `media.py` to be the video you want to analyze.
2. Run the command `python3 media.py`.

## Notes
The video's start and end must be cropped to the start and end of a lift, the camera must not be occluded. The current implementation only allows for analysis of a single lift; the program will cut the video after a single deadflift is performed. The video is assumed to be 600x600 and may not work as intended at different resolutions.
