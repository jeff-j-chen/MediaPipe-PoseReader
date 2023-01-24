# MediaPipe-PoseReader
This is the official implementation of the approaches described in the paper:
> Jeff Chen. [An exploration of computer vision and machine learning approaches in the evaluation of weightlifting technique](https://jeffjchen.org), 2022

## Demo
![](https://github.com/4a454646/MediaPipe-PoseReader/blob/main/DEMO/output.gif)<br>
Analyses of varying lifting forms (found in the DEMO folder)

## Usage
1. Change the `in_path` argument when creating the `PoseAnalyzer` class in `media.py` to be the video you want to analyze.
2. Run the command `python3 media.py`.

## Notes
The video's start and end must be cropped to the start and end of a lift, the camera must not be occluded. The current implementation only allows for analysis of a single lift; the program will cut the video after a single deadflift is performed. The video is assumed to be 600x600 and may not work as intended at different resolutions.

## Future Plans
I'm currently working on developing this into an app! Analyses will be made more robust, refined and expanded to cover more lifts. It will (hopefully) be as simple as taking a video with your phone and seeing the analysis a few minutes later.
