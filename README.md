# MediaPipe-PoseReader
This is the official implementation of the approaches described in the paper:
> Jeff Chen. [An exploration of computer vision and machine learning approaches in the evaluation of weightlifting technique](https://jeffjchen.org), 2022

## Demo
gifs go here

## Usage
1. Change the `in_path` argument when creating the `PoseAnalyzer` class in `media.py` to be the video you want to analyze.
2. Run the command `python3 media.py`.

## Notes
The video's start and end must be cropped to the start and end of a lift, the camera must not be occluded. The current implementation only allows for analysis of a single lift; the program will cut the video after a single deadflift is performed. The video is assumed to be 600x600 and may not work at different resolutions.
