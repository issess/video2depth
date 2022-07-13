# video2dpeth
Depthmap video generator using machine learning

## Requirements
```bash
$ pip install torch torchvision opencv-python timm Pillow numpy ffmpeg-python numpy
```

## Usage
```bash
$ python3 video2depth.py --input [filename]
```

## output
 * output_depth.mp4 = depthmap video
 * output_depth_sound.mp4 = depthmap video + sound
 * output_merged.mp4 = original + depthmap video 
 * output_merged_sound.mp4 = original + depthmap video + sound
