# video2dpeth
Depthmap video generator using machine learning

## Requirements
```bash
$ pip install torch torchvision opencv-python timm Pillow numpy ffmpeg-python numpy
```

## Usage
```bash
$ python3 video2depth.py [filename]
```

## output
 * output_sound.mp4 = depthmap video
 * output_merged.mp4 = original + depthmap video
