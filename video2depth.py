import argparse
import glob
import os
import time
from pathlib import Path

import cv2
import ffmpeg
import numpy as np
import torch
import shutil
import shutil


def runImage(input):
    output_dir = "out/" + Path(input).stem
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start = time.time()
    
    print("==================================================================")
    print(" copy to image... ")
    print("==================================================================")
    copy2image(input, output_dir)
    
    print()
    print("==================================================================")
    print(" image to depth... ")
    print("==================================================================")
    image2depth(output_dir, "DPT_Large")  # DPT_Large or DPT_Hybrid or MiDaS_small

    print()
    print("==================================================================")
    print(" merge to image and depth... ")
    print("==================================================================")
    merge_image_depth(output_dir)
    
    end = time.time()
    print()
    print("elapsed time", end - start)

def copy2image(input, output_dir):
    image_dir = output_dir + "/image/"

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    if len(glob.glob(glob.escape(image_dir) + '*')) == 1:
        print("copy2image already done.")
        return
    
    shutil.copy(input, image_dir)

def runVideo(input):
    output_dir = "out/" + Path(input).stem
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    probe = ffmpeg.probe(input)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    frame_rate = '%0.2f' % eval(video_stream['r_frame_rate'])
    nb_frames = int(video_stream['nb_frames'])

    start = time.time()

    print("file=" + input)
    print("frame_rate=", frame_rate, " nb_frames=", nb_frames, " width=", width, " height=", height)
    print()

    print("==================================================================")
    print(" video to image... ")
    print("==================================================================")
    video2image(input, nb_frames, output_dir)

    print()
    print("==================================================================")
    print(" image to depth... ")
    print("==================================================================")
    image2depth(output_dir, "DPT_Large")  # DPT_Large or DPT_Hybrid or MiDaS_small

    print()
    print("==================================================================")
    print(" merge to image and depth... ")
    print("==================================================================")
    merge_image_depth(output_dir)

    print()
    print("==================================================================")
    print(" depth to video... ")
    print("==================================================================")
    depth2video(input, output_dir, "depth", frame_rate)
    print()
    depth2video(input, output_dir, "merged", frame_rate)
    print()

    end = time.time()
    print()
    print("elapsed time", end - start)


def video2image(input, nb_frames, output_dir):
    image_dir = output_dir + "/image/"
    depth_dir = output_dir + "/depth/"

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    if nb_frames <= len(glob.glob(glob.escape(image_dir) + '*')):
        print("video2image already done.")
        return

    if os.path.exists(depth_dir):
        shutil.rmtree(depth_dir)

    stream = ffmpeg.input(input).output(image_dir + '%06d.png',
                                        **{"start_number": 0, "qmin": 1, "qmax": 1, "qscale:v": 1})
    ffmpeg.run(stream)

    print("done.")


def image2depth(output_dir, model_type):
    image_dir = output_dir + "/image/"
    depth_dir = output_dir + "/depth/"
    merged_dir = output_dir + "/merged/"

    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir)

    if len(glob.glob(glob.escape(image_dir) + '/*')) == len(glob.glob(glob.escape(depth_dir) + '/*')):
        print("image2depth already done.")
        return

    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)

    midas = torch.hub.load('intel-isl/MiDaS', model_type)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    for file in sorted(glob.glob(glob.escape(image_dir) + '/*'), key=os.path.basename):
        output_file = depth_dir + "/" + Path(file).name
        if not os.path.exists(output_file):
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            input_batch = transform(img).to(device)

            with torch.no_grad():
                prediction = midas(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode='bicubic',
                    align_corners=False,
                ).squeeze()

            output = prediction.cpu().numpy()
            
            depth_map = cv2.normalize(output, None,0,1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
            depth_map = (depth_map*255).astype(np.uint8)
            #depth_map = cv2.applyColorMap(depth_map,cv2.COLORMAP_MAGMA)
            cv2.imwrite(output_file, depth_map)
        
            print('Converted: ' + Path(file).name)

    print("done.")

def merge_image_depth(output_dir):
    image_dir = output_dir + "/image/"
    depth_dir = output_dir + "/depth/"
    merged_dir = output_dir + "/merged/"

    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)

    input_dir_len = len(glob.glob(glob.escape(image_dir) + '/*'))
    if input_dir_len == len(glob.glob(glob.escape(depth_dir) + '/*')) and \
            input_dir_len == len(glob.glob(glob.escape(merged_dir) + '/*')):
        print("merge_image_depth already done.")
        return

    for file in sorted(glob.glob(glob.escape(image_dir) + '/*'), key=os.path.basename):
        depth_file = depth_dir + "/" + Path(file).name
        output_file = merged_dir + "/" + Path(file).name
        if not os.path.exists(output_file):
            image_file = cv2.imread(file)
            depth_file = cv2.imread(depth_file)
            
            height1, width1, _ = image_file.shape
            height2, width2, _ = depth_file.shape
            
            max_width = max(width1, width2)
            max_height = max(height1, height2)                        
            
            # horizontal
            # result_image = np.zeros((max_height, width1 + width2, 3), dtype=np.uint8)
            # result_image[0:height1, 0:width1] = image_file
            # result_image[0:height2, width1:width1+width2] = depth_file
            
            # vertical
            result_image = np.zeros((height1+height2, max_width, 3), dtype=np.uint8)
            result_image[0:height1, 0:width1] = image_file
            result_image[height1:height1+height2, 0:width2] = depth_file
            cv2.imwrite(output_file, result_image)

            print("Merged: " + Path(file).name)

    print("done.")

def depth2video(input, output_dir, output_name, frame_rate):
    depth_dir = output_dir + "/" + output_name + "/"
    output_file = output_dir + "/output_" + output_name + ".mp4"
    output_sound_file = output_dir + "/output_" + output_name + "_sound.mp4"

    if not os.path.exists(output_file):
        stream = ffmpeg.input(depth_dir + '%06d.png').output(output_file,
                                                             **{"framerate": frame_rate, "vcodec": "libx264",
                                                                "pix_fmt": "yuv420p"})
        ffmpeg.run(stream)
        print(Path(output_file).name + " done.")
    else:
        print(Path(output_file).name + " already done.")

    if not os.path.exists(output_sound_file):
        output_audio = ffmpeg.input(input).audio
        output_video = ffmpeg.input(output_file)
        stream = ffmpeg.output(output_audio, output_video, output_sound_file,
                               **{"c": "copy",
                                  "shortest": None})
        ffmpeg.run(stream)
        print(Path(output_sound_file).name + " done.")
    else:
        print(Path(output_sound_file).name + " already done.")

if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v","--video", type=str, help="input video file")
    parser.add_argument("-i","--image", type=str, help="input image file")
    args = parser.parse_args()
    if args.video is not None:
        runVideo(args.video)
    elif args.image is not None:
        runImage(args.image)
    else:
        parser.print_usage()
