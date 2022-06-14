import argparse
import glob
import os
import time
from pathlib import Path

import cv2
import ffmpeg
import numpy as np
import torch
from PIL import Image


def run(input):
    output_dir = Path(input).stem
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
    video2image(input, nb_frames, output_dir + "/image/")

    print()
    print("==================================================================")
    print(" image to depth... ")
    print("==================================================================")
    image2depth(output_dir + "/image/", output_dir + "/depth/", False)

    print()
    print("==================================================================")
    print(" merge to image and depth... ")
    print("==================================================================")
    merge_image_depth(output_dir + "/image/", output_dir + "/depth/", output_dir + "/merged/")

    print()
    print("==================================================================")
    print(" depth to video... ")
    print("==================================================================")
    depth2video(input, output_dir + "/depth/", output_dir + "/output.mp4", output_dir + "/output_sound.mp4", frame_rate)
    print()
    depth2video(input, output_dir + "/merged/", output_dir + "/output_merged.mp4",
                output_dir + "/output_merged_sound.mp4", frame_rate)
    print()

    end = time.time()
    print()
    print("elapsed time", end - start)


def video2image(input, nb_frames, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if nb_frames == len(glob.glob(glob.escape(output_dir) + '*')):
        print("video2image already done.")
        return

    stream = ffmpeg.input(input).output(output_dir + '%06d.jpg',
                                        **{"start_number": 0, "qmin": 1, "qmax": 1, "qscale:v": 1})
    ffmpeg.run(stream)

    print("done.")


def image2depth(input_dir, depth_dir, use_large_model):
    if len(glob.glob(glob.escape(input_dir) + '/*')) == len(glob.glob(glob.escape(depth_dir) + '/*')):
        print("image2depth already done.")
        return

    if use_large_model:
        midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
    else:
        midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

    if use_large_model:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    for file in sorted(glob.glob(glob.escape(input_dir) + '/*'), key=os.path.basename):
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

            output_normalized = (output * 255 / np.max(output)).astype('uint8')
            output_image = Image.fromarray(output_normalized)
            output_image_converted = output_image.convert('RGB').save(output_file)
            print('Converted: ' + Path(file).name)

    print("done.")


def merge_image_depth(input_dir, depth_dir, merged_dir):
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)

    input_dir_len = len(glob.glob(glob.escape(input_dir) + '/*'))
    if input_dir_len == len(glob.glob(glob.escape(depth_dir) + '/*')) and \
            input_dir_len == len(glob.glob(glob.escape(merged_dir) + '/*')):
        print("merge_image_depth already done.")
        return

    for file in sorted(glob.glob(glob.escape(input_dir) + '/*'), key=os.path.basename):
        depth_file = depth_dir + "/" + Path(file).name
        output_file = merged_dir + "/" + Path(file).name
        if not os.path.exists(output_file):
            image_file = Image.open(file)
            depth_file = Image.open(depth_file)
            get_concat_v(image_file, depth_file).save(output_file)
            print("Merged: " + Path(file).name)

    print("done.")


def depth2video(input, depth_dir, output_file, output_sound_file, frame_rate):
    if not os.path.exists(output_file):
        stream = ffmpeg.input(depth_dir + '%06d.jpg').output(output_file,
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


def get_concat_v(image1, image2):
    dst = Image.new('RGB', (image1.width, image1.height + image2.height))
    dst.paste(image1, (0, 0))
    dst.paste(image2, (0, image1.height))
    return dst


if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str,
                        help="input video file")
    args = parser.parse_args()
    print(args)
    run(args.input)
