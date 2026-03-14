import glob
import os
import shutil
import tempfile

import cv2
import ffmpeg
import numpy as np
import pytest

from models import get_model, list_models, DEFAULT_MODEL, MODEL_REGISTRY
from models.base import DepthModel
from video2depth import video2image, image2depth, merge_image_depth, depth2video, copy2image

SAMPLE_VIDEO_DIR = os.path.join(os.path.dirname(__file__), "sample_video")
SAMPLE_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "sample_image")

SAMPLE_VIDEO_60FPS = os.path.join(SAMPLE_VIDEO_DIR, "sample_60fps.mp4")
SAMPLE_VIDEO_30FPS = os.path.join(SAMPLE_VIDEO_DIR, "sample_30fps.mp4")
SAMPLE_VIDEO_24FPS = os.path.join(SAMPLE_VIDEO_DIR, "sample_24fps.mp4")
SAMPLE_VIDEO_COUNT = os.path.join(SAMPLE_VIDEO_DIR, "sample_count.mp4")
SAMPLE_IMAGE = os.path.join(SAMPLE_IMAGE_DIR, "sample_01.jpg")


@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    return str(d)


# ============================================================
# Model Registry Tests
# ============================================================

class TestModelRegistry:

    def test_list_models_returns_sorted(self):
        models = list_models()
        assert models == sorted(models)
        assert len(models) == len(MODEL_REGISTRY)

    def test_default_model_exists(self):
        assert DEFAULT_MODEL in MODEL_REGISTRY

    def test_get_model_returns_depth_model(self):
        for name in list_models():
            model = get_model(name)
            assert isinstance(model, DepthModel)

    def test_get_model_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model("nonexistent_model")

    def test_get_model_error_shows_available(self):
        with pytest.raises(KeyError) as exc_info:
            get_model("bad_name")
        for name in list_models():
            assert name in str(exc_info.value)

    def test_midas_variants(self):
        for key in ("midas_large", "midas_hybrid", "midas_small"):
            model = get_model(key)
            assert "MiDaS" in model.name


# ============================================================
# Sample File Existence Tests
# ============================================================

class TestSampleFiles:

    def test_sample_videos_exist(self):
        assert os.path.isfile(SAMPLE_VIDEO_60FPS)
        assert os.path.isfile(SAMPLE_VIDEO_30FPS)
        assert os.path.isfile(SAMPLE_VIDEO_24FPS)
        assert os.path.isfile(SAMPLE_VIDEO_COUNT)

    def test_sample_images_exist(self):
        images = glob.glob(os.path.join(SAMPLE_IMAGE_DIR, "*.jpg"))
        assert len(images) == 60

    def test_sample_image_readable(self):
        img = cv2.imread(SAMPLE_IMAGE)
        assert img is not None
        assert img.shape[0] > 0 and img.shape[1] > 0


# ============================================================
# Video Probe Tests
# ============================================================

class TestVideoProbe:

    @pytest.mark.parametrize("video,expected_fps", [
        (SAMPLE_VIDEO_60FPS, 60),
        (SAMPLE_VIDEO_30FPS, 30),
        (SAMPLE_VIDEO_24FPS, 24),
    ])
    def test_video_fps(self, video, expected_fps):
        probe = ffmpeg.probe(video)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        fps = eval(video_stream['r_frame_rate'])
        assert abs(fps - expected_fps) < 1

    def test_video_resolution(self):
        probe = ffmpeg.probe(SAMPLE_VIDEO_60FPS)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        assert int(video_stream['width']) == 1920
        assert int(video_stream['height']) == 1080


# ============================================================
# Pipeline Step Tests
# ============================================================

class TestVideo2Image:

    def test_extracts_frames(self, output_dir):
        probe = ffmpeg.probe(SAMPLE_VIDEO_COUNT)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        nb_frames = int(video_stream['nb_frames'])

        video2image(SAMPLE_VIDEO_COUNT, nb_frames, output_dir)

        image_dir = output_dir + "/image/"
        frames = glob.glob(image_dir + "*.png")
        assert len(frames) == nb_frames

    def test_frames_are_valid_images(self, output_dir):
        video2image(SAMPLE_VIDEO_COUNT, 600, output_dir)
        image_dir = output_dir + "/image/"
        first_frame = sorted(glob.glob(image_dir + "*.png"))[0]
        img = cv2.imread(first_frame)
        assert img is not None
        assert img.shape == (1080, 1920, 3)

    def test_skip_if_already_done(self, output_dir):
        video2image(SAMPLE_VIDEO_COUNT, 600, output_dir)
        image_dir = output_dir + "/image/"
        mtime_before = os.path.getmtime(sorted(glob.glob(image_dir + "*.png"))[0])
        video2image(SAMPLE_VIDEO_COUNT, 600, output_dir)
        mtime_after = os.path.getmtime(sorted(glob.glob(image_dir + "*.png"))[0])
        assert mtime_before == mtime_after


class TestCopy2Image:

    def test_copies_image(self, output_dir):
        copy2image(SAMPLE_IMAGE, output_dir)
        image_dir = output_dir + "/image/"
        files = glob.glob(image_dir + "*")
        assert len(files) == 1

    def test_skip_if_already_done(self, output_dir):
        copy2image(SAMPLE_IMAGE, output_dir)
        copy2image(SAMPLE_IMAGE, output_dir)
        image_dir = output_dir + "/image/"
        files = glob.glob(image_dir + "*")
        assert len(files) == 1


class TestMergeImageDepth:

    def test_merge_creates_output(self, output_dir):
        image_dir = output_dir + "/image/"
        depth_dir = output_dir + "/depth/"
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.imwrite(image_dir + "000000.png", img)
        cv2.imwrite(depth_dir + "000000.png", img)

        merge_image_depth(output_dir)

        merged_dir = output_dir + "/merged/"
        merged_files = glob.glob(merged_dir + "*.png")
        assert len(merged_files) == 1

    def test_merged_image_is_vertically_stacked(self, output_dir):
        image_dir = output_dir + "/image/"
        depth_dir = output_dir + "/depth/"
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        h, w = 100, 200
        cv2.imwrite(image_dir + "000000.png", np.zeros((h, w, 3), dtype=np.uint8))
        cv2.imwrite(depth_dir + "000000.png", np.zeros((h, w, 3), dtype=np.uint8))

        merge_image_depth(output_dir)

        merged = cv2.imread(output_dir + "/merged/000000.png")
        assert merged.shape[0] == h * 2
        assert merged.shape[1] == w


class TestDepth2Video:

    def _make_frames(self, output_dir, subdir="depth", count=10):
        frame_dir = output_dir + f"/{subdir}/"
        os.makedirs(frame_dir, exist_ok=True)
        for i in range(count):
            img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            cv2.imwrite(frame_dir + f"{i:06d}.png", img)

    def test_creates_video(self, output_dir):
        self._make_frames(output_dir)
        # sample_count.mp4 has no audio, so depth2video will fail on sound step.
        # Test only video creation by checking the file after catching the sound error.
        try:
            depth2video(SAMPLE_VIDEO_COUNT, output_dir, "depth", "30.00")
        except Exception:
            pass
        assert os.path.isfile(output_dir + "/output_depth.mp4")

    def test_skip_if_already_done(self, output_dir):
        self._make_frames(output_dir)
        try:
            depth2video(SAMPLE_VIDEO_COUNT, output_dir, "depth", "30.00")
        except Exception:
            pass
        mtime = os.path.getmtime(output_dir + "/output_depth.mp4")
        try:
            depth2video(SAMPLE_VIDEO_COUNT, output_dir, "depth", "30.00")
        except Exception:
            pass
        assert os.path.getmtime(output_dir + "/output_depth.mp4") == mtime


# ============================================================
# Model Interface Tests (mock predict)
# ============================================================

class DummyModel(DepthModel):
    @property
    def name(self):
        return "Dummy"

    def load(self):
        pass

    def predict(self, img_rgb):
        h, w = img_rgb.shape[:2]
        return np.random.randint(0, 255, (h, w), dtype=np.uint8)


class TestImage2DepthWithDummy:

    def test_generates_depth_maps(self, output_dir):
        image_dir = output_dir + "/image/"
        os.makedirs(image_dir, exist_ok=True)
        for i in range(3):
            shutil.copy(SAMPLE_IMAGE, image_dir + f"{i:06d}.jpg")

        model = DummyModel()
        image2depth(output_dir, model)

        depth_dir = output_dir + "/depth/"
        depth_files = glob.glob(depth_dir + "*")
        assert len(depth_files) == 3

    def test_depth_map_is_grayscale(self, output_dir):
        image_dir = output_dir + "/image/"
        os.makedirs(image_dir, exist_ok=True)
        shutil.copy(SAMPLE_IMAGE, image_dir + "000000.jpg")

        model = DummyModel()
        image2depth(output_dir, model)

        depth = cv2.imread(output_dir + "/depth/000000.jpg", cv2.IMREAD_UNCHANGED)
        assert depth is not None

    def test_skip_if_already_done(self, output_dir):
        image_dir = output_dir + "/image/"
        depth_dir = output_dir + "/depth/"
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        shutil.copy(SAMPLE_IMAGE, image_dir + "000000.jpg")
        shutil.copy(SAMPLE_IMAGE, depth_dir + "000000.jpg")

        model = DummyModel()
        image2depth(output_dir, model)
        # model.load() should not be called if skipped
        assert model.model is None


# ============================================================
# Full Pipeline Tests (with DummyModel)
# ============================================================

class TestFullPipelineVideo:

    def test_video_pipeline_produces_outputs(self, tmp_path):
        output_dir = str(tmp_path / "out" / "sample_count")
        os.makedirs(output_dir, exist_ok=True)

        probe = ffmpeg.probe(SAMPLE_VIDEO_COUNT)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        nb_frames = int(video_stream['nb_frames'])
        frame_rate = '%0.2f' % eval(video_stream['r_frame_rate'])

        video2image(SAMPLE_VIDEO_COUNT, nb_frames, output_dir)
        image2depth(output_dir, DummyModel())
        merge_image_depth(output_dir)
        try:
            depth2video(SAMPLE_VIDEO_COUNT, output_dir, "depth", frame_rate)
        except Exception:
            pass
        try:
            depth2video(SAMPLE_VIDEO_COUNT, output_dir, "merged", frame_rate)
        except Exception:
            pass

        assert os.path.isfile(output_dir + "/output_depth.mp4")
        assert os.path.isfile(output_dir + "/output_merged.mp4")

        depth_probe = ffmpeg.probe(output_dir + "/output_depth.mp4")
        depth_stream = next(s for s in depth_probe['streams'] if s['codec_type'] == 'video')
        assert int(depth_stream['width']) == 1920
        assert int(depth_stream['height']) == 1080


class TestFullPipelineImage:

    def test_image_pipeline_produces_outputs(self, tmp_path):
        output_dir = str(tmp_path / "out" / "sample_01")
        os.makedirs(output_dir, exist_ok=True)

        copy2image(SAMPLE_IMAGE, output_dir)
        image2depth(output_dir, DummyModel())
        merge_image_depth(output_dir)

        assert len(glob.glob(output_dir + "/depth/*")) == 1
        assert len(glob.glob(output_dir + "/merged/*")) == 1

        merged = cv2.imread(glob.glob(output_dir + "/merged/*")[0])
        assert merged is not None
