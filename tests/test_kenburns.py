import json
import subprocess

import numpy as np
import pytest
from PIL import Image

from clea.kenburns import render_ken_burns_clip, render_ken_burns_clips


def _probe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,codec_name",
         "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    return {**data["streams"][0], "duration": float(data["format"]["duration"])}


@pytest.fixture()
def sample_image(tmp_path):
    path = tmp_path / "scene.png"
    arr = (np.random.default_rng(1).random((400, 400, 3)) * 255).astype("uint8")
    Image.fromarray(arr).save(path)
    return str(path)


def test_single_clip_matches_target_size_and_duration(sample_image, tmp_path):
    out = tmp_path / "kb.mp4"
    render_ken_burns_clip(sample_image, str(out), duration=2.0,
                          width=1080, height=1920, fps=30)
    info = _probe(str(out))
    assert (info["width"], info["height"]) == (1080, 1920)
    assert info["codec_name"] == "h264"
    assert abs(info["duration"] - 2.0) < 0.15


def test_direction_alternates_without_error(sample_image, tmp_path):
    for i in range(4):
        out = tmp_path / f"kb_{i}.mp4"
        render_ken_burns_clip(sample_image, str(out), duration=1.0,
                              width=540, height=960, fps=24, direction_index=i)
        assert out.exists() and out.stat().st_size > 0


def test_render_ken_burns_clips_batch(sample_image, tmp_path, cfg):
    out_dir = tmp_path / "clips"
    paths = render_ken_burns_clips([sample_image, sample_image, sample_image],
                                   str(out_dir), cfg, seconds_per_image=1.5)
    assert len(paths) == 3
    for p in paths:
        info = _probe(p)
        assert abs(info["duration"] - 1.5) < 0.15
