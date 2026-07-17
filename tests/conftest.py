import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from clea.audio import BeatGrid
from clea.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture()
def beat_grid():
    """Synthetic 120 BPM grid, alternating strong/weak beats."""
    times = np.arange(0.5, 30.0, 0.5)
    strengths = np.tile([0.9, 0.3], len(times) // 2 + 1)[: len(times)]
    return BeatGrid(tempo=120.0, duration=30.0,
                    beat_times=times, beat_strengths=strengths)


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                   check=True)


@pytest.fixture(scope="session")
def media(tmp_path_factory):
    """Small real clips + beat track for end-to-end render tests."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available")
    import soundfile as sf

    root = tmp_path_factory.mktemp("media")
    clips = root / "clips"
    clips.mkdir()
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30", "-t", "6",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(clips / "a_first.mp4"))
    _ffmpeg("-f", "lavfi", "-i", "mandelbrot=size=640x360:rate=30", "-t", "6",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(clips / "b_second.mp4"))

    # 16s click track @120BPM so librosa locks on and target durations fit.
    sr, seconds, t_beat = 22050, 16.0, 0.5
    audio = np.zeros(int(sr * seconds))
    tt = np.arange(int(0.1 * sr)) / sr
    kick = np.sin(2 * np.pi * 70 * tt) * np.exp(-tt * 30)
    t = 0.0
    while t < seconds - 0.2:
        i = int(t * sr)
        audio[i:i + len(kick)] += kick
        t += t_beat
    track = root / "beat.wav"
    sf.write(track, (audio / np.abs(audio).max() * 0.8).astype(np.float32), sr)

    speech = None
    if shutil.which("espeak-ng"):
        wav = root / "speech.wav"
        subprocess.run(["espeak-ng", "-v", "en-us", "-s", "150", "-w", str(wav),
                        "Brushing twice a day helps prevent cavities. "
                        "Most people should visit a dentist every six months."],
                       check=True)
        speech = clips / "c_talking.mp4"
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30",
                "-i", str(wav), "-shortest",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", str(speech))

    return {"clips_dir": clips, "audio": track, "speech_clip": speech}
