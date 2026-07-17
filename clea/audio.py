"""Beat and tempo detection on the music track (librosa)."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class BeatGrid:
    tempo: float
    duration: float
    beat_times: np.ndarray      # seconds
    beat_strengths: np.ndarray  # onset-envelope value at each beat, 0..1 normalised


def analyze_beats(audio_path: str, sr: int = 22050) -> BeatGrid:
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    duration = float(len(y) / sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    strengths = onset_env[np.clip(beat_frames, 0, len(onset_env) - 1)]
    peak = float(strengths.max()) if len(strengths) and strengths.max() > 0 else 1.0
    strengths = strengths / peak

    return BeatGrid(
        tempo=float(np.atleast_1d(tempo)[0]),
        duration=duration,
        beat_times=np.asarray(beat_times, dtype=float),
        beat_strengths=np.asarray(strengths, dtype=float),
    )
