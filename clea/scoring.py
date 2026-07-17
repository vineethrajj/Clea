"""Per-clip "interestingness" scoring via OpenCV.

Motion (mean absolute frame difference) is the primary signal, with a small
sharpness/contrast bonus, so static or flat shots score low and high-energy
moments float to the top. Analysis runs on downscaled greyscale frames so a
whole folder of clips scores in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ClipAnalysis:
    path: str
    duration: float
    sample_times: np.ndarray  # seconds, one per analysed frame interval
    scores: np.ndarray        # same length, higher = more interesting


def analyze_clip(path: str, sample_fps: float = 4.0, analysis_width: int = 160) -> ClipAnalysis:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / src_fps if frame_count else 0.0
    step = max(1, round(src_fps / sample_fps))

    times: list[float] = []
    scores: list[float] = []
    prev: np.ndarray | None = None
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = analysis_width / max(w, 1)
            small = cv2.resize(frame, (analysis_width, max(2, int(h * scale))))
            grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                motion = float(np.mean(cv2.absdiff(grey, prev))) / 255.0
                sharpness = min(float(cv2.Laplacian(grey, cv2.CV_64F).var()) / 500.0, 1.0)
                contrast = float(grey.std()) / 128.0
                times.append(idx / src_fps)
                scores.append(motion + 0.15 * sharpness + 0.10 * contrast)
            prev = grey
        idx += 1
    cap.release()

    if not duration and times:
        duration = times[-1] + 1.0 / sample_fps
    if not times:  # single-frame or unreadable-body clip: neutral flat score
        times, scores = [0.0], [0.1]

    return ClipAnalysis(
        path=path,
        duration=float(duration),
        sample_times=np.asarray(times, dtype=float),
        scores=np.asarray(scores, dtype=float),
    )


def segment_score(analysis: ClipAnalysis, start: float, length: float) -> float:
    """Mean score over [start, start+length]."""
    mask = (analysis.sample_times >= start) & (analysis.sample_times <= start + length)
    if not mask.any():
        i = int(np.argmin(np.abs(analysis.sample_times - (start + length / 2))))
        return float(analysis.scores[i])
    return float(analysis.scores[mask].mean())
