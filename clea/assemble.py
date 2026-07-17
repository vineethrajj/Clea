"""Edit-decision-list assembly: map beat boundaries to scored clip segments.

Pacing model ("high-engagement" Reels style):
  * hook section (first ~6s): fastest cuts, best-scored material first
  * body: slightly longer cuts, still beat-aligned
  * hard cut on strong beats, short crossfade on weak beats

Crossfade timing: an xfade overlaps the two segments by `crossfade` seconds,
which would normally shift every later cut off the beat. To keep the grid
aligned we source EXTRA material at the end of the outgoing segment equal to
the fade duration, so the transition consumes the extension and every
boundary still lands exactly on its beat.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .audio import BeatGrid
from .scoring import ClipAnalysis, segment_score


@dataclass
class Slot:
    start: float          # timeline position (s)
    duration: float       # timeline length (s)
    transition_out: str   # "cut" | "xfade" | "end"
    fade_dur: float = 0.0


@dataclass
class EDLEntry:
    clip: str
    src_start: float
    src_duration: float   # includes fade extension when transition_out == xfade
    timeline_duration: float
    transition_out: str
    fade_dur: float
    score: float


@dataclass
class EditPlan:
    entries: list[EDLEntry] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return sum(e.timeline_duration for e in self.entries)


def build_slots(grid: BeatGrid, target_duration: float, pacing: dict) -> list[Slot]:
    hook_seconds = float(pacing["hook_seconds"])
    hook_min = float(pacing["hook_min_cut"])
    body_min = float(pacing["body_min_cut"])
    max_cut = float(pacing["max_cut"])
    fade = float(pacing["crossfade"])
    strong_pct = float(pacing["strong_beat_percentile"])

    target = min(target_duration, grid.duration)
    strong_threshold = float(np.percentile(grid.beat_strengths, strong_pct)) \
        if len(grid.beat_strengths) else 0.5

    # Walk the beat list accumulating boundaries; a boundary is accepted once
    # the running cut is long enough for the current pacing phase.
    boundaries: list[tuple[float, float]] = []  # (time, strength)
    last = 0.0
    for t, s in zip(grid.beat_times, grid.beat_strengths):
        if t <= last or t >= target:
            continue
        min_cut = hook_min if t < hook_seconds else body_min
        if t - last >= min_cut:
            boundaries.append((float(t), float(s)))
            last = float(t)

    slots: list[Slot] = []
    prev = 0.0
    for t, s in boundaries:
        length = t - prev
        # Split cuts that overshoot max_cut (sparse beats / quiet intros).
        while length > max_cut * 1.5:
            slots.append(Slot(prev, max_cut, "cut"))
            prev += max_cut
            length = t - prev
        transition = "cut" if s >= strong_threshold else "xfade"
        slots.append(Slot(prev, length, transition, fade if transition == "xfade" else 0.0))
        prev = t
    if target - prev > 0.25:
        slots.append(Slot(prev, target - prev, "cut"))
    if slots:
        slots[-1].transition_out = "end"
        slots[-1].fade_dur = 0.0
    return slots


def _pick_segment(
    analyses: list[ClipAnalysis],
    needed: float,
    used: dict[str, list[tuple[float, float]]],
    last_clip: str | None,
    repeat_penalty: float,
    rng: random.Random,
) -> tuple[ClipAnalysis, float, float]:
    """Best-scoring unused window of length `needed` across all clips."""
    best: tuple[float, ClipAnalysis, float] | None = None
    for a in analyses:
        if a.duration < needed + 0.05:
            continue
        starts = np.arange(0.0, a.duration - needed, 0.25)
        if not len(starts):
            starts = np.array([0.0])
        for st in starts:
            if any(st < ue and st + needed > us for us, ue in used.get(a.path, [])):
                continue
            sc = segment_score(a, float(st), needed) + rng.uniform(0, 1e-4)
            if a.path == last_clip:
                sc -= repeat_penalty
            if best is None or sc > best[0]:
                best = (sc, a, float(st))
    if best is None:
        # Everything consumed: allow reuse, pick the globally best window.
        a = max(analyses, key=lambda c: c.duration)
        st = rng.uniform(0, max(a.duration - needed, 0.0))
        return a, st, segment_score(a, st, needed)
    return best[1], best[2], best[0]


def build_edl(
    analyses: list[ClipAnalysis],
    slots: list[Slot],
    scoring_cfg: dict,
    seed: int = 42,
) -> EditPlan:
    rng = random.Random(seed)
    repeat_penalty = float(scoring_cfg["repeat_clip_penalty"])
    used: dict[str, list[tuple[float, float]]] = {}
    plan = EditPlan()
    last_clip: str | None = None

    for slot in slots:
        fade_ext = slot.fade_dur if slot.transition_out == "xfade" else 0.0
        needed = slot.duration + fade_ext
        analysis, src_start, score = _pick_segment(
            analyses, needed, used, last_clip, repeat_penalty, rng
        )
        # Clamp the fade if the clip genuinely runs out of material.
        avail = analysis.duration - src_start
        if avail < needed:
            fade_ext = max(0.0, min(fade_ext, avail - slot.duration))
            needed = slot.duration + fade_ext
        used.setdefault(analysis.path, []).append((src_start, src_start + needed))
        transition = slot.transition_out if fade_ext > 0 or slot.transition_out != "xfade" else "cut"
        plan.entries.append(EDLEntry(
            clip=analysis.path,
            src_start=src_start,
            src_duration=needed,
            timeline_duration=slot.duration,
            transition_out=transition,
            fade_dur=fade_ext,
            score=score,
        ))
        last_clip = analysis.path
    return plan
