import numpy as np

from clea.assemble import build_edl, build_slots
from clea.scoring import ClipAnalysis

PACING = {"hook_seconds": 6.0, "hook_min_cut": 0.45, "body_min_cut": 0.95,
          "max_cut": 2.5, "strong_beat_percentile": 60, "crossfade": 0.2}
SCORING = {"repeat_clip_penalty": 0.15}


def _analysis(path: str, duration: float, base: float) -> ClipAnalysis:
    times = np.arange(0, duration, 0.25)
    scores = np.full(len(times), base)
    return ClipAnalysis(path=path, duration=duration,
                        sample_times=times, scores=scores)


def test_slots_land_on_beats(beat_grid):
    slots = build_slots(beat_grid, 20.0, PACING)
    boundaries = np.cumsum([s.duration for s in slots])
    beats = set(np.round(beat_grid.beat_times, 3))
    # every internal boundary is a beat (final boundary is the target cut-off)
    for b in boundaries[:-1]:
        assert round(float(b), 3) in beats, f"boundary {b} not on a beat"


def test_slot_durations_respect_pacing(beat_grid):
    slots = build_slots(beat_grid, 20.0, PACING)
    total = sum(s.duration for s in slots)
    assert abs(total - 20.0) < 0.05
    for s in slots[:-1]:
        assert s.duration >= PACING["hook_min_cut"] - 1e-6
        assert s.duration <= PACING["max_cut"] * 1.5 + 1e-6
    assert slots[-1].transition_out == "end"


def test_standard_transitions_strong_cut_weak_fade(beat_grid):
    slots = build_slots(beat_grid, 20.0, PACING, transitions="standard")
    kinds = {s.transition_out for s in slots}
    assert "cut" in kinds and "xfade" in kinds
    for s in slots:
        if s.transition_out == "xfade":
            assert s.fade_kind == "fade"


def test_flash_transitions_use_fadewhite(beat_grid):
    slots = build_slots(beat_grid, 20.0, PACING, transitions="flash")
    fades = [s for s in slots if s.transition_out == "xfade"]
    assert fades, "flash style should produce xfade slots on strong beats"
    assert all(s.fade_kind == "fadewhite" for s in fades)


def test_edl_hook_first_prefers_best_clip(beat_grid):
    analyses = [
        _analysis("boring.mp4", 12.0, 0.05),
        _analysis("great.mp4", 12.0, 0.90),
    ]
    slots = build_slots(beat_grid, 18.0, PACING)
    plan = build_edl(analyses, slots, SCORING, ordering="hook_first")
    assert plan.entries[0].clip == "great.mp4"
    assert abs(plan.total_duration - sum(s.duration for s in slots)) < 1e-6


def test_edl_no_source_overlap_within_clip(beat_grid):
    analyses = [_analysis(f"c{i}.mp4", 15.0, 0.5) for i in range(3)]
    slots = build_slots(beat_grid, 20.0, PACING)
    plan = build_edl(analyses, slots, SCORING)
    by_clip: dict[str, list[tuple[float, float]]] = {}
    for e in plan.entries:
        for s0, e0 in by_clip.get(e.clip, []):
            assert e.src_start >= e0 - 1e-6 or e.src_start + e.src_duration <= s0 + 1e-6
        by_clip.setdefault(e.clip, []).append(
            (e.src_start, e.src_start + e.src_duration))


def test_chronological_preserves_clip_and_time_order(beat_grid):
    analyses = [
        _analysis("a_before.mp4", 12.0, 0.5),
        _analysis("b_after.mp4", 12.0, 0.5),
    ]
    slots = build_slots(beat_grid, 20.0, PACING)
    plan = build_edl(analyses, slots, SCORING, ordering="chronological")
    clip_seq = [e.clip for e in plan.entries]
    # once we switch to the second clip we never go back
    switched = clip_seq.index("b_after.mp4")
    assert all(c == "b_after.mp4" for c in clip_seq[switched:])
    # source positions move forward within each clip
    for clip in ("a_before.mp4", "b_after.mp4"):
        starts = [e.src_start for e in plan.entries if e.clip == clip]
        assert starts == sorted(starts)


def test_xfade_extension_preserves_timeline(beat_grid):
    analyses = [_analysis(f"c{i}.mp4", 15.0, 0.5) for i in range(2)]
    slots = build_slots(beat_grid, 20.0, PACING)
    plan = build_edl(analyses, slots, SCORING)
    for e in plan.entries:
        if e.transition_out == "xfade":
            assert abs(e.src_duration - e.timeline_duration - e.fade_dur) < 1e-6
        else:
            assert abs(e.src_duration - e.timeline_duration) < 1e-6
