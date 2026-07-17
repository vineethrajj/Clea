import pytest

from clea.assemble import EditPlan, EDLEntry
from clea.hardware import HardwareReport
from clea.render import _escape_filter_path, build_ffmpeg_command, choose_encoder_args


def _plan(fade_kind="fade") -> EditPlan:
    return EditPlan(entries=[
        EDLEntry("a.mp4", 0.0, 1.2, 1.0, "xfade", 0.2, 0.5, fade_kind=fade_kind),
        EDLEntry("b.mp4", 3.0, 1.5, 1.5, "cut", 0.0, 0.4),
        EDLEntry("a.mp4", 5.0, 2.0, 2.0, "end", 0.0, 0.3),
    ])


def _hw(nvenc: bool) -> HardwareReport:
    return HardwareReport(ffmpeg=True, nvenc=nvenc)


def test_encoder_selection_auto(cfg):
    assert choose_encoder_args(cfg, _hw(True))[0] == "h264_nvenc"
    assert choose_encoder_args(cfg, _hw(False))[0] == "libx264"


def test_nvenc_args_used_when_available(cfg):
    cmd = build_ffmpeg_command(_plan(), "music.wav", "out.mp4", cfg, _hw(True))
    assert "h264_nvenc" in cmd and "libx264" not in cmd


def test_xfade_offset_is_timeline_boundary(cfg):
    cmd = build_ffmpeg_command(_plan(), "music.wav", "out.mp4", cfg, _hw(False))
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "xfade=transition=fade:duration=0.2000:offset=1.0000" in graph
    assert "concat=n=2:v=1:a=0" in graph      # the hard cut join


def test_flash_transition_name_propagates(cfg):
    cmd = build_ffmpeg_command(_plan("fadewhite"), "music.wav", "out.mp4",
                               cfg, _hw(False))
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "xfade=transition=fadewhite" in graph


def test_punch_in_adds_zoompan(cfg):
    cmd = build_ffmpeg_command(_plan(), "music.wav", "out.mp4", cfg, _hw(False),
                               punch_in=True)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph.count("zoompan=") == 3       # one per segment
    assert "min(1+0.0025*on,1.12)" in graph   # push in
    assert "max(1.12-0.0025*on,1.0)" in graph  # pull out


def test_keep_voice_builds_mix(cfg, monkeypatch):
    monkeypatch.setattr("clea.render.clip_has_audio", lambda p: p == "a.mp4")
    cmd = build_ffmpeg_command(_plan(), "music.wav", "out.mp4", cfg, _hw(False),
                               keep_voice=True)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "amix=inputs=2" in graph
    assert "anullsrc" in graph                # b.mp4 has no audio -> silence pad
    assert f"volume={cfg.audio_mix['music_volume_with_voice']}" in graph


def test_ass_burn_in_appended(cfg):
    cmd = build_ffmpeg_command(_plan(), "music.wav", "out.mp4", cfg, _hw(False),
                               ass_path="subs/reel.ass")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "ass='subs/reel.ass'[vout]" in graph


@pytest.mark.parametrize("raw,expected", [
    (r"C:\videos\reel.ass", r"C\:/videos/reel.ass"),
    ("with'quote.ass", r"with\'quote.ass"),
])
def test_escape_filter_path(raw, expected):
    assert _escape_filter_path(raw) == expected


def test_total_duration_flag_matches_plan(cfg):
    plan = _plan()
    cmd = build_ffmpeg_command(plan, "music.wav", "out.mp4", cfg, _hw(False))
    assert cmd[cmd.index("-t") + 1] == f"{plan.total_duration:.4f}"
