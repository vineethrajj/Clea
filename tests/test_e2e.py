"""End-to-end regression: every style preset renders a valid 9:16 reel."""

import json
import subprocess

import pytest

from clea.pipeline import EditOptions, collect_clips, run_edit


def _probe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,codec_name",
         "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    return {**data["streams"][0], "duration": float(data["format"]["duration"])}


@pytest.mark.parametrize("style", ["informational", "cool-edit",
                                   "educational", "case-study"])
def test_style_renders_valid_reel(style, media, cfg, tmp_path):
    out = tmp_path / f"{style}.mp4"
    opts = EditOptions(style=style, duration=15.0,
                       captions=False, notes=False, keep_voice=False)
    result = run_edit(collect_clips(media["clips_dir"]), str(media["audio"]),
                      str(out), cfg, opts)
    info = _probe(str(out))
    assert (info["width"], info["height"]) == (1080, 1920)
    assert info["codec_name"] == "h264"
    # audio is 16s -> target clamps to 15s
    assert abs(info["duration"] - result.total_duration) < 0.2
    assert result.n_cuts >= 5


def test_hook_text_burns_in(media, cfg, tmp_path):
    out = tmp_path / "hook.mp4"
    opts = EditOptions(style="cool-edit", duration=15.0, captions=False,
                       notes=False, keep_voice=False,
                       hook_text="SMILE CHECK")
    run_edit(collect_clips(media["clips_dir"]), str(media["audio"]),
             str(out), cfg, opts)
    ass = out.with_suffix(".ass").read_text()
    assert "SMILE CHECK" in ass and "Style: Hook," in ass


def test_captions_and_notes_end_to_end(media, cfg, tmp_path):
    if media["speech_clip"] is None:
        pytest.skip("espeak-ng not available for speech clip")
    out = tmp_path / "capnotes.mp4"
    opts = EditOptions(style="informational", duration=15.0,
                       whisper_model="small")   # cached CPU model
    result = run_edit(collect_clips(media["clips_dir"]), str(media["audio"]),
                      str(out), cfg, opts)
    assert out.exists()
    ass = out.with_suffix(".ass")
    assert ass.exists()
    text = ass.read_text()
    assert "Dialogue:" in text
    assert result.n_caption_words > 0
