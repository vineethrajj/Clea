import pytest

from clea.pipeline import EditOptions, resolve_style


def test_style_defaults_apply(cfg):
    s = resolve_style(cfg, EditOptions(style="informational"))
    assert s.captions and s.notes and s.keep_voice
    assert s.ordering == "hook_first" and s.transitions == "standard"
    assert not s.punch_in


def test_cool_edit_is_hype(cfg):
    s = resolve_style(cfg, EditOptions(style="cool-edit"))
    assert not s.captions and not s.keep_voice
    assert s.transitions == "flash"
    assert s.punch_in
    assert s.pacing["body_min_cut"] < 1.0    # faster cuts than base


def test_case_study_is_chronological(cfg):
    s = resolve_style(cfg, EditOptions(style="case-study"))
    assert s.ordering == "chronological"
    assert s.keep_voice


def test_explicit_flags_override_style(cfg):
    s = resolve_style(cfg, EditOptions(style="cool-edit", captions=True,
                                       keep_voice=True))
    assert s.captions and s.keep_voice       # user said so
    assert s.punch_in                        # style still drives the rest


def test_unknown_style_raises(cfg):
    with pytest.raises(ValueError, match="unknown style"):
        resolve_style(cfg, EditOptions(style="vlog"))


def test_style_pacing_merges_over_base(cfg):
    s = resolve_style(cfg, EditOptions(style="educational"))
    assert s.pacing["max_cut"] == 3.5        # from preset
    # keys not overridden by the preset still exist
    assert "crossfade" in s.pacing
