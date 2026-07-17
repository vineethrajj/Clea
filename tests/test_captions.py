from clea.assemble import EditPlan, EDLEntry
from clea.captions import build_ass, group_words, remap_words
from clea.notes import Note
from clea.transcribe import Word


def _plan() -> EditPlan:
    return EditPlan(entries=[
        EDLEntry("a.mp4", src_start=2.0, src_duration=1.5, timeline_duration=1.5,
                 transition_out="cut", fade_dur=0.0, score=0.5),
        EDLEntry("b.mp4", src_start=0.0, src_duration=2.0, timeline_duration=2.0,
                 transition_out="end", fade_dur=0.0, score=0.5),
    ])


def test_remap_words_into_timeline():
    words = {"a.mp4": [Word(2.1, 2.4, "hello"), Word(9.0, 9.3, "unused")],
             "b.mp4": [Word(0.5, 0.9, "world")]}
    remapped = remap_words(_plan(), words)
    texts = [(w.text, round(w.start, 2)) for w in remapped]
    assert ("hello", 0.10) in texts          # 2.1 - 2.0 src offset
    assert ("world", 2.00) in texts          # 0.5 + 1.5 slot offset
    assert all(w.text != "unused" for w in remapped)


def test_group_words_splits_on_punctuation_and_gap():
    words = [Word(0.0, 0.2, "one"), Word(0.3, 0.5, "two."),
             Word(0.6, 0.8, "three"), Word(2.0, 2.2, "four")]
    groups = group_words(words, max_words=4, max_gap=0.6)
    assert [len(g) for g in groups] == [2, 1, 1]


def test_ass_contains_styles_and_highlight(cfg):
    words = [Word(0.0, 0.4, "shiny"), Word(0.5, 0.9, "teeth")]
    ass = build_ass(words, cfg)
    assert "Style: Caption," in ass and "Style: Note," in ass
    assert ass.count("Dialogue:") == 2       # one event per word state
    assert r"\c&H0000D4FF" in ass            # #FFD400 highlight in BGR


def test_ass_notes_and_hook(cfg):
    ass = build_ass([], cfg,
                    notes=[Note("Floss daily", 1.0, 4.0)],
                    hook_text="STOP brushing like this")
    assert "Style: Hook," in ass
    assert "STOP brushing like this" in ass
    assert "Floss daily" in ass
    assert r"\fad(250,250)" in ass           # note fade
