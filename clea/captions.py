"""Caption generation: remap transcript words onto the edit timeline and
render them as an ASS subtitle file with word-by-word highlight.

Style: the whole caption group stays on screen while the currently spoken
word is highlighted (colour pop) — one Dialogue event per word state, which
is more robust across libass versions than \\k karaoke timing.
"""

from __future__ import annotations

from pathlib import Path

from .assemble import EditPlan
from .config import Config
from .transcribe import Word

_SENTENCE_END = (".", "!", "?", ",", ":", ";")


def remap_words(plan: EditPlan, words_by_clip: dict[str, list[Word]]) -> list[Word]:
    """Project source-clip words into timeline coordinates.

    A word belongs to a timeline segment when its midpoint falls inside the
    segment's source window; crossfade extension frames are excluded so words
    never bleed past a cut boundary.
    """
    out: list[Word] = []
    tl_pos = 0.0
    for e in plan.entries:
        for w in words_by_clip.get(e.clip, []):
            mid = (w.start + w.end) / 2
            if e.src_start <= mid < e.src_start + e.timeline_duration:
                start = max(w.start - e.src_start, 0.0) + tl_pos
                end = min(w.end - e.src_start, e.timeline_duration) + tl_pos
                if end - start > 0.02:
                    out.append(Word(start, end, w.text))
        tl_pos += e.timeline_duration
    out.sort(key=lambda w: w.start)
    return out


def group_words(words: list[Word], max_words: int = 4,
                max_gap: float = 0.6) -> list[list[Word]]:
    groups: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        if current and (
            len(current) >= max_words
            or w.start - current[-1].end > max_gap
            or current[-1].text.endswith(_SENTENCE_END)
        ):
            groups.append(current)
            current = []
        current.append(w)
    if current:
        groups.append(current)
    return groups


def _ass_time(t: float) -> str:
    t = max(t, 0.0)
    cs = round(t * 100)
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _hex_to_ass(colour: str) -> str:
    """'#RRGGBB' -> ASS '&H00BBGGRR'."""
    c = colour.lstrip("#")
    r, g, b = c[0:2], c[2:4], c[4:6]
    return f"&H00{b}{g}{r}".upper()


def _clean(text: str) -> str:
    return text.replace("{", "(").replace("}", ")")


def build_ass(words: list[Word], cfg: Config, notes: list | None = None,
              hook_text: str | None = None) -> str:
    ccfg = cfg.captions
    ncfg = cfg.notes
    hcfg = cfg.hook_title
    w, h = cfg.video["width"], cfg.video["height"]
    highlight = _hex_to_ass(ccfg["highlight_color"])
    note_bg = _hex_to_ass(ncfg["bg_color"]).replace("&H00", f"&H{ncfg['bg_alpha']:02X}")
    note_text = _hex_to_ass(ncfg["text_color"])
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{ccfg["font"]},{ccfg["font_size"]},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{ccfg["outline"]},2,2,60,60,{ccfg["margin_v"]},1
Style: Note,{ccfg["font"]},{ncfg["font_size"]},{note_text},{note_text},{note_bg},{note_bg},-1,0,0,0,100,100,0,0,3,14,0,8,90,90,{ncfg["margin_top"]},1
Style: Hook,{ccfg["font"]},{hcfg["font_size"]},{_hex_to_ass(hcfg["color"])},{_hex_to_ass(hcfg["color"])},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,3,5,70,70,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    if hook_text and hook_text.strip():
        # Big opening title, centre of frame, quick pop-in (Alignment 5 = middle).
        events.append(
            f"Dialogue: 2,{_ass_time(0.15)},{_ass_time(hcfg['seconds'])},Hook,,0,0,0,,"
            r"{\fad(120,220)}" + _clean(hook_text.strip())
        )
    for note in notes or []:
        events.append(
            f"Dialogue: 1,{_ass_time(note.start)},{_ass_time(note.end)},Note,,0,0,0,,"
            r"{\fad(250,250)}" + _clean(note.text)
        )
    for group in group_words(words, ccfg["max_words_per_line"]):
        for i, word in enumerate(group):
            start = word.start
            # Hold the highlight until the next word starts (no flicker in gaps),
            # and give the last word a short tail so lines don't vanish abruptly.
            end = group[i + 1].start if i + 1 < len(group) else group[-1].end + 0.15
            if end <= start:
                continue
            parts = []
            for j, other in enumerate(group):
                text = _clean(other.text)
                if j == i:
                    parts.append(
                        r"{\c" + highlight + r"\b1}" + text + r"{\c&H00FFFFFF&\b0}"
                    )
                else:
                    parts.append(text)
            events.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,"
                + " ".join(parts)
            )
    return header + "\n".join(events) + "\n"


def write_ass(words: list[Word], cfg: Config, out_path: str | Path,
              notes: list | None = None, hook_text: str | None = None) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_ass(words, cfg, notes=notes, hook_text=hook_text),
                    encoding="utf-8")
    return path
