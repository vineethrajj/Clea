"""Phase 3: informational note-card overlays.

From the Whisper transcripts, pick 2-4 key factual sentences, optionally
compress them with the local LLM into short card-sized lines, anchor each to
the moment its sentence is spoken on the edit timeline, and hand them to the
ASS renderer as a visually distinct "card" style (boxed, top of frame).

Works fully offline: when Ollama isn't running, a heuristic picks and trims
the sentences instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import llm
from .assemble import EditPlan
from .config import Config
from .transcribe import Word

_KEYWORDS = re.compile(
    r"\b(help|helps|prevent|prevents|because|means|should|most|cause|causes|"
    r"reduce|reduces|improve|improves|avoid|remember|important|key|always|"
    r"never|usually|typically|recommend|recommended)\b", re.I)


@dataclass
class Note:
    text: str
    start: float
    end: float


@dataclass
class _Sentence:
    text: str
    clip: str
    src_start: float
    src_end: float
    score: float = 0.0


def sentences_from_transcripts(words_by_clip: dict[str, list[Word]]) -> list[_Sentence]:
    sentences: list[_Sentence] = []
    for clip, words in words_by_clip.items():
        current: list[Word] = []
        for w in words:
            current.append(w)
            if w.text.endswith((".", "!", "?")):
                sentences.append(_Sentence(
                    " ".join(x.text for x in current).strip(),
                    clip, current[0].start, current[-1].end))
                current = []
        if len(current) >= 4:  # trailing clause without punctuation
            sentences.append(_Sentence(
                " ".join(x.text for x in current).strip(),
                clip, current[0].start, current[-1].end))
    return sentences


def _score(s: _Sentence) -> float:
    n = len(s.text.split())
    score = 1.0 - abs(n - 12) / 20.0          # sweet spot ~12 words
    if any(ch.isdigit() for ch in s.text):
        score += 0.5                           # facts with numbers
    score += 0.25 * len(_KEYWORDS.findall(s.text))
    return score


def pick_key_sentences(sentences: list[_Sentence], max_notes: int = 4) -> list[_Sentence]:
    for s in sentences:
        s.score = _score(s)
    ranked = sorted(sentences, key=lambda s: s.score, reverse=True)[:max_notes]
    # Chronological order (per clip then time) reads better on screen.
    return sorted(ranked, key=lambda s: (s.clip, s.src_start))


def _compress_with_llm(texts: list[str], cfg: Config) -> list[str]:
    prompt = (
        "Rewrite each sentence as a short on-screen note card of at most 8 "
        "words. Keep it factual, no hype, no emojis. Return ONLY a JSON "
        'object like {"notes": ["...", "..."]} with one entry per input, '
        "same order.\n\nSentences:\n"
        + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    )
    reply = llm.generate(prompt, system="You write concise educational video overlays.",
                         cfg=cfg, json_mode=True)
    notes = llm.parse_json_response(reply).get("notes", [])
    if len(notes) != len(texts) or not all(isinstance(n, str) and n.strip() for n in notes):
        raise llm.LLMError("unexpected notes shape")
    return [n.strip() for n in notes]


def _trim(text: str, limit: int = 64) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "…"


def build_notes(
    words_by_clip: dict[str, list[Word]],
    plan: EditPlan,
    cfg: Config,
    max_notes: int = 4,
    use_llm: bool = True,
) -> list[Note]:
    sentences = pick_key_sentences(sentences_from_transcripts(words_by_clip), max_notes)
    if not sentences:
        return []

    texts = [s.text for s in sentences]
    if use_llm and llm.ollama_available(cfg):
        try:
            texts = _compress_with_llm(texts, cfg)
            print("      notes: compressed via local LLM")
        except llm.LLMError as exc:
            print(f"      notes: LLM compression unavailable ({exc}); using trimmed sentences")
            texts = [_trim(t) for t in texts]
    else:
        texts = [_trim(t) for t in texts]

    total = plan.total_duration
    dur = float(cfg.notes["display_seconds"])

    # Anchor each note to the first timeline segment that shows (part of) the
    # moment its sentence is spoken; unanchored notes fill free slots later.
    notes: list[Note] = []
    unanchored: list[str] = []
    for s, text in zip(sentences, texts):
        anchor = None
        tl_pos = 0.0
        for e in plan.entries:
            if e.clip == s.clip and s.src_start < e.src_start + e.timeline_duration \
                    and s.src_end > e.src_start:
                anchor = tl_pos + max(0.0, s.src_start - e.src_start)
                break
            tl_pos += e.timeline_duration
        if anchor is None:
            unanchored.append(text)
        else:
            notes.append(Note(text, anchor, min(anchor + dur, total)))

    for i, text in enumerate(unanchored):
        start = (i + 1) * total / (len(unanchored) + 1)
        notes.append(Note(text, start, min(start + dur, total)))

    # De-overlap: one card on screen at a time.
    notes.sort(key=lambda n: n.start)
    for i in range(1, len(notes)):
        prev = notes[i - 1]
        if notes[i].start < prev.end + 0.3:
            shift = prev.end + 0.3 - notes[i].start
            notes[i].start += shift
            notes[i].end = min(notes[i].start + dur, total)
    return [n for n in notes if n.end - n.start > 1.0 and n.start < total - 1.0]
