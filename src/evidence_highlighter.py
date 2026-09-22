from __future__ import annotations

import html
import re

MAX_HIGHLIGHTS = 2

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on for with from by as is are was were
    be been being it this that these those at not no nor so such than into over
    under about after before between during without within also may can could should
    would will shall must have has had do does did their its his her they them
    patient patients treatment medicinal product products
    """.split()
)


def split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    parts = _SENTENCE_SPLIT.split(stripped)
    return [part.strip() for part in parts if part.strip()]


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 2
    }


def _sentence_score(sentence: str, answer_tokens: set[str]) -> float:
    sentence_tokens = _tokens(sentence)
    if not sentence_tokens or not answer_tokens:
        return 0.0
    overlap = sentence_tokens & answer_tokens
    return len(overlap) / len(sentence_tokens)


def select_highlight_indices(
    sentences: list[str],
    answer: str,
    max_highlights: int = MAX_HIGHLIGHTS,
) -> set[int]:
    answer_tokens = _tokens(answer)
    ranked = sorted(
        (
            (index, _sentence_score(sentence, answer_tokens))
            for index, sentence in enumerate(sentences)
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    selected: set[int] = set()
    for index, score in ranked:
        if score <= 0:
            break
        selected.add(index)
        if len(selected) >= max_highlights:
            break
    return selected


def highlight_text(text: str, answer: str, max_highlights: int = MAX_HIGHLIGHTS) -> str:
    """Wrap up to two answer-relevant sentences in <mark>. Presentation only."""
    sentences = split_sentences(text)
    if not sentences:
        return html.escape(text)

    highlight = select_highlight_indices(sentences, answer, max_highlights)
    rendered = []
    for index, sentence in enumerate(sentences):
        escaped = html.escape(sentence)
        if index in highlight:
            rendered.append(f"<mark>{escaped}</mark>")
        else:
            rendered.append(escaped)
    return " ".join(rendered)
