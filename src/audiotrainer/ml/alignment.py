"""Ordered word alignment for local transcript comparison."""

from __future__ import annotations

from difflib import SequenceMatcher
import re

from audiotrainer.api.schemas import TranscriptAlignment, TranscriptReport


def align_transcripts(user: TranscriptReport, reference: TranscriptReport) -> TranscriptAlignment:
    user_words = [_normalize(item.word) for item in user.words]
    reference_words = [_normalize(item.word) for item in reference.words]
    matcher = SequenceMatcher(a=reference_words, b=user_words, autojunk=False)
    matched = 0
    omitted: list[str] = []
    substituted: list[tuple[str, str]] = []
    added: list[str] = []
    for tag, ref_start, ref_end, user_start, user_end in matcher.get_opcodes():
        ref_chunk = reference_words[ref_start:ref_end]
        user_chunk = user_words[user_start:user_end]
        if tag == "equal":
            matched += len(ref_chunk)
        elif tag == "delete":
            omitted.extend(ref_chunk)
        elif tag == "insert":
            added.extend(user_chunk)
        else:
            paired = min(len(ref_chunk), len(user_chunk))
            substituted.extend(zip(ref_chunk[:paired], user_chunk[:paired], strict=True))
            omitted.extend(ref_chunk[paired:])
            added.extend(user_chunk[paired:])
    denominator = max(1, len(reference_words), len(user_words))
    timing_difference = user.duration - reference.duration if user.duration and reference.duration else None
    return TranscriptAlignment(
        matched_words=matched,
        omitted_words=[word for word in omitted if word],
        substituted_words=[pair for pair in substituted if pair[0] or pair[1]],
        added_words=[word for word in added if word],
        match_score=matched / denominator,
        delivery_timing_difference=timing_difference,
    )


def _normalize(value: str) -> str:
    return re.sub(r"[^\w']+", "", value.lower(), flags=re.UNICODE)
