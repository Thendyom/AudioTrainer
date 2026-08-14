"""Beat quantization and editable monophonic score documents."""

from __future__ import annotations

from statistics import median

from audiotrainer.api.schemas import NoteEvent, ScoreDocument, ScoreEvent


def create_score_document(
    events: list[NoteEvent],
    *,
    bpm: int = 120,
    time_signature: str = "4/4",
    quantization: int = 4,
) -> ScoreDocument:
    """Quantize detected notes, preserve gaps as rests, and split measures."""

    beats_per_measure, beat_unit = parse_time_signature(time_signature)
    if bpm <= 0 or bpm > 400:
        raise ValueError("bpm must be between 1 and 400")
    if quantization < 1 or quantization > 32:
        raise ValueError("quantization must be between 1 and 32")
    grid = 1.0 / quantization
    cursor = 0.0
    raw: list[ScoreEvent] = []
    for event in sorted(events, key=lambda item: item.start_time):
        start = max(0.0, _quantize(event.start_time * bpm / 60.0, grid))
        end = max(start + grid, _quantize(event.end_time * bpm / 60.0, grid))
        start = max(cursor, start)
        if end <= start:
            end = start + grid
        if start > cursor + 1e-7:
            raw.append(ScoreEvent(kind="rest", start_beat=cursor, duration_beats=start - cursor))
        raw.append(
            ScoreEvent(
                kind="note",
                start_beat=start,
                duration_beats=end - start,
                note=event.note,
                frequency_hz=event.frequency_hz,
                confidence=event.confidence,
            )
        )
        cursor = end

    measure_length = beats_per_measure * 4.0 / beat_unit
    split = _split_at_measures(raw, measure_length)
    return ScoreDocument(
        bpm=bpm,
        beats_per_measure=beats_per_measure,
        beat_unit=beat_unit,
        quantization=quantization,
        events=split,
        suggested_bpm=suggest_tempo(events),
    )


def parse_time_signature(value: str) -> tuple[int, int]:
    """Parse a conventional numerator/denominator time signature."""

    try:
        numerator_text, denominator_text = value.strip().split("/", maxsplit=1)
        numerator, denominator = int(numerator_text), int(denominator_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("time signature must look like 4/4") from exc
    if numerator <= 0 or numerator > 32 or denominator not in {1, 2, 4, 8, 16, 32}:
        raise ValueError("unsupported time signature")
    return numerator, denominator


def suggest_tempo(events: list[NoteEvent]) -> int | None:
    """Return a conservative onset-based tempo suggestion."""

    if len(events) < 3:
        return None
    intervals = [
        right.start_time - left.start_time
        for left, right in zip(events, events[1:])
        if right.start_time > left.start_time
    ]
    if not intervals:
        return None
    bpm = 60.0 / median(intervals)
    while bpm < 60:
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    return int(round(bpm))


def score_document_to_note_events(document: ScoreDocument) -> list[NoteEvent]:
    """Convert editable score notes back to time-domain note events."""

    seconds_per_beat = 60.0 / document.bpm
    output: list[NoteEvent] = []
    pending: NoteEvent | None = None
    for event in document.events:
        if event.kind == "rest" or event.note is None:
            if pending:
                output.append(pending)
                pending = None
            continue
        start = event.start_beat * seconds_per_beat
        end = (event.start_beat + event.duration_beats) * seconds_per_beat
        candidate = NoteEvent(
            start_time=start,
            end_time=end,
            frequency_hz=event.frequency_hz or 0.0,
            note=event.note,
            confidence=event.confidence or 0.0,
        )
        if (
            pending
            and event.tie_stop
            and pending.note == candidate.note
            and abs(pending.end_time - candidate.start_time) < 1e-6
        ):
            pending = pending.model_copy(update={"end_time": candidate.end_time})
        else:
            if pending:
                output.append(pending)
            pending = candidate
    if pending:
        output.append(pending)
    return output


def _quantize(value: float, grid: float) -> float:
    return round(value / grid) * grid


def _split_at_measures(events: list[ScoreEvent], measure_length: float) -> list[ScoreEvent]:
    output: list[ScoreEvent] = []
    for event in events:
        position = event.start_beat
        remaining = event.duration_beats
        fragment_index = 0
        while remaining > 1e-7:
            measure_index = int((position + 1e-8) // measure_length)
            boundary = (measure_index + 1) * measure_length
            duration = min(remaining, boundary - position)
            continues = remaining - duration > 1e-7
            output.append(
                event.model_copy(
                    update={
                        "start_beat": position,
                        "duration_beats": duration,
                        "measure": measure_index + 1,
                        "tie_stop": event.kind == "note" and fragment_index > 0,
                        "tie_start": event.kind == "note" and continues,
                    }
                )
            )
            position += duration
            remaining -= duration
            fragment_index += 1
    return output
