"""Human-readable feedback generation."""

from __future__ import annotations

from audiotrainer.api.schemas import FeedbackItem, PitchScore, ProsodyReport, VocalRange, VoiceTypeEstimate


def generate_pitch_feedback(score: PitchScore) -> list[FeedbackItem]:
    """Generate feedback from a pitch score."""

    feedback: list[FeedbackItem] = []
    if score.voiced_frame_count == 0:
        return [
            FeedbackItem(
                severity="critical",
                category="pitch",
                message="No stable pitch was detected.",
                suggestion="Use a clearer sustained tone, move closer to the microphone, or reduce background noise.",
            )
        ]

    if score.mean_abs_cents is not None:
        if score.mean_abs_cents > 35:
            feedback.append(
                FeedbackItem(
                    severity="critical",
                    category="pitch",
                    message=f"You are averaging {score.mean_abs_cents:.0f} cents away from the target.",
                    suggestion="Slow down and sustain each note while checking against a reference pitch.",
                )
            )
        elif score.mean_abs_cents > 15:
            feedback.append(
                FeedbackItem(
                    severity="warning",
                    category="pitch",
                    message=f"You are consistently around {score.mean_abs_cents:.0f} cents off center.",
                    suggestion="Practice small pitch corrections and listen for beating against the reference tone.",
                )
            )
        else:
            feedback.append(
                FeedbackItem(
                    severity="info",
                    category="pitch",
                    message="Pitch accuracy is close to the target.",
                    suggestion="Keep the same setup and work on smooth starts and releases.",
                )
            )

    if score.stability < 0.45:
        feedback.append(
            FeedbackItem(
                severity="warning",
                category="pitch",
                message="Sustained notes are unstable.",
                suggestion="Hold notes with steady breath or bow pressure and avoid correcting too abruptly.",
            )
        )
    return feedback


def generate_speech_feedback(report: ProsodyReport) -> list[FeedbackItem]:
    """Generate feedback from a prosody report."""

    feedback: list[FeedbackItem] = []
    if report.monotony_score > 0.7:
        feedback.append(
            FeedbackItem(
                severity="warning",
                category="pronunciation",
                message="Your speaking pitch range is narrow, so the delivery may sound monotone.",
                suggestion="Emphasize key words with gentle pitch movement and phrase-level contrast.",
            )
        )
    if report.estimated_speech_rate is not None:
        if report.estimated_speech_rate < 0.7:
            feedback.append(
                FeedbackItem(
                    severity="info",
                    category="timing",
                    message="The speaking-rate proxy is low.",
                    suggestion="Try a slightly more connected delivery while keeping pauses intentional.",
                )
            )
        elif report.estimated_speech_rate > 5.0:
            feedback.append(
                FeedbackItem(
                    severity="warning",
                    category="timing",
                    message="The speaking-rate proxy is high.",
                    suggestion="Add short pauses after important phrases and reduce rushed attacks.",
                )
            )
    if report.pause_count > max(3, report.duration):
        feedback.append(
            FeedbackItem(
                severity="warning",
                category="timing",
                message="Pauses are frequent for the recording duration.",
                suggestion="Group words into longer phrases and pause at sentence boundaries.",
            )
        )
    if not feedback:
        feedback.append(
            FeedbackItem(
                severity="info",
                category="pronunciation",
                message="Prosody measurements look balanced for this recording.",
                suggestion="For pronunciation-specific scoring, compare against a clean reference recording.",
            )
        )
    return feedback


def generate_voice_feedback(profile: VocalRange | VoiceTypeEstimate) -> list[FeedbackItem]:
    """Generate feedback from a voice range or voice type estimate."""

    if isinstance(profile, VocalRange):
        if profile.confidence < 0.4:
            return [
                FeedbackItem(
                    severity="warning",
                    category="voice",
                    message="The vocal range estimate has low confidence.",
                    suggestion="Record a longer, comfortable scale with steady vowels and moderate volume.",
                )
            ]
        return [
            FeedbackItem(
                severity="info",
                category="voice",
                message=f"Stable range estimate: {profile.lowest_note} to {profile.highest_note}.",
                suggestion="Treat this as a practical range estimate, not a fixed voice-type diagnosis.",
            )
        ]

    severity = "warning" if profile.confidence < 0.5 else "info"
    return [
        FeedbackItem(
            severity=severity,
            category="voice",
            message=profile.explanation,
            suggestion="Confirm voice type over multiple recordings and with comfortable low, middle, and high notes.",
        )
    ]
