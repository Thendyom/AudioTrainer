"""Speech prosody, pronunciation comparison, and voice profiling."""

from audiotrainer.speech.pronunciation import compare_reference_speech
from audiotrainer.speech.prosody import analyze_prosody, detect_pause_patterns, score_monotony
from audiotrainer.speech.voice_profile import classify_voice_type, estimate_vocal_range

__all__ = [
    "analyze_prosody",
    "classify_voice_type",
    "compare_reference_speech",
    "detect_pause_patterns",
    "estimate_vocal_range",
    "score_monotony",
]
