from pathlib import Path

import numpy as np
import pytest

from audiotrainer.audio.framing import frame_audio
from audiotrainer.audio.io import load_audio, write_audio
from audiotrainer.audio.preprocessing import normalize_peak, remove_dc, to_mono


def test_to_mono_averages_channels() -> None:
    stereo = np.array([[1.0, -1.0], [0.5, 0.25], [-0.5, 0.75]])
    mono = to_mono(stereo)
    assert mono.tolist() == [0.0, 0.375, 0.125]


def test_frame_audio_pads_short_signal() -> None:
    frames = frame_audio(np.array([1.0, -1.0]), frame_length=4, hop_length=2, center=False)
    assert frames.shape == (1, 4)
    assert frames[0].tolist() == [1.0, -1.0, 0.0, 0.0]


def test_remove_dc_and_normalize_peak_preserve_silence() -> None:
    assert np.allclose(remove_dc(np.array([2.0, 4.0, 6.0])), [-2.0, 0.0, 2.0])
    assert np.allclose(normalize_peak(np.zeros(4)), np.zeros(4))


def test_load_audio_roundtrip_converts_to_mono(tmp_path: Path) -> None:
    sr = 8_000
    stereo = np.array([[0.5, -0.5], [0.25, 0.75], [-0.25, 0.25]], dtype=np.float64)
    path = write_audio(tmp_path / "sample.wav", stereo, sr)

    audio, loaded_sr = load_audio(path)

    assert loaded_sr == sr
    assert audio.shape == (3,)
    assert audio == pytest.approx([0.0, 0.5, 0.0], abs=1e-4)
