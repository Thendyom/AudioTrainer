"""Record short microphone chunks and print detected pitch."""

from __future__ import annotations

import argparse
import time

import numpy as np

from audiotrainer.audio.device import record_audio
from audiotrainer.pitch import detect_pitch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=44_100)
    args = parser.parse_args()

    while True:
        audio = record_audio(args.duration, args.sample_rate)
        track = detect_pitch(audio, args.sample_rate)
        frequencies = [frame.frequency_hz for frame in track.frames if frame.frequency_hz is not None]
        notes = [frame.note for frame in track.frames if frame.note is not None]
        if frequencies and notes:
            print(f"{np.median(frequencies):7.1f} Hz  {max(set(notes), key=notes.count)}")
        else:
            print("unvoiced")
        time.sleep(0.05)


if __name__ == "__main__":
    main()
