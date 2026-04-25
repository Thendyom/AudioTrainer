"""Detect pitch from an audio file."""

from __future__ import annotations

import argparse

from audiotrainer.api.service import analyze_pitch_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    track, score, feedback = analyze_pitch_file(args.path, args.target)
    print(f"frames={len(track.frames)}")
    print(score.model_dump_json(indent=2))
    for item in feedback:
        print(f"[{item.severity}] {item.message} {item.suggestion}")


if __name__ == "__main__":
    main()
