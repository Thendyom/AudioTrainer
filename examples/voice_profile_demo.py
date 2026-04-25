"""Estimate vocal range and rough voice type from a recording."""

from __future__ import annotations

import argparse

from audiotrainer.api.service import analyze_voice_profile_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    vocal_range, estimate, feedback = analyze_voice_profile_file(args.path)
    print(vocal_range.model_dump_json(indent=2))
    print(estimate.model_dump_json(indent=2))
    for item in feedback:
        print(f"[{item.severity}] {item.message} {item.suggestion}")


if __name__ == "__main__":
    main()
