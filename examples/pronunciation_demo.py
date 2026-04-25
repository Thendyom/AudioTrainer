"""Analyze speech prosody and optionally compare against a reference file."""

from __future__ import annotations

import argparse

from audiotrainer.api.service import analyze_speech_file, compare_speech_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--reference")
    args = parser.parse_args()

    report, feedback = analyze_speech_file(args.path)
    print(report.model_dump_json(indent=2))
    if args.reference:
        print(compare_speech_files(args.path, args.reference).model_dump_json(indent=2))
    for item in feedback:
        print(f"[{item.severity}] {item.message} {item.suggestion}")


if __name__ == "__main__":
    main()
