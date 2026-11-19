#!/usr/bin/env python3
import argparse
from pathlib import Path

from overseas_search_relevance.evaluation.report import generate_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate_report(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
