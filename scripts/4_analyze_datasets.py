#!/usr/bin/env python3
"""
Analyze Datasets - CLI entry point for analyzing extracted datasets by citation count.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.processors.dataset_analyzer import analyze_datasets


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze extracted datasets and rank by citation count",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python analyze_datasets.py --extraction datasets.json --contexts contexts.json
        """,
    )

    parser.add_argument(
        "--extraction",
        "-e",
        type=str,
        required=True,
        help="Dataset extraction results file",
    )
    parser.add_argument(
        "--contexts", "-c", type=str, required=True, help="Citation contexts file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="analyzed_datasets.tsv", help="Output file"
    )
    parser.add_argument(
        "--topic", "-t", type=str, default="topic modeling", help="Research topic"
    )
    parser.add_argument(
        "--validation",
        type=str,
        choices=["off", "basic", "smart", "strict", "llm"],
        default="smart",
        help="Validation mode",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 60)
    print("Dataset Analysis by Citation Count")
    print("=" * 60)

    success = analyze_datasets(
        extraction_file=args.extraction,
        contexts_file=args.contexts,
        output_file=args.output,
        topic=args.topic,
        validation_mode=args.validation,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
