#!/usr/bin/env python3
"""
Dataset Extractor - CLI entry point for extracting dataset names from citations.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.extractors import DatasetExtractor


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract dataset names from citation contexts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python extract_datasets.py --input topic_model_contexts.json --output datasets.json
  python extract_datasets.py --input contexts.json --topic "topic modeling"
        """,
    )

    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input JSON file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="extracted_datasets.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--topic", "-t", type=str, default="topic modeling", help="Research topic"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        default=False,
        help="Disable topic filtering",
    )
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--base-url", type=str, default=None, help="LLM API base URL")

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 60)
    print("Dataset Extractor")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Research topic: {args.topic}")
    print("=" * 60)

    extractor = DatasetExtractor(model_name=args.model, base_url=args.base_url)

    success = extractor.extract_from_citation_contexts(
        json_file_path=args.input,
        output_file=args.output,
        filter_topic=not args.no_filter,
        topic=args.topic,
    )

    if success:
        print(f"\n[OK] Extraction complete! Results saved to: {args.output}")
        return 0
    else:
        print("\n[ERROR] Extraction failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
