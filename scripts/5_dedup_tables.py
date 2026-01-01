#!/usr/bin/env python3
"""
Dedup Merged Tables - CLI entry point for deduplicating merged analysis tables.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.dedup_utils import walk_and_process


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Deduplicate citation links in merged analysis tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python dedup_tables.py --dirs ./output
  python dedup_tables.py --dirs ./output1 ./output2 --dry-run
        """,
    )

    parser.add_argument(
        "--dirs", type=str, nargs="+", required=True, help="Directories to process"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview only, do not write to files"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 60)
    print("Dedup Merged Tables")
    print("=" * 60)
    print(f"Directories: {args.dirs}")
    print(f"Mode: {'Preview' if args.dry_run else 'Processing'}")
    print("=" * 60)

    walk_and_process(args.dirs, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
