#!/usr/bin/env python3
"""
Build Cache - CLI entry point for building optimized cache tables.

Supports:
- Full rebuild of cache tables
- Incremental updates (only process new files)
- Conversion from jsonl.gz format
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.cache import PapersCacheBuilder, CitationCacheBuilder


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build optimized cache tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Build papers cache (skip if exists)
  python 1_build_cache.py --papers
  
  # Incremental update (only process new files)
  python 1_build_cache.py --papers --incremental
  
  # Incremental update from jsonl.gz format
  python 1_build_cache.py --papers --incremental --from-jsonl /path/to/new_data/
  
  # Force full rebuild
  python 1_build_cache.py --papers --force
  
  # Build all caches
  python 1_build_cache.py --all --incremental
        """,
    )

    parser.add_argument(
        "--papers", action="store_true", help="Build papers cache table"
    )
    parser.add_argument(
        "--citations", action="store_true", help="Build citations cache table"
    )
    parser.add_argument("--all", action="store_true", help="Build all cache tables")

    parser.add_argument(
        "--incremental",
        "-i",
        action="store_true",
        help="Incremental update mode (only process new files)",
    )
    parser.add_argument(
        "--from-jsonl",
        type=str,
        default=None,
        help="Convert and update from jsonl.gz files in specified directory",
    )

    parser.add_argument("--force", action="store_true", help="Force full rebuild")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--cache-dir", type=str, default="cache", help="Cache directory"
    )

    return parser.parse_args()


def build_papers_cache(args) -> bool:
    """Build the papers optimized cache table."""
    print("\n=== Building Papers Cache Table ===")

    try:
        with PapersCacheBuilder(
            data_dir=args.data_dir, cache_dir=args.cache_dir
        ) as builder:
            if args.incremental:
                # Incremental update mode
                success = builder.incremental_update(from_jsonl_dir=args.from_jsonl)
            else:
                # Full build mode
                builder.check_data_files()
                success = builder.create_optimized_papers_table(
                    force_recreate=args.force
                )

            if success:
                builder.show_table_info()
                print("[OK] Papers cache build complete")
                return True

    except Exception as e:
        print(f"[ERROR] Papers cache build failed: {e}")
        import traceback

        traceback.print_exc()

    return False


def build_citation_cache(args) -> bool:
    """Build the citation optimized cache table."""
    print("\n=== Building Citation Cache Table ===")

    citation_data_dir = os.path.join(args.data_dir, "citations")

    try:
        with CitationCacheBuilder(
            data_dir=citation_data_dir, cache_dir=args.cache_dir
        ) as builder:
            # TODO: Add incremental support for citations
            if builder.create_optimized_cache_table(force_recreate=args.force):
                builder.show_cache_info()
                print("[OK] Citation cache build complete")
                return True

    except Exception as e:
        print(f"[ERROR] Citation cache build failed: {e}")
        import traceback

        traceback.print_exc()

    return False


def main():
    args = parse_arguments()

    if not any([args.papers, args.citations, args.all]):
        print("Please specify cache type to build. Use --help for options.")
        return 1

    success = True

    if args.all or args.papers:
        if not build_papers_cache(args):
            success = False

    if args.all or args.citations:
        if not build_citation_cache(args):
            success = False

    if success:
        print("\n[OK] Cache build complete!")
    else:
        print("\n[WARN] Some cache builds failed")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
