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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache import PapersCacheBuilder, CitationCacheBuilder, DuckDBSettings

# S2 dataset sub-directories needed by each cache
PAPERS_SUBDIRS = ["papers", "abstracts", "paper-ids"]
CITATIONS_SUBDIRS = ["citations"]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build optimized cache tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Convert raw S2 jsonl.gz downloads to parquet, then build everything
  python 1_build_cache.py --all --convert-from /path/to/s2_jsonl --data-dir data

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

DuckDB memory, threads and spill directory are detected from the Slurm
allocation / cgroup / host; override them with the options below or the
CCDD_DUCKDB_MEMORY_LIMIT, CCDD_DUCKDB_THREADS, CCDD_DUCKDB_TEMP_DIR and
CCDD_DUCKDB_MAX_TEMP_SIZE environment variables.
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
    parser.add_argument(
        "--convert-from",
        type=str,
        default=None,
        help="Before a full build, convert jsonl.gz files from this directory "
        "(sub-directories papers/, abstracts/, paper-ids/, citations/) to "
        "parquet under --data-dir; already converted files are skipped",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for jsonl.gz conversion (default: 8)",
    )

    parser.add_argument("--force", action="store_true", help="Force full rebuild")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--cache-dir", type=str, default="cache", help="Cache directory"
    )

    resources = parser.add_argument_group("DuckDB resources")
    resources.add_argument(
        "--memory-limit",
        type=str,
        default=None,
        help="DuckDB memory_limit, e.g. 1200GB (default: 80%% of available memory)",
    )
    resources.add_argument(
        "--threads", type=int, default=None, help="DuckDB threads (default: allocated CPUs)"
    )
    resources.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="DuckDB spill directory (default: <cache-dir>/duckdb_tmp)",
    )
    resources.add_argument(
        "--max-temp-size",
        type=str,
        default=None,
        help="DuckDB max_temp_directory_size, e.g. 5TB",
    )

    return parser.parse_args()


def convert_raw_data(args, subdirs) -> bool:
    """Convert jsonl.gz sub-directories of --convert-from into parquet."""
    from pathlib import Path
    from src.utils.jsonl_converter import find_input_files, parallel_convert

    print("\n=== Converting jsonl.gz to parquet ===")
    ok = True
    for subdir in subdirs:
        source = os.path.join(args.convert_from, subdir)
        if not os.path.isdir(source):
            print(f"[WARN] {source} not found, skipping")
            continue
        input_files = find_input_files(source)
        print(f"{subdir}: {len(input_files)} files")
        _, errors, _ = parallel_convert(
            input_files, Path(args.data_dir) / subdir, workers=args.workers
        )
        if errors:
            ok = False
            print(f"[ERROR] {errors} {subdir} file(s) failed to convert")
    return ok


def build_papers_cache(args, settings) -> bool:
    """Build the papers optimized cache table."""
    print("\n=== Building Papers Cache Table ===")

    try:
        with PapersCacheBuilder(
            data_dir=args.data_dir, cache_dir=args.cache_dir, settings=settings
        ) as builder:
            if args.incremental:
                # Incremental update mode
                success = builder.incremental_update(
                    from_jsonl_dir=args.from_jsonl, workers=args.workers
                )
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


def build_citation_cache(args, settings) -> bool:
    """Build the citation optimized cache table."""
    print("\n=== Building Citation Cache Table ===")

    citation_data_dir = os.path.join(args.data_dir, "citations")

    try:
        with CitationCacheBuilder(
            data_dir=citation_data_dir, cache_dir=args.cache_dir, settings=settings
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

    settings = DuckDBSettings(
        memory_limit=args.memory_limit,
        threads=args.threads,
        temp_directory=args.temp_dir,
        max_temp_directory_size=args.max_temp_size,
    )
    build_papers = args.all or args.papers
    build_citations = args.all or args.citations

    if args.convert_from:
        subdirs = (PAPERS_SUBDIRS if build_papers else []) + (
            CITATIONS_SUBDIRS if build_citations else []
        )
        if not convert_raw_data(args, subdirs):
            print(
                "\n[ERROR] Some files could not be converted (see above); "
                "fix or remove them and rerun"
            )
            return 1

    success = True

    if build_papers:
        if not build_papers_cache(args, settings):
            success = False

    if build_citations:
        if not build_citation_cache(args, settings):
            success = False

    if success:
        print("\n[OK] Cache build complete!")
    else:
        print("\n[WARN] Some cache builds failed")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
