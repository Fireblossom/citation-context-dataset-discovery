"""
JSONL Converter - Converts jsonl.gz files to parquet format.

Based on 0_gz2parquet.py with enhancements for incremental processing.
"""

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def find_input_files(
    input_dir: str, patterns: Optional[List[str]] = None
) -> List[Path]:
    """Find all jsonl/jsonl.gz files in directory recursively."""
    base = Path(input_dir)
    if not base.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if patterns is None:
        patterns = ["*.jsonl.gz", "*.jsonl", "*.ndjson.gz", "*.ndjson", "*.gz"]

    files: List[Path] = []
    for pat in patterns:
        files.extend(base.rglob(pat))

    # De-duplicate while preserving order
    seen = set()
    unique_files: List[Path] = []
    for p in sorted(files):
        if p not in seen:
            seen.add(p)
            unique_files.append(p)
    return unique_files


def derive_output_path(input_path: Path, output_dir: Path) -> Path:
    """Derive parquet output path from input path."""
    stem = input_path.name
    for suffix in [".jsonl.gz", ".ndjson.gz", ".jsonl", ".ndjson", ".gz"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return output_dir / f"{stem}.parquet"


def convert_one_file(
    input_file: Path,
    output_file: Path,
    chunksize: int = 200_000,
    force_override: bool = False,
) -> Tuple[bool, str, int]:
    """
    Convert one file and return (success, message, row_count)
    """
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.exists() and not force_override:
            return True, "File already exists, skipped", 0
        elif output_file.exists() and force_override:
            output_file.unlink()

        compression: Optional[str]
        if (
            input_file.suffix == ".gz"
            or input_file.name.endswith(".jsonl.gz")
            or input_file.name.endswith(".ndjson.gz")
        ):
            compression = "gzip"
        else:
            compression = None

        writer: Optional[pq.ParquetWriter] = None
        first_columns: Optional[List[str]] = None
        first_schema: Optional[pa.Schema] = None
        total_rows = 0

        try:
            chunk_iter = pd.read_json(
                input_file,
                lines=True,
                compression=compression,
                chunksize=chunksize,
                dtype_backend="pyarrow",
            )
            for idx, df in enumerate(chunk_iter):
                if df.empty:
                    continue
                total_rows += len(df)

                if first_columns is None:
                    first_columns = list(df.columns)
                else:
                    missing = [c for c in first_columns if c not in df.columns]
                    for col in missing:
                        df[col] = pd.Series([pd.NA] * len(df), dtype="object")
                    df = df[first_columns]

                table = pa.Table.from_pandas(df, preserve_index=False)

                if writer is None:
                    schema = table.schema
                    nullable_fields = []
                    for field in schema:
                        if not field.nullable:
                            nullable_field = pa.field(
                                field.name,
                                field.type,
                                nullable=True,
                                metadata=field.metadata,
                            )
                            nullable_fields.append(nullable_field)
                        else:
                            nullable_fields.append(field)

                    first_schema = pa.schema(nullable_fields)
                    writer = pq.ParquetWriter(
                        output_file, first_schema, compression="zstd"
                    )

                if table.schema != first_schema:
                    table = table.cast(first_schema)

                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

        return True, f"Converted {total_rows} rows", total_rows

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return False, error_msg, 0


def parallel_convert(
    input_files: List[Path],
    output_dir: Path,
    workers: int = 8,
    chunksize: int = 200_000,
    force_override: bool = False,
) -> Tuple[int, int, List[Tuple[Path, int]]]:
    """
    Convert files in parallel.

    Returns:
        (success_count, error_count, list of (output_path, row_count))
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    file_mapping = {}

    print(f"Starting conversion with {workers} workers...")

    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        for f in input_files:
            out_path = derive_output_path(f, output_dir)
            future = ex.submit(convert_one_file, f, out_path, chunksize, force_override)
            tasks.append(future)
            file_mapping[future] = (f, out_path)

        success_count = 0
        error_count = 0
        converted_files = []

        for fut in tqdm(
            as_completed(tasks), total=len(tasks), desc="Converting", unit="file"
        ):
            input_path, output_path = file_mapping[fut]
            try:
                success, message, row_count = fut.result()
                if success:
                    success_count += 1
                    if row_count > 0:
                        converted_files.append((output_path, row_count))
                else:
                    error_count += 1
                    print(f"\n[ERROR] {input_path.name}: {message}")
            except Exception as e:
                error_count += 1
                print(f"\n[ERROR] {input_path.name}: {str(e)}")

        return success_count, error_count, converted_files


def convert_jsonl_to_parquet(
    input_dir: str,
    output_dir: str,
    workers: int = 8,
    chunksize: int = 200_000,
    force_override: bool = False,
    patterns: Optional[List[str]] = None,
) -> List[Tuple[Path, int]]:
    """
    Main function to convert JSONL files to Parquet.

    Args:
        input_dir: Directory containing jsonl.gz files
        output_dir: Directory to write parquet files
        workers: Number of parallel workers
        chunksize: Rows per chunk when reading
        force_override: Force override existing files
        patterns: Custom glob patterns

    Returns:
        List of (output_path, row_count) for newly converted files
    """
    input_files = find_input_files(input_dir, patterns)

    if not input_files:
        print(f"No JSONL files found in: {input_dir}")
        return []

    print(f"Found {len(input_files)} files in {input_dir}")

    success, errors, converted = parallel_convert(
        input_files, Path(output_dir), workers, chunksize, force_override
    )

    print(f"\n[OK] Converted: {success}, [ERROR] Errors: {errors}")

    return converted
