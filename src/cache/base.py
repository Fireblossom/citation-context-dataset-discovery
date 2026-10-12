"""
Shared DuckDB plumbing for the cache builders.

Resource settings (memory, threads, spill directory) are resolved in this order:
explicit argument > environment variable > detected from the Slurm allocation,
the process cgroup, or the host.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import duckdb

ENV_MEMORY_LIMIT = "CCDD_DUCKDB_MEMORY_LIMIT"
ENV_THREADS = "CCDD_DUCKDB_THREADS"
ENV_TEMP_DIR = "CCDD_DUCKDB_TEMP_DIR"
ENV_MAX_TEMP_SIZE = "CCDD_DUCKDB_MAX_TEMP_SIZE"

# DuckDB's memory_limit does not cover every allocation (Python objects, pandas
# conversions, some operator state), and exceeding a Slurm/cgroup limit gets the
# process OOM-killed, so only hand DuckDB part of what is available.
MEMORY_FRACTION = 0.8


def _cgroup_memory_limit() -> Optional[int]:
    """Memory limit of this process's cgroup (v2 or v1), if any."""
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        hierarchy, controllers, path = line.split(":", 2)
        if hierarchy == "0":
            candidate = Path("/sys/fs/cgroup") / path.lstrip("/") / "memory.max"
        elif "memory" in controllers.split(","):
            candidate = (
                Path("/sys/fs/cgroup/memory") / path.lstrip("/") / "memory.limit_in_bytes"
            )
        else:
            continue
        try:
            value = candidate.read_text().strip()
        except OSError:
            continue
        # cgroup v1 reports "unlimited" as a huge number
        if value.isdigit() and int(value) < 1 << 60:
            return int(value)
    return None


def available_memory_bytes() -> Optional[int]:
    """Smallest of the Slurm allocation, the cgroup limit and physical RAM."""
    candidates = []

    mem_per_node = os.environ.get("SLURM_MEM_PER_NODE", "")
    mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU", "")
    cpus_on_node = os.environ.get("SLURM_CPUS_ON_NODE", "")
    if mem_per_node.isdigit():
        candidates.append(int(mem_per_node) * 1024**2)
    elif mem_per_cpu.isdigit() and cpus_on_node.isdigit():
        candidates.append(int(mem_per_cpu) * int(cpus_on_node) * 1024**2)

    cgroup_limit = _cgroup_memory_limit()
    if cgroup_limit:
        candidates.append(cgroup_limit)

    try:
        candidates.append(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        pass

    return min(candidates) if candidates else None


def available_cpus() -> int:
    cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if cpus_per_task.isdigit():
        return int(cpus_per_task)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


@dataclass
class DuckDBSettings:
    """DuckDB resource settings; None means "detect / use the default"."""

    memory_limit: Optional[str] = None
    threads: Optional[int] = None
    temp_directory: Optional[str] = None
    max_temp_directory_size: Optional[str] = None
    progress_bar: Optional[bool] = None

    def resolved(self, default_temp_dir: Path) -> "DuckDBSettings":
        memory_limit = self.memory_limit or os.environ.get(ENV_MEMORY_LIMIT)
        if not memory_limit:
            available = available_memory_bytes()
            if available:
                memory_limit = f"{max(1, int(available * MEMORY_FRACTION / 1e9))}GB"

        threads = self.threads or int(os.environ.get(ENV_THREADS, 0)) or available_cpus()
        temp_directory = (
            self.temp_directory or os.environ.get(ENV_TEMP_DIR) or str(default_temp_dir)
        )
        max_temp_size = self.max_temp_directory_size or os.environ.get(ENV_MAX_TEMP_SIZE)
        # A progress bar only makes sense on a terminal; in batch logs it is noise.
        progress_bar = (
            self.progress_bar if self.progress_bar is not None else sys.stdout.isatty()
        )

        return DuckDBSettings(
            memory_limit=memory_limit,
            threads=threads,
            temp_directory=temp_directory,
            max_temp_directory_size=max_temp_size,
            progress_bar=progress_bar,
        )


def list_parquet_files(directory) -> List[Path]:
    """Parquet files in a directory, sorted by name.

    S2 release files start with a YYYYMMDD_HHMMSS timestamp, so name order is
    chronological order.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.parquet"))


def sql_literal(value) -> str:
    """Quote a string (e.g. a file path) as a SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


# Base name of the source file of a row read with read_parquet(..., filename = true).
# S2 file names start with a timestamp, so comparing them orders versions in time.
FILE_BASENAME = "regexp_extract(filename, '[^/]+$')"


def deletes_dir_for(data_dir) -> Path:
    """S2 delete files of a dataset live next to it in <dataset>-deletes/."""
    data_dir = Path(data_dir)
    return data_dir.parent / f"{data_dir.name}-deletes"


# Exports of one S2 release are produced within minutes of each other.
RELEASE_EXPORT_WINDOW = timedelta(minutes=30)


def first_export_per_release(files: List[Path]) -> List[Path]:
    """
    Keep only the files of the earliest export of each release.

    S2 file names look like ``YYYYMMDD_HHMMSS_NNNNN_xxxxx_<uuid>``; the first
    four fields identify an export. Exports starting within
    ``RELEASE_EXPORT_WINDOW`` of a release's first export belong to that
    release. Files whose name does not parse are kept.
    """
    exports, unparsed = {}, []
    for f in files:
        parts = f.name.split("_")
        try:
            started = datetime.strptime("_".join(parts[:2]), "%Y%m%d_%H%M%S")
        except ValueError:
            unparsed.append(f)
            continue
        exports.setdefault(("_".join(parts[:4]), started), []).append(f)

    kept, release_start = [], None
    for (_, started), export_files in sorted(exports.items(), key=lambda e: e[0][1]):
        if release_start is None or started - release_start > RELEASE_EXPORT_WINDOW:
            release_start = started
            kept += export_files
    return sorted(kept + unparsed)


def create_latest_deletes_table(
    con, table: str, deletes_dir, key: str, key_type: str, first_export_only: bool = False
) -> int:
    """
    Create TEMP table ``table(key, deleted_in)`` holding, for every key in the
    S2 delete files of ``deletes_dir``, the base name of the newest delete file.
    A row version is deleted when ``deleted_in`` is newer than its own file.
    With ``first_export_only`` only the earliest delete export of each release
    is used (see ``first_export_per_release``).
    Returns the number of deleted keys (0, with an empty table, if none).
    """
    files = list_parquet_files(deletes_dir)
    if first_export_only:
        files = first_export_per_release(files)
    if not files:
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE {table} ({key} {key_type}, deleted_in VARCHAR)"
        )
        return 0
    file_list = "[" + ", ".join(sql_literal(f) for f in files) + "]"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT {key}, MAX({FILE_BASENAME}) AS deleted_in
        FROM read_parquet({file_list}, filename = true, union_by_name = true)
        WHERE {key} IS NOT NULL
        GROUP BY {key}
    """)
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


class DuckDBCache:
    """Owns the connection to one persistent DuckDB cache file."""

    def __init__(self, cache_dir, db_name: str, settings: Optional[DuckDBSettings] = None):
        self.cache_dir = str(cache_dir)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.db_file = Path(self.cache_dir) / db_name
        self.settings = (settings or DuckDBSettings()).resolved(
            Path(self.cache_dir) / "duckdb_tmp"
        )

        self.con = duckdb.connect(str(self.db_file))
        print(f"Connected to database: {self.db_file}")
        self._apply_settings()

    def _apply_settings(self):
        s = self.settings
        if s.memory_limit:
            self.con.execute(f"SET memory_limit={sql_literal(s.memory_limit)}")
        self.con.execute(f"SET threads={int(s.threads)}")
        Path(s.temp_directory).mkdir(parents=True, exist_ok=True)
        self.con.execute(f"SET temp_directory={sql_literal(s.temp_directory)}")
        if s.max_temp_directory_size:
            self.con.execute(
                f"SET max_temp_directory_size={sql_literal(s.max_temp_directory_size)}"
            )
        self.con.execute(f"SET enable_progress_bar={str(bool(s.progress_bar)).lower()}")
        self.con.execute("SET preserve_insertion_order=false")
        self.con.execute("SET checkpoint_threshold='1GB'")
        print(
            f"DuckDB settings: memory_limit={s.memory_limit}, threads={s.threads}, "
            f"temp_directory={s.temp_directory}"
        )

    def table_exists(self, table: str) -> bool:
        return (
            self.con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table],
            ).fetchone()[0]
            > 0
        )

    def row_count(self, table: str) -> int:
        return self.con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

    def close(self):
        try:
            if getattr(self, "con", None) is not None:
                self.con.close()
                self.con = None
        except Exception as e:
            print(f"Error closing connection: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
