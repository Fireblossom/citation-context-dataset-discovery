"""
Papers Cache Builder

Builds and queries the ``papers_optimized`` table (paper metadata keyed by
corpusid) from Semantic Scholar ``papers``, ``abstracts`` and ``paper-ids``
parquet files, and applies incremental updates from newer release files.

Each input directory may hold a full S2 release plus later incremental update
files; for every key only the version from the newest file is kept. Keys listed
in S2 delete files (``<dataset>-deletes/`` next to each dataset directory) are
dropped when the delete is newer than that version.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

from src.cache.base import (
    FILE_BASENAME,
    DuckDBCache,
    DuckDBSettings,
    create_latest_deletes_table,
    list_parquet_files,
    sql_literal,
)
from src.config import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, PAPERS_CACHE_DB

PAPERS_TABLE_SCHEMA = """
    corpusid BIGINT PRIMARY KEY,
    title VARCHAR,
    year INTEGER,
    venue VARCHAR,
    authors STRUCT(authorId VARCHAR, "name" VARCHAR)[],
    abstract TEXT,
    sha_id VARCHAR,
    externalids STRUCT(
        ACL VARCHAR, ArXiv VARCHAR, CorpusId VARCHAR, DBLP VARCHAR,
        DOI VARCHAR, MAG VARCHAR, PubMed VARCHAR, PubMedCentral VARCHAR
    ),
    url VARCHAR,
    publicationvenueid VARCHAR,
    referencecount INTEGER,
    citationcount INTEGER,
    influentialcitationcount INTEGER,
    isopenaccess BOOLEAN,
    s2fieldsofstudy STRUCT(category VARCHAR, source VARCHAR)[],
    publicationtypes VARCHAR[],
    publicationdate VARCHAR,
    journal STRUCT(name VARCHAR, pages VARCHAR, volume VARCHAR),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
"""

PAPERS_COLUMNS = """
    corpusid, title, year, venue, authors, abstract, sha_id,
    externalids, url, publicationvenueid, referencecount,
    citationcount, influentialcitationcount, isopenaccess,
    s2fieldsofstudy, publicationtypes, publicationdate, journal
"""


def _paper_metadata_select(alias: str) -> str:
    """SELECT list for the paper-level columns after abstract and sha_id."""
    p = alias
    return f"""
        {p}.externalids,
        {p}.url,
        {p}.publicationvenueid,
        {p}.referencecount,
        {p}.citationcount,
        {p}.influentialcitationcount,
        {p}.isopenaccess,
        CASE
            WHEN {p}.s2fieldsofstudy IS NOT NULL THEN
                list_transform({p}.s2fieldsofstudy, x -> struct_pack(category := lower(x.category), source := x.source))
            ELSE {p}.s2fieldsofstudy
        END AS s2fieldsofstudy,
        {p}.publicationtypes,
        {p}.publicationdate,
        {p}.journal
    """


class PapersCacheBuilder(DuckDBCache):
    """
    Optimized cache builder for paper metadata.

    Provides:
    - Table creation from parquet files
    - Incremental updates from newer release files
    - Fast corpus ID lookups and batch queries
    """

    # (source_type recorded in processed_files, sub-directory of data_dir)
    SOURCES = (("papers", "papers"), ("abstracts", "abstracts"), ("paper_ids", "paper-ids"))
    # (source_type, sub-directory, key column) of the S2 delete files
    DELETE_SOURCES = (
        ("papers_deletes", "papers-deletes", "corpusid"),
        ("abstracts_deletes", "abstracts-deletes", "corpusid"),
        ("paper_ids_deletes", "paper-ids-deletes", "sha"),
    )

    def __init__(
        self,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        settings: Optional[DuckDBSettings] = None,
    ):
        """
        Args:
            data_dir: Directory containing papers/, abstracts/ and paper-ids/ parquet dirs
            cache_dir: Directory holding the DuckDB cache database
            settings: DuckDB resource settings (detected when omitted)
        """
        super().__init__(cache_dir or DEFAULT_CACHE_DIR, PAPERS_CACHE_DB, settings)
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.papers_dir = Path(self.data_dir) / "papers"
        self.abstracts_dir = Path(self.data_dir) / "abstracts"
        self.paper_ids_dir = Path(self.data_dir) / "paper-ids"

    def _source_dirs(self, with_deletes: bool = False):
        sources = list(self.SOURCES)
        if with_deletes:
            sources += [(source, subdir) for source, subdir, _ in self.DELETE_SOURCES]
        return [(source, Path(self.data_dir) / subdir) for source, subdir in sources]

    def check_data_files(self) -> Dict:
        """Check that the parquet directories exist and report their size."""
        print("=== Checking data files ===")
        result = {}
        for source, directory in self._source_dirs():
            if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")
            files = list_parquet_files(directory)
            size = sum(f.stat().st_size for f in files)
            print(f"{directory.name}: {len(files)} files, {size / (1024**3):.2f} GB")
            result[source] = {"count": len(files), "size_gb": size / (1024**3)}
        return result

    def check_table_ready(self) -> bool:
        """Check if the papers_optimized table exists and has data."""
        try:
            if not self.table_exists("papers_optimized"):
                return False
            count = self.row_count("papers_optimized")
            print("[OK] Papers cache table ready")
            print(f"  Papers table: {count:,} rows")
            return count > 0
        except Exception:
            return False

    # ==================== Full build ====================

    def create_optimized_papers_table(self, force_recreate: bool = False) -> bool:
        """
        Create the papers_optimized table from parquet files.

        Args:
            force_recreate: Rebuild even if the table already exists

        Returns:
            True if successful
        """
        print("\n=== Building papers cache ===")

        if self.table_exists("papers_optimized") and not force_recreate:
            print(f"Existing cache found: {self.row_count('papers_optimized'):,} rows")
            print("Using existing cache (pass force_recreate=True to rebuild)")
            return True

        start_time = time.time()
        papers_glob = sql_literal(self.papers_dir / "*.parquet")
        abstracts_glob = sql_literal(self.abstracts_dir / "*.parquet")
        paper_ids_glob = sql_literal(self.paper_ids_dir / "*.parquet")

        self.con.execute(f"CREATE OR REPLACE TABLE papers_optimized ({PAPERS_TABLE_SCHEMA})")

        for source, subdir, key in self.DELETE_SOURCES:
            deleted = create_latest_deletes_table(
                self.con, f"deleted_{source}", Path(self.data_dir) / subdir, key,
                "VARCHAR" if key == "sha" else "BIGINT",
            )
            print(f"{subdir}: {deleted:,} deleted keys")

        # papers and abstracts are keyed by corpusid, paper-ids by sha. For each
        # key keep the row from the newest file unless a newer delete file lists
        # it; then pick one sha per paper, preferring the primary one.
        print("Deduplicating, joining and inserting...")
        self.con.execute(f"""
            INSERT INTO papers_optimized ({PAPERS_COLUMNS})
            WITH papers AS (
                SELECT p.*
                FROM (
                    SELECT *, {FILE_BASENAME} AS version_file
                    FROM read_parquet({papers_glob}, filename = true, union_by_name = true)
                    WHERE corpusid IS NOT NULL
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY corpusid ORDER BY filename DESC, title DESC
                    ) = 1
                ) p
                LEFT JOIN deleted_papers_deletes d ON p.corpusid = d.corpusid
                WHERE d.deleted_in IS NULL OR d.deleted_in < p.version_file
            ),
            abstracts AS (
                SELECT a.corpusid, a.abstract
                FROM (
                    SELECT corpusid, abstract, {FILE_BASENAME} AS version_file
                    FROM read_parquet({abstracts_glob}, filename = true, union_by_name = true)
                    WHERE corpusid IS NOT NULL
                    QUALIFY ROW_NUMBER() OVER (PARTITION BY corpusid ORDER BY filename DESC) = 1
                ) a
                LEFT JOIN deleted_abstracts_deletes d ON a.corpusid = d.corpusid
                WHERE d.deleted_in IS NULL OR d.deleted_in < a.version_file
            ),
            paper_ids AS (
                SELECT corpusid, sha
                FROM (
                    SELECT i.corpusid, i.sha, i."primary"
                    FROM (
                        SELECT corpusid, sha, "primary", {FILE_BASENAME} AS version_file
                        FROM read_parquet({paper_ids_glob}, filename = true, union_by_name = true)
                        WHERE corpusid IS NOT NULL
                        QUALIFY ROW_NUMBER() OVER (PARTITION BY sha ORDER BY filename DESC) = 1
                    ) i
                    LEFT JOIN deleted_paper_ids_deletes d ON i.sha = d.sha
                    WHERE d.deleted_in IS NULL OR d.deleted_in < i.version_file
                )
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY corpusid ORDER BY "primary" DESC NULLS LAST, sha
                ) = 1
            )
            SELECT
                p.corpusid,
                COALESCE(p.title, '') AS title,
                p.year,
                COALESCE(p.venue, '') AS venue,
                p.authors,
                COALESCE(a.abstract, '') AS abstract,
                COALESCE(pid.sha, '') AS sha_id,
                {_paper_metadata_select("p")}
            FROM papers p
            LEFT JOIN abstracts a ON p.corpusid = a.corpusid
            LEFT JOIN paper_ids pid ON p.corpusid = pid.corpusid
        """)

        for source, _, _ in self.DELETE_SOURCES:
            self.con.execute(f"DROP TABLE IF EXISTS deleted_{source}")

        # Everything currently on disk is now in the table, so a later
        # incremental update only has to look at files added after this.
        for source, directory in self._source_dirs(with_deletes=True):
            for parquet_file in list_parquet_files(directory):
                self.mark_file_processed(parquet_file.name, source)

        self.con.execute("CHECKPOINT")
        final_count = self.row_count("papers_optimized")
        print("\n=== Papers cache built ===")
        print(f"Total rows: {final_count:,}")
        print(f"Elapsed: {time.time() - start_time:.0f}s")
        return True

    # ==================== Incremental update ====================

    def _init_processed_files_table(self):
        if getattr(self, "_processed_files_ready", False):
            return

        create_sql = """
            CREATE TABLE processed_files (
                filename VARCHAR,
                source_type VARCHAR,
                row_count BIGINT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source_type, filename)
            )
        """
        if not self.table_exists("processed_files"):
            self.con.execute(create_sql)
        else:
            # Older caches keyed this table by filename alone, so files with the
            # same name in papers/, abstracts/ and paper-ids/ overwrote each
            # other's entries; migrate them to the (source_type, filename) key.
            pk = self.con.execute("""
                SELECT constraint_column_names FROM duckdb_constraints()
                WHERE table_name = 'processed_files' AND constraint_type = 'PRIMARY KEY'
            """).fetchone()
            if pk and list(pk[0]) == ["filename"]:
                self.con.execute("ALTER TABLE processed_files RENAME TO processed_files_old")
                self.con.execute(create_sql)
                self.con.execute("""
                    INSERT INTO processed_files (filename, source_type, row_count, processed_at)
                    SELECT filename, source_type, row_count, processed_at FROM processed_files_old
                """)
                self.con.execute("DROP TABLE processed_files_old")

        self._processed_files_ready = True

    def get_processed_files(self, source_type: Optional[str] = None) -> set:
        """Get the set of already processed filenames."""
        self._init_processed_files_table()
        if source_type:
            rows = self.con.execute(
                "SELECT filename FROM processed_files WHERE source_type = ?",
                [source_type],
            ).fetchall()
        else:
            rows = self.con.execute("SELECT filename FROM processed_files").fetchall()
        return {row[0] for row in rows}

    def get_new_files(self, directory: Path, source_type: str) -> List[Path]:
        """Parquet files not processed yet, oldest first."""
        all_files = list_parquet_files(directory)
        processed = self.get_processed_files(source_type)
        new_files = [f for f in all_files if f.name not in processed]
        print(
            f"{source_type}: {len(all_files)} files, "
            f"{len(all_files) - len(new_files)} processed, {len(new_files)} new"
        )
        return new_files

    def mark_file_processed(self, filename: str, source_type: str, row_count: int = 0):
        """Record a file as processed."""
        self._init_processed_files_table()
        self.con.execute(
            """
            INSERT OR REPLACE INTO processed_files (filename, source_type, row_count, processed_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [filename, source_type, row_count],
        )

    def incremental_update(
        self, from_jsonl_dir: Optional[str] = None, workers: int = 8
    ) -> bool:
        """
        Apply new parquet files (optionally converted from jsonl.gz first).

        Args:
            from_jsonl_dir: Directory with papers/, abstracts/, paper-ids/
                sub-directories of jsonl.gz files to convert first
            workers: Parallel workers for the jsonl.gz conversion

        Returns:
            True if every new file was applied
        """
        print("\n=== Incremental update ===")

        if from_jsonl_dir:
            print("\nStep 1: Converting jsonl.gz files...")
            from src.utils.jsonl_converter import convert_jsonl_to_parquet

            subdirs = [s for _, s in self.SOURCES] + [s for _, s, _ in self.DELETE_SOURCES]
            for subdir in subdirs:
                jsonl_dir = Path(from_jsonl_dir) / subdir
                if jsonl_dir.exists():
                    convert_jsonl_to_parquet(
                        str(jsonl_dir), str(Path(self.data_dir) / subdir), workers=workers
                    )

        print("\nStep 2: Detecting new files...")
        handlers = {
            "papers": self._apply_papers_file,
            "abstracts": self._apply_abstracts_file,
            "paper_ids": self._apply_paper_ids_file,
            "papers_deletes": self._apply_papers_deletes,
            "abstracts_deletes": self._apply_abstracts_deletes,
            "paper_ids_deletes": self._apply_paper_ids_deletes,
        }
        # Updates and deletes are applied in file-name (= time) order, so a
        # record deleted and later re-added ends up present, and vice versa.
        events = sorted(
            (parquet_file.name, source, parquet_file)
            for source, directory in self._source_dirs(with_deletes=True)
            for parquet_file in self.get_new_files(directory, source)
        )

        if not events:
            print("[OK] No new files to process")
            return True

        if not self.check_table_ready():
            print("Cache table missing, running a full build...")
            return self.create_optimized_papers_table()

        print(f"\nStep 3: Applying {len(events)} new files...")
        start_time = time.time()
        total_new_rows = 0
        errors = 0
        for name, source, parquet_file in events:
            try:
                rows = handlers[source](sql_literal(parquet_file))
                self.mark_file_processed(name, source, rows)
                total_new_rows += rows if source == "papers" else 0
                print(f"  [OK] {source}: {name}: {rows:,} rows")
            except Exception as e:
                errors += 1
                print(f"  [ERROR] {source}: {name}: {e}")

        self.con.execute("CHECKPOINT")
        print("\n=== Incremental update finished ===")
        print(f"New/updated paper rows: {total_new_rows:,}")
        print(f"Total rows: {self.row_count('papers_optimized'):,}")
        print(f"Failed files: {errors}")
        print(f"Elapsed: {time.time() - start_time:.0f}s")
        return errors == 0

    def _file_rows(self, path: str) -> int:
        return self.con.execute(f"SELECT COUNT(*) FROM read_parquet({path})").fetchone()[0]

    def _apply_papers_file(self, path: str) -> int:
        # Paper rows are replaced wholesale, but abstract and sha_id come from
        # other datasets, so carry the existing values over.
        self.con.execute(f"""
            INSERT OR REPLACE INTO papers_optimized ({PAPERS_COLUMNS}, created_at)
            SELECT
                n.corpusid,
                COALESCE(n.title, '') AS title,
                n.year,
                COALESCE(n.venue, '') AS venue,
                n.authors,
                COALESCE(old.abstract, '') AS abstract,
                COALESCE(old.sha_id, '') AS sha_id,
                {_paper_metadata_select("n")},
                CURRENT_TIMESTAMP AS created_at
            FROM (
                SELECT * FROM read_parquet({path})
                WHERE corpusid IS NOT NULL
                QUALIFY ROW_NUMBER() OVER (PARTITION BY corpusid ORDER BY title DESC) = 1
            ) n
            LEFT JOIN papers_optimized old ON n.corpusid = old.corpusid
        """)
        return self._file_rows(path)

    def _apply_abstracts_file(self, path: str) -> int:
        self.con.execute(f"""
            UPDATE papers_optimized SET abstract = COALESCE(a.abstract, '')
            FROM (
                SELECT corpusid, abstract FROM read_parquet({path})
                WHERE corpusid IS NOT NULL
                QUALIFY ROW_NUMBER() OVER (PARTITION BY corpusid) = 1
            ) a
            WHERE papers_optimized.corpusid = a.corpusid
        """)
        return self._file_rows(path)

    def _apply_paper_ids_file(self, path: str) -> int:
        # A non-primary sha only fills in papers that have no sha yet.
        self.con.execute(f"""
            UPDATE papers_optimized SET sha_id = p.sha
            FROM (
                SELECT corpusid, sha, "primary"
                FROM read_parquet({path})
                WHERE corpusid IS NOT NULL
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY corpusid ORDER BY "primary" DESC NULLS LAST, sha
                ) = 1
            ) p
            WHERE papers_optimized.corpusid = p.corpusid
              AND (p."primary" OR papers_optimized.sha_id = '')
        """)
        return self._file_rows(path)

    def _apply_papers_deletes(self, path: str) -> int:
        self.con.execute(f"""
            DELETE FROM papers_optimized
            WHERE corpusid IN (SELECT corpusid FROM read_parquet({path}) WHERE corpusid IS NOT NULL)
        """)
        return self._file_rows(path)

    def _apply_abstracts_deletes(self, path: str) -> int:
        self.con.execute(f"""
            UPDATE papers_optimized SET abstract = ''
            WHERE corpusid IN (SELECT corpusid FROM read_parquet({path}) WHERE corpusid IS NOT NULL)
        """)
        return self._file_rows(path)

    def _apply_paper_ids_deletes(self, path: str) -> int:
        # Only the stored sha is known here, so a deleted sha is cleared rather
        # than replaced by another sha of the paper (a full build would pick one).
        self.con.execute(f"""
            UPDATE papers_optimized SET sha_id = ''
            WHERE sha_id IN (SELECT sha FROM read_parquet({path}) WHERE sha IS NOT NULL)
        """)
        return self._file_rows(path)

    # ==================== Queries ====================

    def batch_query_papers(self, corpusid_list: list, fields: Optional[list] = None) -> dict:
        """Batch query paper metadata by corpus IDs."""
        if not corpusid_list:
            return {}
        try:
            field_str = ", ".join(fields) if fields else "*"
            placeholders = ",".join(["?" for _ in corpusid_list])
            results = self.con.execute(
                f"SELECT {field_str} FROM papers_optimized WHERE corpusid IN ({placeholders})",
                corpusid_list,
            ).df()
            return {row["corpusid"]: row.to_dict() for _, row in results.iterrows()}
        except Exception as e:
            print(f"Error in batch query papers: {e}")
            return {}

    def get_paper_title(self, corpusid: int) -> str:
        """Get the title of a single paper."""
        try:
            result = self.con.execute(
                "SELECT title FROM papers_optimized WHERE corpusid = ?", [corpusid]
            ).fetchone()
            return result[0] if result else "Unknown Title"
        except Exception:
            return "Unknown Title"

    def show_table_info(self):
        """Display table statistics."""
        print("\n=== Papers cache info ===")
        try:
            columns = self.con.execute("DESCRIBE papers_optimized").fetchall()
            print(f"Columns: {len(columns)}")
            total, with_abstract, with_citations = self.con.execute("""
                SELECT
                    COUNT(*),
                    COUNT(CASE WHEN abstract != '' THEN 1 END),
                    COUNT(CASE WHEN citationcount IS NOT NULL THEN 1 END)
                FROM papers_optimized
            """).fetchone()
            print(f"Total papers: {total:,}")
            if total:
                print(f"With abstract: {with_abstract:,} ({with_abstract / total * 100:.1f}%)")
                print(
                    f"With citation count: {with_citations:,} "
                    f"({with_citations / total * 100:.1f}%)"
                )
        except Exception as e:
            print(f"Error reading table info: {e}")

    def test_query_performance(self):
        """Time a single and a 1000-ID batch lookup."""
        print("\n=== Query performance ===")
        try:
            test_id = self.con.execute(
                "SELECT corpusid FROM papers_optimized LIMIT 1"
            ).fetchone()[0]
            start = time.time()
            self.con.execute(
                "SELECT * FROM papers_optimized WHERE corpusid = ?", [test_id]
            ).df()
            print(f"Single lookup: {time.time() - start:.4f}s")

            test_ids = [
                row[0]
                for row in self.con.execute(
                    "SELECT corpusid FROM papers_optimized ORDER BY RANDOM() LIMIT 1000"
                ).fetchall()
            ]
            placeholders = ",".join(["?" for _ in test_ids])
            start = time.time()
            self.con.execute(
                f"SELECT corpusid, title, year FROM papers_optimized WHERE corpusid IN ({placeholders})",
                test_ids,
            ).df()
            elapsed = time.time() - start
            print(f"Batch lookup of {len(test_ids)}: {elapsed:.4f}s")
        except Exception as e:
            print(f"Performance test failed: {e}")

    def export_table(self, output_file: Optional[str] = None) -> bool:
        """Export the table to a parquet file."""
        output_path = (
            Path(output_file)
            if output_file
            else Path(self.cache_dir) / "papers_optimized.parquet"
        )
        print(f"\nExporting to: {output_path}")
        try:
            self.con.execute(
                f"COPY papers_optimized TO {sql_literal(output_path)} (FORMAT PARQUET)"
            )
            print(f"Export done: {output_path.stat().st_size / (1024**2):.2f} MB")
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False


# Alias for backward compatibility
PapersOptimizedCacheBuilder = PapersCacheBuilder
