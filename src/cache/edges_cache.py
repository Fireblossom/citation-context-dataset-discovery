"""
Citation Edges Cache Builder

Builds ``citations_all.db``: every S2 citation, with or without contexts, as a
narrow edge list. It complements ``optimized_cache.db`` (which only holds
citations that have contexts) for questions such as "who cites X" or "how often
is X cited" that must not miss publisher-restricted or context-less citations.

Tables:

``edges``            citingcorpusid, citedcorpusid, isinfluential, has_context,
                     citationid; sorted by (citingcorpusid, citedcorpusid)
``edges_by_cited``   the same rows sorted by (citedcorpusid, citingcorpusid)
``cited_counts``     citedcorpusid -> n_citations, n_with_context, n_influential
``citing_counts``    citingcorpusid -> n_references, n_with_context, n_influential

Both edge tables are sorted on their lookup key, so DuckDB's zone maps reduce
a lookup to the few row groups holding that key; no ART index is needed.
``has_context`` is true exactly for the citations that are in
``optimized_cache.citation_details`` (same version and delete rules), whose
``contexts``/``intents`` can be joined on ``citationid``.

Versioning follows ``citation_cache``: one row per ``citationid``, taken from
its newest version, dropped when a (first-export) S2 delete is newer than that
version. ``has_context`` is computed from the newest version that has
contexts, with the same delete check.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from src.cache.base import (
    FILE_BASENAME,
    DuckDBCache,
    DuckDBSettings,
    create_latest_deletes_table,
    deletes_dir_for,
    list_parquet_files,
    sql_literal,
)
from src.config import CITATION_EDGES_DB, DEFAULT_CACHE_DIR

# Same definition of "has contexts" as citation_cache.
WITH_CONTEXTS = (
    "contexts IS NOT NULL AND ARRAY_LENGTH(contexts) > 0 "
    "AND contexts != ARRAY['null']"
)

# Ranges the dedup step and the re-sort by cited ID are split into. The input
# is read once into a narrow staging table, so each range only rescans that.
DEFAULT_PARTITIONS = 8

TABLES = ("edges", "edges_by_cited", "cited_counts", "citing_counts")


class CitationEdgesBuilder(DuckDBCache):
    """Builds and queries the all-citations edge cache."""

    def __init__(
        self,
        data_dir: str = "data/citations",
        cache_dir: Optional[str] = None,
        settings: Optional[DuckDBSettings] = None,
    ):
        super().__init__(cache_dir or DEFAULT_CACHE_DIR, CITATION_EDGES_DB, settings)
        self.data_dir = data_dir

    def check_table_ready(self) -> bool:
        return all(self.table_exists(t) for t in TABLES) and self.row_count("edges") > 0

    # ==================== Build ====================

    def create_edges_tables(
        self, force_recreate: bool = False, partitions: int = DEFAULT_PARTITIONS
    ) -> bool:
        print("=== Building citation edges cache (all citations) ===")
        if self.check_table_ready() and not force_recreate:
            print(f"Existing edges cache found: {self.row_count('edges'):,} edges")
            print("Using existing cache (pass force_recreate=True to rebuild)")
            return True

        files = list_parquet_files(self.data_dir)
        if not files:
            raise FileNotFoundError(f"No parquet files found in: {self.data_dir}")
        print(
            f"Found {len(files)} parquet files, "
            f"{sum(f.stat().st_size for f in files) / 1024**3:.2f} GB"
        )
        start = time.time()

        print("\nStep 1: Read all citation versions into a narrow staging table...")
        self._create_versions_table()

        print("\nStep 2: Deduplicate into edges (sorted by citing paper)...")
        self._create_edges_table(partitions)
        self.con.execute("DROP TABLE IF EXISTS _versions")
        self.con.execute("CHECKPOINT")

        print("\nStep 3: Re-sort into edges_by_cited...")
        self._create_by_cited_table(partitions)

        print("\nStep 4: Per-paper counts...")
        self._create_count_tables()

        self.con.execute("CHECKPOINT")
        print(f"\n=== Citation edges cache built in {time.time() - start:.0f}s ===")
        return True

    def _create_versions_table(self):
        t = time.time()
        source = (
            f"read_parquet({sql_literal(Path(self.data_dir) / '*.parquet')}, "
            "filename = true, union_by_name = true)"
        )
        # A persistent (compressed) table rather than a TEMP one: it is billions
        # of rows and is scanned once per partition.
        self.con.execute(f"""
            CREATE OR REPLACE TABLE _versions AS
            SELECT citingcorpusid, citedcorpusid, isinfluential, citationid,
                   {FILE_BASENAME} AS version_file,
                   COALESCE({WITH_CONTEXTS}, false) AS ctx
            FROM {source}
        """)
        n, n_ctx = self.con.execute(
            "SELECT COUNT(*), COUNT(*) FILTER (WHERE ctx) FROM _versions"
        ).fetchone()
        print(f"_versions: {n:,} rows, {n_ctx:,} with contexts ({time.time() - t:.0f}s)")

    def _create_edges_table(self, partitions: int):
        t = time.time()
        self.con.execute("""
            CREATE OR REPLACE TABLE edges (
                citingcorpusid BIGINT,
                citedcorpusid BIGINT,
                isinfluential BOOLEAN,
                has_context BOOLEAN,
                citationid BIGINT
            )
        """)
        deleted = create_latest_deletes_table(
            self.con, "deleted_citations", deletes_dir_for(self.data_dir), "citationid", "BIGINT",
            first_export_only=True,
        )
        print(f"citationids in S2 delete files: {deleted:,}")

        min_id, max_id = self.con.execute(
            "SELECT MIN(citingcorpusid), MAX(citingcorpusid) FROM _versions"
        ).fetchone()
        ranges = []
        if min_id is not None:
            width = (max_id - min_id) // partitions + 1
            ranges = [(min_id + i * width, min_id + (i + 1) * width - 1) for i in range(partitions)]

        # All versions of a citation belong to the same citing paper, so
        # deduplicating within a citingcorpusid range is enough (as in
        # citation_cache). Rows without citationid cannot be deduplicated and
        # are kept as they are, like citation_cache does.
        for num, id_range in enumerate(ranges + [None], start=1):
            if id_range is None:
                where, params = "citingcorpusid IS NULL", []
            else:
                where, params = "citingcorpusid BETWEEN ? AND ?", list(id_range)
            self.con.execute(
                f"""
                INSERT INTO edges
                SELECT v.citingcorpusid, v.citedcorpusid, v.isinfluential,
                       v.ctx_file IS NOT NULL
                           AND (d.deleted_in IS NULL OR d.deleted_in < v.ctx_file),
                       v.citationid
                FROM (
                    SELECT citationid, newest.citingcorpusid, newest.citedcorpusid,
                           newest.isinfluential, newest_file, ctx_file
                    FROM (
                        SELECT citationid,
                               ARG_MAX({{'citingcorpusid': citingcorpusid,
                                        'citedcorpusid': citedcorpusid,
                                        'isinfluential': isinfluential}}, version_file) AS newest,
                               MAX(version_file) AS newest_file,
                               MAX(version_file) FILTER (WHERE ctx) AS ctx_file
                        FROM _versions
                        WHERE {where} AND citationid IS NOT NULL
                        GROUP BY citationid
                    )
                    UNION ALL
                    SELECT citationid, citingcorpusid, citedcorpusid, isinfluential,
                           version_file, CASE WHEN ctx THEN version_file END
                    FROM _versions
                    WHERE {where} AND citationid IS NULL
                ) v
                LEFT JOIN deleted_citations d ON v.citationid = d.citationid
                WHERE d.deleted_in IS NULL OR d.deleted_in < v.newest_file
                ORDER BY v.citingcorpusid, v.citedcorpusid
                """,
                params * 2,  # {where} appears twice
            )
            print(f"  range {num}/{len(ranges) + 1} done ({time.time() - t:.0f}s)")
        self.con.execute("DROP TABLE IF EXISTS deleted_citations")

        n, n_ctx = self.con.execute(
            "SELECT COUNT(*), COUNT(*) FILTER (WHERE has_context) FROM edges"
        ).fetchone()
        print(f"edges: {n:,} rows, {n_ctx:,} with contexts ({time.time() - t:.0f}s)")

    def _create_by_cited_table(self, partitions: int):
        t = time.time()
        self.con.execute("CREATE OR REPLACE TABLE edges_by_cited AS FROM edges LIMIT 0")
        min_id, max_id = self.con.execute(
            "SELECT MIN(citedcorpusid), MAX(citedcorpusid) FROM edges"
        ).fetchone()
        ranges = []
        if min_id is not None:
            width = (max_id - min_id) // partitions + 1
            ranges = [(min_id + i * width, min_id + (i + 1) * width - 1) for i in range(partitions)]
        for num, id_range in enumerate(ranges + [None], start=1):
            if id_range is None:
                where, params = "citedcorpusid IS NULL", []
            else:
                where, params = "citedcorpusid BETWEEN ? AND ?", list(id_range)
            self.con.execute(
                f"INSERT INTO edges_by_cited FROM edges WHERE {where} "
                "ORDER BY citedcorpusid, citingcorpusid",
                params,
            )
            print(f"  range {num}/{len(ranges) + 1} done ({time.time() - t:.0f}s)")
        print(f"edges_by_cited: {self.row_count('edges_by_cited'):,} rows ({time.time() - t:.0f}s)")

    def _create_count_tables(self):
        t = time.time()
        # Inputs are already sorted on the group key, so the output is too.
        self.con.execute("""
            CREATE OR REPLACE TABLE cited_counts AS
            SELECT citedcorpusid,
                   COUNT(*) AS n_citations,
                   COUNT(*) FILTER (WHERE has_context) AS n_with_context,
                   COUNT(*) FILTER (WHERE isinfluential) AS n_influential
            FROM edges_by_cited WHERE citedcorpusid IS NOT NULL
            GROUP BY citedcorpusid ORDER BY citedcorpusid
        """)
        self.con.execute("""
            CREATE OR REPLACE TABLE citing_counts AS
            SELECT citingcorpusid,
                   COUNT(*) AS n_references,
                   COUNT(*) FILTER (WHERE has_context) AS n_with_context,
                   COUNT(*) FILTER (WHERE isinfluential) AS n_influential
            FROM edges WHERE citingcorpusid IS NOT NULL
            GROUP BY citingcorpusid ORDER BY citingcorpusid
        """)
        print(
            f"cited_counts: {self.row_count('cited_counts'):,} rows, "
            f"citing_counts: {self.row_count('citing_counts'):,} rows ({time.time() - t:.0f}s)"
        )

    def show_cache_info(self):
        print("\n=== Citation edges cache info ===")
        for table in TABLES:
            print(f"{table}: {self.row_count(table):,} rows")
