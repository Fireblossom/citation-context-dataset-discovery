"""
Citation Cache Builder

Builds and queries the citation cache (tables ``citation_details`` and
``citing_aggregated``) from Semantic Scholar ``citations`` parquet files.

The input directory may hold a full S2 release plus later incremental update
files. Every citation is keyed by ``citationid`` and only one version is kept,
so updates replace rows instead of being counted twice: the newest version that
has contexts, or else the newest version.

Preferring versions with contexts matters because S2 ships every citations
release (full and incremental) as two exports a few minutes apart, and the
later one repeats the same citations with the contexts of publisher-restricted
citing papers removed (the Graph API still serves them via /citations).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

from src.cache.base import DuckDBCache, DuckDBSettings, list_parquet_files, sql_literal
from src.config import CITATION_CACHE_DB, DEFAULT_CACHE_DIR

# Range of citingcorpusid values aggregated per INSERT. citation_details is
# sorted by citingcorpusid, so each range only reads its own row groups.
DEFAULT_ID_BATCH_SIZE = 5_000_000


class CitationCacheBuilder(DuckDBCache):
    """
    Optimized cache builder for citation contexts.

    Provides:
    - Table creation from parquet files
    - Batch query for citation contexts by citing paper
    - Batch query for papers that cite a given paper
    """

    def __init__(
        self,
        data_dir: str = "data/citations",
        cache_dir: Optional[str] = None,
        settings: Optional[DuckDBSettings] = None,
    ):
        """
        Args:
            data_dir: Directory with citation parquet files
            cache_dir: Directory holding the DuckDB cache database
            settings: DuckDB resource settings (detected when omitted)
        """
        super().__init__(cache_dir or DEFAULT_CACHE_DIR, CITATION_CACHE_DB, settings)
        self.data_dir = data_dir

    def check_table_ready(self) -> bool:
        """Check if cache tables exist and have data."""
        try:
            if not (
                self.table_exists("citing_aggregated")
                and self.table_exists("citation_details")
            ):
                return False
            agg_count = self.row_count("citing_aggregated")
            detail_count = self.row_count("citation_details")
            print("[OK] Citation cache tables ready")
            print(f"  citing_aggregated: {agg_count:,} rows")
            print(f"  citation_details: {detail_count:,} rows")
            return agg_count > 0 and detail_count > 0
        except Exception:
            return False

    # ==================== Build ====================

    def create_optimized_cache_table(
        self,
        force_recreate: bool = False,
        id_batch_size: int = DEFAULT_ID_BATCH_SIZE,
        create_indexes: bool = True,
    ) -> bool:
        """
        Create the citation cache tables from parquet files.

        Args:
            force_recreate: Rebuild even if the tables already exist
            id_batch_size: citingcorpusid range aggregated per batch
            create_indexes: Create ART indexes on the corpus ID columns

        Returns:
            True if successful
        """
        print("=== Building citation cache ===")

        if (
            self.table_exists("citing_aggregated")
            and self.table_exists("citation_details")
            and not force_recreate
        ):
            print(
                f"Existing cache found: citing_aggregated "
                f"{self.row_count('citing_aggregated'):,} rows, citation_details "
                f"{self.row_count('citation_details'):,} rows"
            )
            print("Using existing cache (pass force_recreate=True to rebuild)")
            return True

        parquet_files = list_parquet_files(self.data_dir)
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in: {self.data_dir}")
        total_size = sum(f.stat().st_size for f in parquet_files)
        print(
            f"Found {len(parquet_files)} parquet files, "
            f"{total_size / (1024**3):.2f} GB"
        )

        start_time = time.time()

        print("\nStep 1: Deduplicate, filter and sort into citation_details...")
        self._create_detail_table()

        print("\nStep 2: Aggregate per citing paper into citing_aggregated...")
        self._create_aggregated_table(id_batch_size)

        if create_indexes:
            print("\nStep 3: Create indexes...")
            step_start = time.time()
            self.con.execute(
                "CREATE INDEX IF NOT EXISTS idx_citingcorpusid ON citing_aggregated(citingcorpusid)"
            )
            self.con.execute(
                "CREATE INDEX IF NOT EXISTS idx_citation_count ON citing_aggregated(citation_count)"
            )
            self.con.execute(
                "CREATE INDEX IF NOT EXISTS idx_detail_citing ON citation_details(citingcorpusid)"
            )
            self.con.execute(
                "CREATE INDEX IF NOT EXISTS idx_detail_cited ON citation_details(citedcorpusid)"
            )
            print(f"Indexes created in {time.time() - step_start:.0f}s")

        self.con.execute("CHECKPOINT")
        print(f"\n=== Citation cache built in {time.time() - start_time:.0f}s ===")
        return True

    def _create_detail_table(self):
        step_start = time.time()
        source = (
            f"read_parquet({sql_literal(Path(self.data_dir) / '*.parquet')}, "
            "filename = true, union_by_name = true)"
        )

        has_contexts = (
            "COALESCE(contexts IS NOT NULL AND ARRAY_LENGTH(contexts) > 0 "
            "AND contexts != ARRAY['null'], false)"
        )

        # Only citations present in more than one file need the (expensive)
        # window; everything else passes straight through.
        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE duplicate_citations AS
            SELECT citationid
            FROM {source}
            WHERE citationid IS NOT NULL
            GROUP BY citationid
            HAVING COUNT(*) > 1
        """)
        duplicates = self.row_count("duplicate_citations")
        print(f"citationids present in more than one file: {duplicates:,}")

        self.con.execute(f"""
            CREATE OR REPLACE TABLE citation_details AS
            WITH source AS (
                SELECT citingcorpusid, citedcorpusid, isinfluential,
                       contexts, intents, citationid, filename
                FROM {source}
            ),
            chosen AS (
                SELECT * FROM source
                WHERE citationid IS NULL
                   OR citationid NOT IN (SELECT citationid FROM duplicate_citations)
                UNION ALL
                SELECT * FROM source
                WHERE citationid IN (SELECT citationid FROM duplicate_citations)
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY citationid ORDER BY {has_contexts} DESC, filename DESC
                ) = 1
            )
            SELECT citingcorpusid, citedcorpusid, isinfluential,
                   contexts, intents, citationid
            FROM chosen
            WHERE {has_contexts}
            ORDER BY citingcorpusid, citedcorpusid
        """)
        self.con.execute("DROP TABLE IF EXISTS duplicate_citations")

        detail_count = self.row_count("citation_details")
        print(
            f"citation_details: {detail_count:,} rows with contexts "
            f"({time.time() - step_start:.0f}s)"
        )

    def _create_aggregated_table(self, id_batch_size: int):
        step_start = time.time()
        self.con.execute("""
            CREATE OR REPLACE TABLE citing_aggregated (
                citingcorpusid BIGINT,
                citation_count BIGINT,
                unique_cited_count BIGINT,
                influential_count BIGINT,
                all_cited_papers VARCHAR,
                all_contexts VARCHAR,
                all_intents VARCHAR
            )
        """)

        min_id, max_id = self.con.execute(
            "SELECT MIN(citingcorpusid), MAX(citingcorpusid) FROM citation_details"
        ).fetchone()
        if min_id is None:
            print("citation_details is empty, nothing to aggregate")
            return

        total_batches = (max_id - min_id) // id_batch_size + 1
        print(
            f"citingcorpusid range {min_id:,}..{max_id:,}: "
            f"{total_batches} batches of {id_batch_size:,} IDs"
        )

        # contexts/intents are lists; cast them explicitly (newer DuckDB versions
        # no longer cast lists to VARCHAR implicitly, older ones did exactly this).
        for batch_num, batch_start in enumerate(
            range(min_id, max_id + 1, id_batch_size), start=1
        ):
            batch_end = batch_start + id_batch_size - 1
            self.con.execute(
                """
                INSERT INTO citing_aggregated
                SELECT
                    citingcorpusid,
                    COUNT(*) AS citation_count,
                    COUNT(DISTINCT citedcorpusid) AS unique_cited_count,
                    SUM(CASE WHEN isinfluential = true THEN 1 ELSE 0 END) AS influential_count,
                    STRING_AGG(DISTINCT citedcorpusid::VARCHAR, ',') AS all_cited_papers,
                    STRING_AGG(DISTINCT contexts::VARCHAR, ' | ') AS all_contexts,
                    STRING_AGG(DISTINCT intents::VARCHAR, ' | ') AS all_intents
                FROM citation_details
                WHERE citingcorpusid BETWEEN ? AND ?
                GROUP BY citingcorpusid
                """,
                [batch_start, batch_end],
            )
            if batch_num % 10 == 0 or batch_num == total_batches:
                print(
                    f"  batch {batch_num}/{total_batches} done "
                    f"({time.time() - step_start:.0f}s)"
                )

        agg_count = self.row_count("citing_aggregated")
        print(
            f"citing_aggregated: {agg_count:,} rows ({time.time() - step_start:.0f}s)"
        )

    # ==================== Queries ====================

    def batch_query_contexts(
        self,
        citingcorpusid_list: List[int],
        include_details: bool = True,
        max_results_per_id: Optional[int] = None,
    ) -> Dict:
        """
        Batch query citation contexts for citing papers.

        Args:
            citingcorpusid_list: List of citing paper corpus IDs
            include_details: Whether to include detailed citation info
            max_results_per_id: Maximum results per ID

        Returns:
            Dictionary with results keyed by corpus ID, plus "query_stats"
        """
        if not citingcorpusid_list:
            return {}

        print("=== Batch Querying Context Info ===")
        print(f"Query ID count: {len(citingcorpusid_list)}")
        print(f"Include details: {include_details}")

        start_time = time.time()
        results = {}
        placeholders = ",".join(["?" for _ in citingcorpusid_list])

        print("Querying aggregated info...")
        agg_results = self.con.execute(
            f"""
            SELECT citingcorpusid, citation_count, unique_cited_count, influential_count
            FROM citing_aggregated
            WHERE citingcorpusid IN ({placeholders})
            """,
            citingcorpusid_list,
        ).df()

        for _, row in agg_results.iterrows():
            results[row["citingcorpusid"]] = {
                "citation_count": int(row["citation_count"]),
                "unique_cited_count": int(row["unique_cited_count"]),
                "influential_count": int(row["influential_count"]),
                "citation_details": [],
            }

        if include_details:
            print("Querying detailed citation info...")
            detail_results = self.con.execute(
                f"""
                SELECT citingcorpusid, citedcorpusid, isinfluential, contexts, intents
                FROM citation_details
                WHERE citingcorpusid IN ({placeholders})
                """,
                citingcorpusid_list,
            ).df()

            print("Processing query results...")
            for _, row in detail_results.iterrows():
                citing_id = row["citingcorpusid"]
                if citing_id not in results:
                    continue
                details = results[citing_id]["citation_details"]
                if max_results_per_id is None or len(details) < max_results_per_id:
                    details.append(
                        {
                            "citedcorpusid": row["citedcorpusid"],
                            "isinfluential": bool(row["isinfluential"]),
                            "contexts": row["contexts"],
                            "intents": row["intents"],
                        }
                    )

        elapsed = time.time() - start_time
        found_ids = len(results)
        total_results = sum(len(r["citation_details"]) for r in results.values())

        print("Query complete!")
        print(f"Found IDs: {found_ids}")
        print(f"Not found IDs: {len(citingcorpusid_list) - found_ids}")
        print(f"Total results: {total_results}")
        print(f"Query time: {elapsed:.4f}s")

        results["query_stats"] = {
            "found_ids": found_ids,
            "not_found_ids": len(citingcorpusid_list) - found_ids,
            "total_results": total_results,
            "query_time": elapsed,
        }
        return results

    def batch_query_citing_papers(
        self, citedcorpusid_list: List[int], max_results_per_id: Optional[int] = None
    ) -> Dict:
        """
        Query papers that cite the given papers.

        Args:
            citedcorpusid_list: List of cited paper corpus IDs
            max_results_per_id: Maximum results per ID

        Returns:
            Dictionary with results keyed by cited corpus ID, plus "query_stats"
        """
        if not citedcorpusid_list:
            return {}

        print("=== Batch Querying Citing Papers (by cited paper) ===")
        print(f"Query cited paper ID count: {len(citedcorpusid_list)}")

        start_time = time.time()
        results = {}
        placeholders = ",".join(["?" for _ in citedcorpusid_list])

        print("Querying citation info (by citedcorpusid)...")
        query_results = self.con.execute(
            f"""
            SELECT citedcorpusid, citingcorpusid, isinfluential, contexts, intents
            FROM citation_details
            WHERE citedcorpusid IN ({placeholders})
            ORDER BY citedcorpusid, citingcorpusid
            """,
            citedcorpusid_list,
        ).df()

        print("Processing query results...")
        for _, row in query_results.iterrows():
            citing_papers = results.setdefault(
                row["citedcorpusid"], {"citing_papers": []}
            )["citing_papers"]
            if max_results_per_id is None or len(citing_papers) < max_results_per_id:
                citing_papers.append(
                    {
                        "citingcorpusid": row["citingcorpusid"],
                        "isinfluential": bool(row["isinfluential"]),
                        "contexts": row["contexts"],
                        "intents": row["intents"],
                    }
                )

        elapsed = time.time() - start_time
        found_ids = len(results)
        total_citing = sum(len(r["citing_papers"]) for r in results.values())

        print("Query complete!")
        print(f"Found cited paper IDs: {found_ids}")
        print(f"Not found IDs: {len(citedcorpusid_list) - found_ids}")
        print(f"Total citation records: {total_citing}")
        print(f"Query time: {elapsed:.4f}s")

        results["query_stats"] = {
            "found_ids": found_ids,
            "not_found_ids": len(citedcorpusid_list) - found_ids,
            "total_citing_papers": total_citing,
            "query_time": elapsed,
        }
        return results

    def show_cache_info(self):
        """Display cache statistics."""
        print("\n=== Citation cache info ===")
        try:
            total, avg_citations, max_citations, total_citations = self.con.execute("""
                SELECT COUNT(*), AVG(citation_count), MAX(citation_count), SUM(citation_count)
                FROM citing_aggregated
            """).fetchone()
            print(f"Citing papers: {total:,}")
            print(f"Avg citations with context per paper: {avg_citations:.2f}")
            print(f"Max citations with context per paper: {max_citations:,}")
            print(f"Total citations with context: {total_citations:,}")
            print(f"citation_details rows: {self.row_count('citation_details'):,}")
        except Exception as e:
            print(f"Error reading cache info: {e}")


# Alias for backward compatibility
OptimizedCacheBuilder = CitationCacheBuilder
