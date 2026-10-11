"""
Citation Cache Builder - Complete implementation

Creates and manages citation cache tables (citing_aggregated, citation_details) from Semantic Scholar parquet files.
"""

import duckdb
import time
from pathlib import Path
from typing import Dict, List, Optional

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import DEFAULT_CACHE_DIR, CITATION_CACHE_DB


class CitationCacheBuilder:
    """
    Optimized cache builder for citation contexts.

    Provides:
    - Table creation from parquet files
    - Batch query for citation contexts by citing paper
    - Batch query for papers that cite a given paper
    """

    def __init__(self, data_dir: str = "data/citations", cache_dir: str = None):
        """
        Initialize the citation cache builder.

        Args:
            data_dir: Path to citation parquet files
            cache_dir: Path to store DuckDB cache database
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR

        # Create cache directory
        Path(self.cache_dir).mkdir(exist_ok=True)

        # Connect to persistent database
        db_file = Path(self.cache_dir) / CITATION_CACHE_DB
        try:
            self.con = duckdb.connect(str(db_file))
            print(f"Connected to database: {db_file}")

            test_result = self.con.execute("SELECT 1").fetchone()
            if test_result[0] != 1:
                raise Exception("数据库连接测试失败")
            print("Database connection test successful")

        except Exception as e:
            print(f"连接数据库失败: {e}")
            raise

        self._setup_optimizations()

    def _setup_optimizations(self):
        """Configure DuckDB performance parameters."""
        self.con.execute("SET memory_limit='1TB'")
        self.con.execute("SET threads=32")
        self.con.execute("SET enable_progress_bar=true")
        self.con.execute("SET preserve_insertion_order=false")
        self.con.execute("SET temp_directory='/tmp'")
        self.con.execute("SET checkpoint_threshold='1GB'")
        self.con.execute("SET wal_autocheckpoint='1GB'")
        print("DuckDB optimization parameters set")

    def check_table_ready(self) -> bool:
        """Check if cache tables exist and have data."""
        try:
            agg_exists = self.con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'citing_aggregated'"
            ).fetchone()[0]
            detail_exists = self.con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'citation_details'"
            ).fetchone()[0]

            if agg_exists and detail_exists:
                agg_count = self.con.execute(
                    "SELECT COUNT(*) FROM citing_aggregated"
                ).fetchone()[0]
                detail_count = self.con.execute(
                    "SELECT COUNT(*) FROM citation_details"
                ).fetchone()[0]
                print(f"[OK] 引用缓存表已就绪")
                print(f"  聚合表: {agg_count:,} rows")
                print(f"  详细表: {detail_count:,} rows")
                return agg_count > 0 and detail_count > 0
            return False
        except Exception:
            return False

    def create_optimized_cache_table(self, force_recreate: bool = False) -> bool:
        """
        Create optimized cache tables from parquet files.

        Args:
            force_recreate: Force recreate even if tables exist

        Returns:
            True if successful
        """
        print("=== 开始创建优化的缓存表 ===")

        # Check existing tables
        agg_exists = self.con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'citing_aggregated'"
        ).fetchone()[0]
        detail_exists = self.con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'citation_details'"
        ).fetchone()[0]

        if agg_exists and detail_exists and not force_recreate:
            agg_count = self.con.execute(
                "SELECT COUNT(*) FROM citing_aggregated"
            ).fetchone()[0]
            detail_count = self.con.execute(
                "SELECT COUNT(*) FROM citation_details"
            ).fetchone()[0]
            print(
                f"发现现有缓存表：聚合表 {agg_count:,} rows，详细表 {detail_count:,} rows"
            )
            print("使用现有缓存表（使用 force_recreate=True 强制重建）")
            return True

        # Check data directory
        data_path = Path(self.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        parquet_files = list(data_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"Not foundparquet文件: {self.data_dir}")

        total_size = sum(f.stat().st_size for f in parquet_files)
        print(
            f"Found {len(parquet_files)} 个parquet文件，总大小: {total_size / (1024**3):.2f} GB"
        )

        start_time = time.time()

        # Step 1: Filter data
        print("\n步骤1: 过滤数据...")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE filtered_data AS
            SELECT 
                citingcorpusid,
                citedcorpusid,
                isinfluential,
                contexts,
                intents,
                citationid
            FROM read_parquet('{self.data_dir}/*.parquet')
            WHERE contexts IS NOT NULL 
            AND ARRAY_LENGTH(contexts) > 0
            AND contexts != ARRAY['null']
        """)

        filtered_count = self.con.execute(
            "SELECT COUNT(*) FROM filtered_data"
        ).fetchone()[0]
        total_count = self.con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{self.data_dir}/*.parquet')"
        ).fetchone()[0]

        print(f"原始数据: {total_count:,} rows")
        print(
            f"过滤后: {filtered_count:,} rows ({filtered_count / total_count * 100:.2f}%)"
        )

        # Step 2: Create aggregated table
        print("\n步骤2: 创建聚合表...")
        self._create_aggregated_table(filtered_count)

        # Step 3: Create indexes
        print("\n步骤3: 创建索引...")
        self.con.execute(
            "CREATE INDEX IF NOT EXISTS idx_citingcorpusid ON citing_aggregated(citingcorpusid)"
        )
        self.con.execute(
            "CREATE INDEX IF NOT EXISTS idx_citation_count ON citing_aggregated(citation_count)"
        )

        # Step 4: Create detail table
        print("\n步骤4: 创建详细表...")
        self.con.execute("""
            CREATE OR REPLACE TABLE citation_details AS
            SELECT 
                citingcorpusid,
                citedcorpusid,
                isinfluential,
                contexts,
                intents,
                citationid
            FROM filtered_data
            ORDER BY citingcorpusid, citedcorpusid
        """)

        detail_count = self.con.execute(
            "SELECT COUNT(*) FROM citation_details"
        ).fetchone()[0]
        print(f"详细表: {detail_count:,} rows")

        self.con.execute(
            "CREATE INDEX IF NOT EXISTS idx_detail_citing ON citation_details(citingcorpusid)"
        )
        self.con.execute(
            "CREATE INDEX IF NOT EXISTS idx_detail_cited ON citation_details(citedcorpusid)"
        )

        # Cleanup
        self.con.execute("DROP TABLE IF EXISTS filtered_data")

        elapsed = time.time() - start_time
        print(f"\n=== 缓存表创建完成！耗时: {elapsed:.2f}s ===")

        return True

    def _create_aggregated_table(self, total_rows: int):
        """Create aggregated table in batches to avoid memory issues."""
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

        self.con.execute("""
            CREATE OR REPLACE TEMP TABLE temp_aggregated (
                citingcorpusid BIGINT,
                citation_count BIGINT,
                unique_cited_count BIGINT,
                influential_count BIGINT,
                all_cited_papers VARCHAR,
                all_contexts VARCHAR,
                all_intents VARCHAR
            )
        """)

        batch_size = 5000000
        total_batches = (total_rows + batch_size - 1) // batch_size

        print(f"分批处理: {total_batches} 批次，每批 {batch_size:,} rows")

        self.con.execute("BEGIN TRANSACTION")

        try:
            for batch_num in range(total_batches):
                batch_start = batch_num * batch_size
                batch_end = (batch_num + 1) * batch_size - 1

                print(f"  处理批次 {batch_num + 1}/{total_batches}...")

                batch_result = self.con.execute(
                    f"""
                    SELECT 
                        citingcorpusid,
                        COUNT(*) as citation_count,
                        COUNT(DISTINCT citedcorpusid) as unique_cited_count,
                        SUM(CASE WHEN isinfluential = true THEN 1 ELSE 0 END) as influential_count,
                        STRING_AGG(DISTINCT citedcorpusid::VARCHAR, ',') as all_cited_papers,
                        STRING_AGG(DISTINCT contexts, ' | ') as all_contexts,
                        STRING_AGG(DISTINCT intents, ' | ') as all_intents
                    FROM filtered_data
                    WHERE citingcorpusid BETWEEN ? AND ?
                    GROUP BY citingcorpusid
                """,
                    [batch_start, batch_end],
                ).df()

                if not batch_result.empty:
                    self.con.execute(
                        "INSERT INTO temp_aggregated SELECT * FROM batch_result"
                    )

                if (batch_num + 1) % 10 == 0:
                    self.con.execute("COMMIT")
                    self.con.execute("BEGIN TRANSACTION")
                    print(f"    已提交 {batch_num + 1}/{total_batches}")

            self.con.execute("COMMIT")
        except Exception as e:
            self.con.execute("ROLLBACK")
            raise e

        # Copy and sort
        self.con.execute("INSERT INTO citing_aggregated SELECT * FROM temp_aggregated")

        agg_count = self.con.execute(
            "SELECT COUNT(*) FROM citing_aggregated"
        ).fetchone()[0]
        print(f"聚合表: {agg_count:,} rows")

        self.con.execute("DROP TABLE IF EXISTS temp_aggregated")

    def batch_query_contexts(
        self,
        citingcorpusid_list: List[int],
        include_details: bool = True,
        max_results_per_id: int = None,
    ) -> Dict:
        """
        Batch query citation contexts for citing papers.

        Args:
            citingcorpusid_list: List of citing paper corpus IDs
            include_details: Whether to include detailed citation info
            max_results_per_id: Maximum results per ID

        Returns:
            Dictionary with results keyed by corpus ID
        """
        if not citingcorpusid_list:
            return {}

        print("=== Batch Querying Context Info ===")
        print(f"Query ID count: {len(citingcorpusid_list)}")
        print(f"Include details: {include_details}")

        start_time = time.time()
        results = {}

        # Query aggregated info
        print("Querying aggregated info...")
        placeholders = ",".join(["?" for _ in citingcorpusid_list])

        agg_query = f"""
            SELECT 
                citingcorpusid,
                citation_count,
                unique_cited_count,
                influential_count
            FROM citing_aggregated
            WHERE citingcorpusid IN ({placeholders})
        """
        agg_results = self.con.execute(agg_query, citingcorpusid_list).df()

        # Initialize results
        for _, row in agg_results.iterrows():
            cid = row["citingcorpusid"]
            results[cid] = {
                "citation_count": int(row["citation_count"]),
                "unique_cited_count": int(row["unique_cited_count"]),
                "influential_count": int(row["influential_count"]),
                "citation_details": [],
            }

        # Query details if requested
        if include_details:
            print("Querying detailed citation info...")
            detail_query = f"""
                SELECT 
                    citingcorpusid,
                    citedcorpusid,
                    isinfluential,
                    contexts,
                    intents
                FROM citation_details
                WHERE citingcorpusid IN ({placeholders})
            """
            detail_results = self.con.execute(detail_query, citingcorpusid_list).df()

            print("Processing query results...")
            for _, row in detail_results.iterrows():
                citing_id = row["citingcorpusid"]
                if citing_id in results:
                    detail = {
                        "citedcorpusid": row["citedcorpusid"],
                        "isinfluential": bool(row["isinfluential"]),
                        "contexts": row["contexts"],
                        "intents": row["intents"],
                    }
                    if (
                        max_results_per_id is None
                        or len(results[citing_id]["citation_details"])
                        < max_results_per_id
                    ):
                        results[citing_id]["citation_details"].append(detail)

        elapsed = time.time() - start_time

        found_ids = len(results)
        total_results = sum(
            len(r.get("citation_details", [])) for r in results.values()
        )

        print(f"Query complete!")
        print(f"Found的ID: {found_ids}")
        print(f"Not found的ID: {len(citingcorpusid_list) - found_ids}")
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
        self, citedcorpusid_list: List[int], max_results_per_id: int = None
    ) -> Dict:
        """
        Query papers that cite the given papers.

        Args:
            citedcorpusid_list: List of cited paper corpus IDs
            max_results_per_id: Maximum results per ID

        Returns:
            Dictionary with results keyed by cited corpus ID
        """
        if not citedcorpusid_list:
            return {}

        print("=== Batch Querying Citing Papers (by cited paper) ===")
        print(f"Query cited paper ID count: {len(citedcorpusid_list)}")

        start_time = time.time()
        results = {}

        print("Querying citation info (by citedcorpusid)...")
        placeholders = ",".join(["?" for _ in citedcorpusid_list])

        query = f"""
            SELECT 
                citedcorpusid,
                citingcorpusid,
                isinfluential,
                contexts,
                intents
            FROM citation_details
            WHERE citedcorpusid IN ({placeholders})
            ORDER BY citedcorpusid, citingcorpusid
        """

        query_results = self.con.execute(query, citedcorpusid_list).df()

        print("Processing query results...")
        for _, row in query_results.iterrows():
            cited_id = row["citedcorpusid"]

            if cited_id not in results:
                results[cited_id] = {"citing_papers": []}

            citing_info = {
                "citingcorpusid": row["citingcorpusid"],
                "isinfluential": bool(row["isinfluential"]),
                "contexts": row["contexts"],
                "intents": row["intents"],
            }

            if (
                max_results_per_id is None
                or len(results[cited_id]["citing_papers"]) < max_results_per_id
            ):
                results[cited_id]["citing_papers"].append(citing_info)

        elapsed = time.time() - start_time

        found_ids = len(results)
        total_citing = sum(len(r["citing_papers"]) for r in results.values())

        print(f"Query complete!")
        print(f"Found的被引论文ID: {found_ids}")
        print(f"Not found的ID: {len(citedcorpusid_list) - found_ids}")
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
        print("\n=== 缓存表信息 ===")

        try:
            agg_stats = self.con.execute("""
                SELECT 
                    COUNT(*) as total_papers,
                    AVG(citation_count) as avg_citations,
                    MAX(citation_count) as max_citations,
                    SUM(citation_count) as total_citations
                FROM citing_aggregated
            """).fetchone()

            print(f"论文总数: {agg_stats[0]:,}")
            print(f"平均引用数: {agg_stats[1]:.2f}")
            print(f"最大引用数: {agg_stats[2]:,}")
            print(f"总引用数: {agg_stats[3]:,}")

            detail_count = self.con.execute(
                "SELECT COUNT(*) FROM citation_details"
            ).fetchone()[0]
            print(f"详细引用关系数: {detail_count:,}")

        except Exception as e:
            print(f"获取信息时出错: {e}")

    def close(self):
        """Close database connection."""
        try:
            if hasattr(self, "con") and self.con:
                self.con.close()
        except Exception as e:
            print(f"关闭连接时出错: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Alias for backward compatibility
OptimizedCacheBuilder = CitationCacheBuilder


def main():
    """Main function for CLI usage."""
    print("=== Citation contexts缓存构建器 ===")

    try:
        with CitationCacheBuilder() as builder:
            if builder.create_optimized_cache_table():
                builder.show_cache_info()

        print("\n[OK] 程序执行完成")

    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
