"""
Papers Cache Builder - Complete implementation

Creates and manages the papers_optimized DuckDB table from Semantic Scholar parquet files.
Supports incremental updates from new parquet/jsonl files.
"""

import duckdb
import time
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, PAPERS_CACHE_DB


class PapersCacheBuilder:
    """
    Optimized cache builder for paper metadata.

    Provides:
    - Table creation from parquet files
    - Fast corpus ID lookups
    - Batch query capabilities
    """

    def __init__(self, data_dir: str = None, cache_dir: str = None):
        """
        Initialize the papers cache builder.

        Args:
            data_dir: Path to paper data directories (papers/, abstracts/, paper-ids/)
            cache_dir: Path to store DuckDB cache database
        """
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR

        # Create cache directory
        Path(self.cache_dir).mkdir(exist_ok=True)

        # Connect to persistent database
        db_file = Path(self.cache_dir) / PAPERS_CACHE_DB
        try:
            self.con = duckdb.connect(str(db_file))
            print(f"Connected to database: {db_file}")

            # Test connection
            test_result = self.con.execute("SELECT 1").fetchone()
            if test_result[0] != 1:
                raise Exception("数据库连接测试失败")
            print("Database connection test successful")

        except Exception as e:
            print(f"连接数据库失败: {e}")
            raise

        # Setup optimizations
        self._setup_optimizations()

        # Data paths
        self.papers_dir = Path(self.data_dir) / "papers"
        self.abstracts_dir = Path(self.data_dir) / "abstracts"
        self.paper_ids_dir = Path(self.data_dir) / "paper-ids"

    def _setup_optimizations(self):
        """Configure DuckDB performance parameters."""
        self.con.execute("SET memory_limit='2TB'")
        self.con.execute("SET threads=32")
        self.con.execute("SET enable_progress_bar=true")
        self.con.execute("SET preserve_insertion_order=false")
        self.con.execute("SET temp_directory='/mlde/s2/temp'")

        try:
            self.con.execute("PRAGMA max_temp_directory_size='5TB'")
        except Exception:
            pass

        self.con.execute("SET checkpoint_threshold='1GB'")
        self.con.execute("SET wal_autocheckpoint='1GB'")
        print("DuckDB optimization parameters set")

    def check_data_files(self) -> Dict:
        """Check if data files exist and return their stats."""
        print("=== 检查数据文件状态 ===")

        result = {}

        # Check papers directory
        if self.papers_dir.exists():
            files = list(self.papers_dir.glob("*.parquet"))
            size = sum(f.stat().st_size for f in files)
            print(f"papers: {len(files)} 个文件, 总大小: {size / (1024**3):.2f} GB")
            result["papers"] = {"count": len(files), "size_gb": size / (1024**3)}
        else:
            raise FileNotFoundError(f"papers目录不存在: {self.papers_dir}")

        # Check abstracts directory
        if self.abstracts_dir.exists():
            files = list(self.abstracts_dir.glob("*.parquet"))
            size = sum(f.stat().st_size for f in files)
            print(f"abstracts: {len(files)} 个文件, 总大小: {size / (1024**3):.2f} GB")
            result["abstracts"] = {"count": len(files), "size_gb": size / (1024**3)}
        else:
            raise FileNotFoundError(f"abstracts目录不存在: {self.abstracts_dir}")

        # Check paper-ids directory
        if self.paper_ids_dir.exists():
            files = list(self.paper_ids_dir.glob("*.parquet"))
            size = sum(f.stat().st_size for f in files)
            print(f"paper-ids: {len(files)} 个文件, 总大小: {size / (1024**3):.2f} GB")
            result["paper_ids"] = {"count": len(files), "size_gb": size / (1024**3)}
        else:
            raise FileNotFoundError(f"paper-ids目录不存在: {self.paper_ids_dir}")

        return result

    def check_table_ready(self) -> bool:
        """Check if the papers_optimized table exists and has data."""
        try:
            exists = self.con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'papers_optimized'"
            ).fetchone()[0]

            if exists:
                count = self.con.execute(
                    "SELECT COUNT(*) FROM papers_optimized"
                ).fetchone()[0]
                print(f"[OK] Papers cache table ready")
                print(f"  Papers table: {count:,} rows")
                return count > 0
            return False
        except Exception:
            return False

    def create_optimized_papers_table(self, force_recreate: bool = False) -> bool:
        """
        Create the optimized papers table from parquet files.

        Args:
            force_recreate: If True, recreate even if table exists

        Returns:
            True if successful
        """
        print("\n=== 开始创建优化信息表 ===")

        # Check if table exists
        exists = self.con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'papers_optimized'"
        ).fetchone()[0]

        if exists and not force_recreate:
            count = self.con.execute(
                "SELECT COUNT(*) FROM papers_optimized"
            ).fetchone()[0]
            print(f"发现现有缓存表: {count:,} rows")
            print("使用现有缓存表（使用 force_recreate=True 强制重建）")
            return True

        start_time = time.time()

        # Step 1: Create base tables
        print("步骤1: 创建基础表...")

        print("创建papers基础表...")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE papers_base AS
            SELECT * FROM read_parquet('{self.papers_dir}/*.parquet', union_by_name=True)
        """)
        papers_count = self.con.execute("SELECT COUNT(*) FROM papers_base").fetchone()[
            0
        ]
        print(f"papers基础表: {papers_count:,} rows")

        print("创建abstracts基础表...")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE abstracts_base AS
            SELECT * FROM read_parquet('{self.abstracts_dir}/*.parquet', union_by_name=True)
        """)
        abstracts_count = self.con.execute(
            "SELECT COUNT(*) FROM abstracts_base"
        ).fetchone()[0]
        print(f"abstracts基础表: {abstracts_count:,} rows")

        print("创建paper_ids基础表...")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE paper_ids_base AS
            SELECT * FROM read_parquet('{self.paper_ids_dir}/*.parquet', union_by_name=True)
        """)
        paper_ids_count = self.con.execute(
            "SELECT COUNT(*) FROM paper_ids_base"
        ).fetchone()[0]
        print(f"paper_ids基础表: {paper_ids_count:,} rows")

        # Step 2: Create optimized table
        print("\n步骤2: 创建优化的整合表...")

        self.con.execute("""
            CREATE OR REPLACE TABLE papers_optimized (
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
            )
        """)

        # Create temp table with joins
        print("整合数据...")
        self.con.execute("""
            CREATE OR REPLACE TEMP TABLE temp_combined AS
            SELECT 
                p.corpusid,
                COALESCE(p.title, '') as title,
                p.year,
                COALESCE(p.venue, '') as venue,
                p.authors as authors,
                COALESCE(a.abstract, '') as abstract,
                COALESCE(pid.sha, '') as sha_id,
                p.externalids,
                p.url,
                p.publicationvenueid,
                p.referencecount,
                p.citationcount,
                p.influentialcitationcount,
                p.isopenaccess,
                CASE 
                    WHEN p.s2fieldsofstudy IS NOT NULL THEN
                        list_transform(p.s2fieldsofstudy, x -> struct_pack(category := lower(x.category), source := x.source))
                    ELSE p.s2fieldsofstudy
                END as s2fieldsofstudy,
                p.publicationtypes,
                p.publicationdate,
                p.journal
            FROM papers_base p
            LEFT JOIN abstracts_base a ON p.corpusid = a.corpusid
            LEFT JOIN paper_ids_base pid ON p.corpusid = pid.corpusid
            WHERE p.corpusid IS NOT NULL
        """)

        print("去重并插入...")
        self.con.execute("""
            INSERT INTO papers_optimized (
                corpusid, title, year, venue, authors, abstract, sha_id,
                externalids, url, publicationvenueid, referencecount, 
                citationcount, influentialcitationcount, isopenaccess,
                s2fieldsofstudy, publicationtypes, publicationdate, journal
            )
            SELECT 
                corpusid, title, year, venue, authors, abstract, sha_id,
                externalids, url, publicationvenueid, referencecount,
                citationcount, influentialcitationcount, isopenaccess,
                s2fieldsofstudy, publicationtypes, publicationdate, journal
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY corpusid ORDER BY title DESC) as rn
                FROM temp_combined
            ) ranked
            WHERE rn = 1
        """)

        # Cleanup
        self.con.execute("DROP TABLE IF EXISTS temp_combined")
        self.con.execute("DROP TABLE IF EXISTS papers_base")
        self.con.execute("DROP TABLE IF EXISTS abstracts_base")
        self.con.execute("DROP TABLE IF EXISTS paper_ids_base")

        # Verify
        final_count = self.con.execute(
            "SELECT COUNT(*) FROM papers_optimized"
        ).fetchone()[0]
        elapsed = time.time() - start_time

        print(f"\n=== 优化表创建完成！===")
        print(f"总行数: {final_count:,}")
        print(f"耗时: {elapsed:.2f}s")

        return True

    def batch_query_papers(self, corpusid_list: list, fields: list = None) -> dict:
        """Batch query paper metadata by corpus IDs."""
        if not corpusid_list:
            return {}

        try:
            field_str = ", ".join(fields) if fields else "*"
            placeholders = ",".join(["?" for _ in corpusid_list])

            query = f"""
                SELECT {field_str}
                FROM papers_optimized
                WHERE corpusid IN ({placeholders})
            """

            results = self.con.execute(query, corpusid_list).df()

            paper_dict = {}
            for _, row in results.iterrows():
                paper_dict[row["corpusid"]] = row.to_dict()

            return paper_dict

        except Exception as e:
            print(f"Error in batch query papers: {e}")
            return {}

    def get_paper_title(self, corpusid: int) -> str:
        """Get title for a single paper."""
        try:
            result = self.con.execute(
                "SELECT title FROM papers_optimized WHERE corpusid = ?", [corpusid]
            ).fetchone()
            return result[0] if result else "Unknown Title"
        except Exception:
            return "Unknown Title"

    def show_table_info(self):
        """Display table statistics."""
        print("\n=== 表信息 ===")

        try:
            columns = self.con.execute("DESCRIBE papers_optimized").fetchall()
            print(f"列数: {len(columns)}")

            stats = self.con.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN abstract != '' THEN 1 END) as with_abstract,
                    COUNT(CASE WHEN citationcount IS NOT NULL THEN 1 END) as with_citations
                FROM papers_optimized
            """).fetchone()

            print(f"Total papers: {stats[0]:,}")
            print(f"有摘要: {stats[1]:,} ({stats[1] / stats[0] * 100:.1f}%)")
            print(f"有引用数: {stats[2]:,} ({stats[2] / stats[0] * 100:.1f}%)")

        except Exception as e:
            print(f"显示信息时出错: {e}")

    def test_query_performance(self):
        """Test query performance."""
        print("\n=== 测试查询性能 ===")

        try:
            # Single query
            test_id = self.con.execute(
                "SELECT corpusid FROM papers_optimized LIMIT 1"
            ).fetchone()[0]

            start = time.time()
            self.con.execute(
                "SELECT * FROM papers_optimized WHERE corpusid = ?", [test_id]
            ).df()
            print(f"单条查询: {time.time() - start:.4f}s")

            # Batch query
            test_ids = (
                self.con.execute(
                    "SELECT corpusid FROM papers_optimized ORDER BY RANDOM() LIMIT 1000"
                )
                .df()["corpusid"]
                .tolist()
            )

            start = time.time()
            placeholders = ",".join(["?" for _ in test_ids])
            self.con.execute(
                f"SELECT corpusid, title, year FROM papers_optimized WHERE corpusid IN ({placeholders})",
                test_ids,
            ).df()
            elapsed = time.time() - start
            print(f"批量查询1000条: {elapsed:.4f}s ({elapsed / 1000:.6f}s/条)")

        except Exception as e:
            print(f"性能测试出错: {e}")

    def export_table(self, output_file: str = None) -> bool:
        """Export table to parquet file."""
        if output_file is None:
            output_file = Path(self.cache_dir) / "papers_optimized.parquet"
        else:
            output_file = Path(output_file)

        print(f"\n导出到: {output_file}")

        try:
            self.con.execute(
                f"COPY papers_optimized TO '{output_file}' (FORMAT PARQUET)"
            )
            size_mb = output_file.stat().st_size / (1024**2)
            print(f"导出成功！大小: {size_mb:.2f} MB")
            return True
        except Exception as e:
            print(f"导出失败: {e}")
            return False

    # ==================== Incremental Update Methods ====================

    def _init_processed_files_table(self):
        """Initialize the processed files tracking table."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS processed_files (
                filename VARCHAR PRIMARY KEY,
                source_type VARCHAR,
                row_count BIGINT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def get_processed_files(self, source_type: str = None) -> set:
        """Get set of already processed filenames."""
        self._init_processed_files_table()

        if source_type:
            result = self.con.execute(
                "SELECT filename FROM processed_files WHERE source_type = ?",
                [source_type],
            ).fetchall()
        else:
            result = self.con.execute("SELECT filename FROM processed_files").fetchall()

        return {row[0] for row in result}

    def get_new_files(self, directory: Path, source_type: str) -> List[Path]:
        """Find parquet files that haven't been processed yet."""
        if not directory.exists():
            return []

        all_files = list(directory.glob("*.parquet"))
        processed = self.get_processed_files(source_type)

        new_files = [f for f in all_files if f.name not in processed]

        print(
            f"{source_type}: Total {len(all_files)} 个文件, 已处理 {len(processed)}, 新增 {len(new_files)}"
        )

        return new_files

    def mark_file_processed(self, filename: str, source_type: str, row_count: int = 0):
        """Mark a file as processed."""
        self._init_processed_files_table()
        self.con.execute(
            """
            INSERT OR REPLACE INTO processed_files (filename, source_type, row_count, processed_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """,
            [filename, source_type, row_count],
        )

    def incremental_update(self, from_jsonl_dir: str = None) -> bool:
        """
        Perform incremental update: only process new files.

        Args:
            from_jsonl_dir: Optional directory with new jsonl.gz files to convert first

        Returns:
            True if successful
        """
        print("\n=== 增量更新模式 ===")

        # Step 1: Convert jsonl.gz if provided
        if from_jsonl_dir:
            print(f"\n步骤1: 转换 jsonl.gz 文件...")
            from src.utils.jsonl_converter import convert_jsonl_to_parquet

            # Convert to respective directories
            jsonl_path = Path(from_jsonl_dir)

            papers_jsonl = jsonl_path / "papers"
            if papers_jsonl.exists():
                convert_jsonl_to_parquet(str(papers_jsonl), str(self.papers_dir))

            abstracts_jsonl = jsonl_path / "abstracts"
            if abstracts_jsonl.exists():
                convert_jsonl_to_parquet(str(abstracts_jsonl), str(self.abstracts_dir))

            paper_ids_jsonl = jsonl_path / "paper-ids"
            if paper_ids_jsonl.exists():
                convert_jsonl_to_parquet(str(paper_ids_jsonl), str(self.paper_ids_dir))

        # Step 2: Find new parquet files
        print("\n步骤2: 检测新文件...")
        new_papers = self.get_new_files(self.papers_dir, "papers")
        new_abstracts = self.get_new_files(self.abstracts_dir, "abstracts")
        new_paper_ids = self.get_new_files(self.paper_ids_dir, "paper_ids")

        if not new_papers and not new_abstracts and not new_paper_ids:
            print("[OK] 没有新文件需要处理")
            return True

        # Step 3: Ensure base table exists
        if not self.check_table_ready():
            print("缓存表不存在，执行完整构建...")
            return self.create_optimized_papers_table()

        # Step 4: Merge new data
        print("\n步骤3: 增量合并新数据...")
        start_time = time.time()

        total_new_rows = 0

        for parquet_file in new_papers:
            try:
                # Read new data
                new_count = self.con.execute(f"""
                    SELECT COUNT(*) FROM read_parquet('{parquet_file}')
                """).fetchone()[0]

                # Insert or replace
                self.con.execute(f"""
                    INSERT OR REPLACE INTO papers_optimized 
                    SELECT 
                        corpusid,
                        COALESCE(title, '') as title,
                        year,
                        COALESCE(venue, '') as venue,
                        authors,
                        '' as abstract,
                        '' as sha_id,
                        externalids,
                        url,
                        publicationvenueid,
                        referencecount,
                        citationcount,
                        influentialcitationcount,
                        isopenaccess,
                        CASE 
                            WHEN s2fieldsofstudy IS NOT NULL THEN
                                list_transform(s2fieldsofstudy, x -> struct_pack(category := lower(x.category), source := x.source))
                            ELSE s2fieldsofstudy
                        END as s2fieldsofstudy,
                        publicationtypes,
                        publicationdate,
                        journal,
                        CURRENT_TIMESTAMP as created_at
                    FROM read_parquet('{parquet_file}')
                    WHERE corpusid IS NOT NULL
                """)

                self.mark_file_processed(parquet_file.name, "papers", new_count)
                total_new_rows += new_count
                print(f"  [OK] {parquet_file.name}: {new_count:,} rows")

            except Exception as e:
                print(f"  [ERROR] {parquet_file.name}: {e}")

        # Update abstracts for new papers
        for parquet_file in new_abstracts:
            try:
                self.con.execute(f"""
                    UPDATE papers_optimized SET abstract = a.abstract
                    FROM (SELECT corpusid, abstract FROM read_parquet('{parquet_file}')) a
                    WHERE papers_optimized.corpusid = a.corpusid
                """)
                self.mark_file_processed(parquet_file.name, "abstracts")
                print(f"  [OK] abstracts: {parquet_file.name}")
            except Exception as e:
                print(f"  [ERROR] abstracts {parquet_file.name}: {e}")

        # Update sha_ids
        for parquet_file in new_paper_ids:
            try:
                self.con.execute(f"""
                    UPDATE papers_optimized SET sha_id = p.sha
                    FROM (SELECT corpusid, sha FROM read_parquet('{parquet_file}')) p
                    WHERE papers_optimized.corpusid = p.corpusid
                """)
                self.mark_file_processed(parquet_file.name, "paper_ids")
                print(f"  [OK] paper_ids: {parquet_file.name}")
            except Exception as e:
                print(f"  [ERROR] paper_ids {parquet_file.name}: {e}")

        elapsed = time.time() - start_time
        final_count = self.con.execute(
            "SELECT COUNT(*) FROM papers_optimized"
        ).fetchone()[0]

        print(f"\n=== 增量更新完成 ===")
        print(f"新增/更新行数: {total_new_rows:,}")
        print(f"总行数: {final_count:,}")
        print(f"耗时: {elapsed:.2f}s")

        return True

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
PapersOptimizedCacheBuilder = PapersCacheBuilder


def main():
    """Main function for CLI usage."""
    print("=== 论文信息优化缓存构建器 ===")

    try:
        with PapersCacheBuilder() as builder:
            builder.check_data_files()

            if builder.create_optimized_papers_table():
                builder.show_table_info()
                builder.test_query_performance()
                builder.export_table()

        print("\n[OK] 程序执行完成")

    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
