# Citation-Context Dataset Discovery

A literature-driven framework for discovering datasets from citation contexts in scientific papers. This approach enables dataset retrieval grounded in actual research use rather than metadata availability.

## Overview

This framework treats scientific publications as semantic bridges between research questions and data resources. By mining citation contexts, it discovers datasets that have been used, modified, or evaluated in prior research—providing citation-verified results with rich semantic context.

**Key Features:**
- Scalable citation-context retrieval from Semantic Scholar Academic Graph (S2AG)
- LLM-based dataset mention extraction with citation-aware quality signals
- Deterministic entity resolution preserving provenance
- Multi-domain applicability across Fields of Science

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Research Query │────▶│ Citation Context │────▶│ Dataset Entities│
└─────────────────┘     │    Retrieval     │     │   with Links    │
                        └──────────────────┘     └─────────────────┘
                               │                        ▲
                               ▼                        │
                        ┌──────────────────┐     ┌─────────────────┐
                        │  LLM Extraction  │────▶│Entity Resolution│
                        └──────────────────┘     └─────────────────┘
```

### Prerequisites

- Python 3.9+
- Access to Semantic Scholar API (optional: set `SEMANTIC_SCHOLAR_API_KEY`)
- vLLM server running locally (default: `http://localhost:8000/v1`)

## Usage

The pipeline consists of sequential stages:

### Stage 1: Build Cache

Build optimized cache tables from S2AG data. The builders read parquet files from
`<data-dir>/{papers,abstracts,paper-ids,citations}/`; raw S2AG downloads
(`*.jsonl.gz`, full release plus incremental update files in the same directories)
can be converted on the way with `--convert-from`:

```bash
# Convert raw S2AG jsonl.gz to parquet, then build both caches
python scripts/1_build_cache.py --all --convert-from /path/to/s2ag/raw --data-dir data --workers 32

python scripts/1_build_cache.py --papers --data-dir /path/to/s2ag/data
python scripts/1_build_cache.py --citations --data-dir /path/to/s2ag/data

# Incremental update
python scripts/1_build_cache.py --papers --incremental
```

When a release and later update files are present, every paper, abstract and
paper ID is taken from the newest file that contains it (S2AG file names start
with a timestamp). For citations the newest version *with contexts* wins: S2AG
publishes each citations release as two exports, and the later one repeats the
citations with the contexts of publisher-restricted papers removed.

The full build is memory hungry (on the order of 1 TB of RAM for the complete
S2AG citations dataset). DuckDB's memory limit, thread count and spill directory
are derived from the Slurm allocation / cgroup / host and can be overridden with
`--memory-limit`, `--threads`, `--temp-dir`, `--max-temp-size` or the
`CCDD_DUCKDB_*` environment variables. A Slurm job only needs CPUs and memory:

```bash
sbatch -c 64 --mem=1500G -t 2-00:00:00 --wrap \
  "python scripts/1_build_cache.py --all --convert-from /path/to/s2ag/raw --data-dir data --workers 60"
```

### Stage 2: Query Citation Contexts

Retrieve citation contexts for a research question:

```bash
python scripts/2_query_contexts.py \
  --topic "Multi-modal Knowledge Graph Reasoning" \
  --output contexts.json
```

### Stage 3: Extract Dataset Mentions

Extract dataset entities from citation contexts using LLM:

```bash
python scripts/3_extract_datasets.py \
  --input contexts.json \
  --output datasets.json \
  --topic "Multi-modal Knowledge Graph Reasoning"
```

### Stage 4: Analyze Datasets

Analyze and rank datasets by citation count:

```bash
python scripts/4_analyze_datasets.py \
  --extraction datasets.json \
  --contexts contexts.json \
  --output analyzed_datasets.tsv
```

### Stage 5: Deduplicate Tables

Deduplicate merged analysis tables:

```bash
python scripts/5_dedup_tables.py --dirs ./output
```

## Configuration

Edit `src/config.py` to customize:

```python
# LLM Configuration
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# Semantic Scholar API
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
```

## Project Structure

```
├── scripts/           # CLI entry points
│   ├── 1_build_cache.py
│   ├── 2_query_contexts.py
│   ├── 3_extract_datasets.py
│   ├── 4_analyze_datasets.py
│   └── 5_dedup_tables.py
├── src/
│   ├── cache/         # DuckDB cache builders
│   ├── clients/       # API clients (Semantic Scholar)
│   ├── extractors/    # LLM-based extraction
│   ├── processors/    # Dataset analysis, LLM filtering
│   ├── output/        # JSON/CSV exporters
│   └── utils/         # Utilities
└── requirements.txt
```

## Acknowledgments

- [Semantic Scholar Academic Graph](https://www.semanticscholar.org/product/api) for the citation data
- [vLLM](https://github.com/vllm-project/vllm) for efficient LLM inference
