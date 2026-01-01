"""
Global configuration for Citation Context Analysis project
"""

import os

# LLM Configuration
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_LLM_API_KEY = "EMPTY"
DEFAULT_LLM_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"
DEFAULT_LLM_TIMEOUT = 360.0
DEFAULT_MAX_CONCURRENT_REQUESTS = 500

# Semantic Scholar API
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# Default paths
DEFAULT_CACHE_DIR = "cache"
DEFAULT_DATA_DIR = "data"
DEFAULT_CITATION_DATA_DIR = "data/citations"

# Database files
CITATION_CACHE_DB = "optimized_cache.db"
PAPERS_CACHE_DB = "papers_optimized_cache_1.db"
