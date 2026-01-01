#!/usr/bin/env python3
"""
Query Citation Contexts - Main CLI Script

This is the refactored main entry point for querying citation contexts.
It provides the same functionality as the original 2_query_research_topic_contexts.py.
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import SEMANTIC_SCHOLAR_API_KEY
from src.cache import CitationCacheBuilder, PapersCacheBuilder
from src.clients import SemanticScholarClient
from src.output import (
    export_citation_contexts_to_json,
    export_citing_papers_to_json,
    export_contexts_to_csv,
)
from src.utils import create_output_filename


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Query citation contexts for papers and export to file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic query
  python query_contexts.py --query "topic model" --limit 100
  
  # Specify context type
  python query_contexts.py --query "topic model" --limit 200 --context-type both
  
  # Enable second-level analysis
  python query_contexts.py --query "topic model" --limit 100 --second
  
  # Enable LLM pre-filtering
  python query_contexts.py --query "topic model" --limit 100 --llm-prefilter
        """,
    )

    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="topic modeling",
        help="Search query (default: topic modeling)",
    )

    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=100,
        help="Number of papers to fetch (default: 100)",
    )

    parser.add_argument(
        "--context-type",
        type=str,
        choices=["citing", "cited", "both"],
        default="citing",
        help="Context type: citing, cited, or both (default: citing)",
    )

    parser.add_argument(
        "--second",
        action="store_true",
        default=False,
        help="Enable second-level citation analysis",
    )

    parser.add_argument(
        "--llm-prefilter",
        action="store_true",
        default=False,
        help="Use LLM to pre-filter paper relevance",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: current directory)",
    )

    parser.add_argument(
        "--api-key", type=str, default=None, help="Semantic Scholar API key"
    )

    parser.add_argument(
        "--timestamp",
        action="store_true",
        default=False,
        help="Include timestamp in output filename",
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        default=False,
        help="Do not backup existing files",
    )

    parser.add_argument(
        "--merge",
        action="store_true",
        default=False,
        help="Merge all results into a single JSON file (all_contexts.json)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    print("=" * 60)
    print("Citation Context Query Tool (Refactored)")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"Limit: {args.limit}")
    print(f"Context type: {args.context_type}")
    print(f"Second-level analysis: {'Enabled' if args.second else 'Disabled'}")
    print(f"LLM pre-filter: {'Enabled' if args.llm_prefilter else 'Disabled'}")
    print("=" * 60)

    try:
        # Initialize cache builders
        print("\nInitializing cache...")
        citation_cache = CitationCacheBuilder()
        papers_cache = PapersCacheBuilder()

        # Check if papers cache is ready
        if not papers_cache.check_table_ready():
            print("[ERROR] Papers cache table not available")
            return False

        # Initialize API client
        api_key = args.api_key or SEMANTIC_SCHOLAR_API_KEY
        client = SemanticScholarClient(api_key=api_key)

        # Search for papers
        print("\nSearching for papers...")
        papers = client.search_papers(query=args.query, limit=args.limit)

        if not papers:
            print("[ERROR] No papers found")
            return False

        # LLM prefilter if enabled
        if args.llm_prefilter:
            from src.processors import llm_prefilter_papers

            papers = llm_prefilter_papers(papers, args.query)
            if not papers:
                print("[ERROR] No papers after pre-filtering")
                return False

        # Extract corpus IDs
        corpus_ids = [p.get("corpusId") for p in papers if p.get("corpusId")]
        print(f"Extracted {len(corpus_ids)} valid corpusIds")

        # Query citation contexts based on context type
        output_base = create_output_filename(
            args.query, args.context_type, args.limit, args.timestamp
        )
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_base = os.path.join(args.output_dir, output_base)

        citing_results = None
        cited_results = None

        # === Citing direction ===
        if args.context_type in ["citing", "both"]:
            print(f"\nQuerying citation contexts for {len(corpus_ids)} papers...")
            citing_results = citation_cache.batch_query_contexts(
                citingcorpusid_list=corpus_ids, include_details=True
            )

            if citing_results:
                output_file = (
                    output_base
                    if args.context_type == "citing"
                    else output_base.replace(".json", "_citing.json")
                )
                export_citation_contexts_to_json(
                    citing_results,
                    output_file,
                    papers_cache,
                    f"{args.query} - Citing Contexts",
                    args.no_backup,
                )
                export_contexts_to_csv(
                    citing_results,
                    papers_cache,
                    output_file.replace(".json", "_audit.csv"),
                )

        # === Cited direction ===
        if args.context_type in ["cited", "both"]:
            print(f"\nQuerying who cited {len(corpus_ids)} search results...")
            cited_results = citation_cache.batch_query_citing_papers(
                citedcorpusid_list=corpus_ids
            )

            if cited_results:
                output_file = (
                    output_base
                    if args.context_type == "cited"
                    else output_base.replace(".json", "_cited_by.json")
                )
                export_citing_papers_to_json(
                    cited_results,
                    output_file,
                    papers_cache,
                    f"{args.query} - Cited By",
                    args.no_backup,
                )

        # === Second level analysis ===
        if args.second and citing_results:
            print("\n=== Starting Second-Level Citation Analysis ===")

            # Citing direction: C1 -> C2
            if args.context_type in ["citing", "both"]:
                print("\n--- Citing Direction Second-Level Analysis ---")
                # Collect all cited papers from first level
                second_level_ids = set()
                for key in citing_results.keys():
                    if key == "query_stats":
                        continue
                    for detail in citing_results[key].get("citation_details", []):
                        cited_id = detail.get("citedcorpusid")
                        # Use try/except to handle pandas NA values
                        try:
                            if (
                                cited_id is not None
                                and cited_id == cited_id
                                and cited_id not in corpus_ids
                            ):
                                second_level_ids.add(cited_id)
                        except (TypeError, ValueError):
                            pass  # Skip NA values

                print(f"Found {len(second_level_ids)} unique cited papers (C1)")

                if second_level_ids:
                    # LLM prefilter for second level if enabled
                    if args.llm_prefilter:
                        print("Fetching paper details for LLM pre-filtering...")
                        # Convert to list and ensure native Python int types
                        id_list = [
                            int(x) if hasattr(x, "item") else x
                            for x in second_level_ids
                        ]

                        # Batch query papers from cache
                        paper_infos = papers_cache.batch_query_papers(
                            id_list, fields=["corpusid", "title", "abstract"]
                        )

                        second_level_papers = []
                        for cid in id_list:
                            if cid in paper_infos:
                                info = paper_infos[cid]
                                second_level_papers.append(
                                    {
                                        "corpusId": cid,
                                        "title": info.get("title", "") or "",
                                        "abstract": info.get("abstract", "") or "",
                                    }
                                )

                        print(
                            f"Retrieved details for {len(second_level_papers)} papers from cache"
                        )

                        from src.processors import llm_prefilter_papers

                        filtered_papers = llm_prefilter_papers(
                            second_level_papers, args.query
                        )
                        second_level_ids = {p["corpusId"] for p in filtered_papers}
                        print(
                            f"After LLM filtering: {len(second_level_ids)} relevant papers"
                        )

                    print("Querying second-level citation contexts (C1 → C2)...")
                    second_citing_results = citation_cache.batch_query_contexts(
                        citingcorpusid_list=list(second_level_ids), include_details=True
                    )

                    if second_citing_results:
                        second_output = output_base.replace(".json", "_2nd_citing.json")
                        export_citation_contexts_to_json(
                            second_citing_results,
                            second_output,
                            papers_cache,
                            f"{args.query} - 2nd Level Citing",
                            args.no_backup,
                        )
                        stats = second_citing_results.get("query_stats", {})
                        print(
                            f"Second-level (Citing): {stats.get('found_ids', 0)} papers with contexts, {stats.get('total_results', 0)} records"
                        )

            # Cited direction: who cites the papers that cite P
            if args.context_type in ["cited", "both"] and cited_results:
                print("\n--- Cited Direction Second-Level Analysis ---")
                # Collect all citing papers from first level
                first_level_citing_ids = set()
                for key in cited_results.keys():
                    if key == "query_stats":
                        continue
                    for citing_info in cited_results[key].get("citing_papers", []):
                        citing_id = citing_info.get("citingcorpusid")
                        # Use try/except to handle pandas NA values
                        try:
                            if citing_id is not None and citing_id == citing_id:
                                first_level_citing_ids.add(citing_id)
                        except (TypeError, ValueError):
                            pass  # Skip NA values

                print(
                    f"Found {len(first_level_citing_ids)} papers citing the search results (X)"
                )

                if first_level_citing_ids:
                    # LLM prefilter for second level if enabled
                    if args.llm_prefilter:
                        print("Fetching paper details for LLM pre-filtering...")
                        # Convert to list and ensure native Python int types
                        id_list = [
                            int(x) if hasattr(x, "item") else x
                            for x in first_level_citing_ids
                        ]

                        # Batch query papers from cache
                        paper_infos = papers_cache.batch_query_papers(
                            id_list, fields=["corpusid", "title", "abstract"]
                        )

                        citing_papers_for_filter = []
                        for cid in id_list:
                            if cid in paper_infos:
                                info = paper_infos[cid]
                                citing_papers_for_filter.append(
                                    {
                                        "corpusId": cid,
                                        "title": info.get("title", "") or "",
                                        "abstract": info.get("abstract", "") or "",
                                    }
                                )

                        print(
                            f"Retrieved details for {len(citing_papers_for_filter)} papers from cache"
                        )

                        from src.processors import llm_prefilter_papers

                        filtered_citing = llm_prefilter_papers(
                            citing_papers_for_filter, args.query
                        )
                        first_level_citing_ids = {
                            p["corpusId"] for p in filtered_citing
                        }
                        print(
                            f"After LLM filtering: {len(first_level_citing_ids)} relevant papers"
                        )

                    print("Querying second-level: who cites X...")
                    second_cited_results = citation_cache.batch_query_citing_papers(
                        citedcorpusid_list=list(first_level_citing_ids)
                    )

                    if second_cited_results:
                        second_output = output_base.replace(".json", "_2nd_cited.json")
                        export_citing_papers_to_json(
                            second_cited_results,
                            second_output,
                            papers_cache,
                            f"{args.query} - 2nd Level Cited By",
                            args.no_backup,
                        )
                        stats = second_cited_results.get("query_stats", {})
                        print(
                            f"Second-level (Cited): {stats.get('found_ids', 0)} papers cited, {stats.get('total_citing_papers', 0)} citation records"
                        )

        # === Merge all results if requested ===
        if args.merge:
            print("\n=== Merging all contexts into a single file ===")
            import json

            all_contexts = []
            merge_output = (
                os.path.join(args.output_dir, "all_contexts.json")
                if args.output_dir
                else "all_contexts.json"
            )

            # Helper function to extract contexts from JSON files
            def extract_contexts_from_file(filepath):
                contexts = []
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Handle citing format (papers with citations)
                    if "papers" in data:
                        for paper in data["papers"]:
                            citing_id = paper.get("corpusid")
                            for citation in paper.get("citations", []):
                                cited_id = citation.get("cited_corpusid")
                                for ctx in citation.get("contexts", []):
                                    if ctx and str(ctx).strip():
                                        contexts.append(
                                            {
                                                "context": str(ctx),
                                                "citing_paper_id": citing_id,
                                                "cited_paper_id": cited_id,
                                                "source": os.path.basename(filepath),
                                            }
                                        )
                            for citation in paper.get("citing_papers", []):
                                citing_id = citation.get("citing_corpusid")
                                for ctx in citation.get("contexts", []):
                                    if ctx and str(ctx).strip():
                                        contexts.append(
                                            {
                                                "context": str(ctx),
                                                "citing_paper_id": citing_id,
                                                "cited_paper_id": paper.get("corpusid"),
                                                "source": os.path.basename(filepath),
                                            }
                                        )
                except Exception as e:
                    print(f"[WARN] Error reading {filepath}: {e}")
                return contexts

            # Find all JSON files in output directory
            search_dir = args.output_dir if args.output_dir else "."
            for filename in os.listdir(search_dir):
                if filename.endswith(".json") and filename != "all_contexts.json":
                    filepath = os.path.join(search_dir, filename)
                    contexts = extract_contexts_from_file(filepath)
                    all_contexts.extend(contexts)
                    print(f"  {filename}: {len(contexts)} contexts")

            # Save merged file
            merged_data = {
                "query": args.query,
                "total_contexts": len(all_contexts),
                "contexts": all_contexts,
            }

            with open(merge_output, "w", encoding="utf-8") as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)

            print(f"\n[OK] Merged {len(all_contexts)} contexts into: {merge_output}")

        print("\n=== Query Complete ===")

        # Clean up
        citation_cache.close()
        papers_cache.close()

        return True

    except Exception as e:
        print(f"[ERROR] Execution error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
