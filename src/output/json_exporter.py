"""
JSON Export functions for citation contexts
"""

import json
import time
import os
from typing import Dict, Any

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.numpy_encoder import NumpyEncoder


def export_citation_contexts_to_json(
    results: Dict[str, Any],
    output_file: str,
    papers_cache_builder=None,
    description: str = "Citation Contexts",
    no_backup: bool = False,
) -> str:
    """
    Export citation context results to JSON file.

    Args:
        results: Query results from batch_query_contexts
        output_file: Output JSON file path
        papers_cache_builder: Optional PapersCacheBuilder for fetching titles
        description: Description for the output
        no_backup: Whether to skip backing up existing file

    Returns:
        Path to the created JSON file
    """
    print(f"\n正在将Citation contextsOutput到JSON文件: {output_file}")

    # Backup existing file
    if not no_backup and os.path.exists(output_file):
        backup_suffix = 0
        while True:
            if backup_suffix == 0:
                backup_path = f"{output_file}.backup"
            else:
                backup_path = f"{output_file}.backup_{backup_suffix}"

            if not os.path.exists(backup_path):
                os.rename(output_file, backup_path)
                print(f"Existing file backed up to: {backup_path}")
                break
            backup_suffix += 1

    # Build output structure
    output_data = {
        "description": description,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_papers": 0,
        "total_citations": 0,
        "papers": [],
    }

    # Process results
    for key in results.keys():
        if key == "query_stats":
            continue

        paper_data = results[key]
        citation_details = paper_data.get("citation_details", [])

        paper_entry = {
            "corpusid": key,
            "citation_count": len(citation_details),
            "citations": [],
        }

        for detail in citation_details:
            # Handle numpy arrays
            contexts = detail.get("contexts", [])
            if hasattr(contexts, "tolist"):
                contexts = contexts.tolist()

            intents = detail.get("intents", [])
            if hasattr(intents, "tolist"):
                intents = intents.tolist()

            citation_entry = {
                "cited_corpusid": detail.get("citedcorpusid"),
                "contexts": contexts if contexts else [],
                "intents": intents if intents else [],
                "is_influential": bool(detail.get("isinfluential", False)),
            }
            paper_entry["citations"].append(citation_entry)

        output_data["papers"].append(paper_entry)
        output_data["total_papers"] += 1
        output_data["total_citations"] += len(citation_details)

    # Add query stats
    if "query_stats" in results:
        output_data["query_stats"] = results["query_stats"]

    # Write JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(f"Data saved to: {output_file}")
    return output_file


def export_citing_papers_to_json(
    results: Dict[str, Any],
    output_file: str,
    papers_cache_builder=None,
    description: str = "Citing Papers",
    no_backup: bool = False,
) -> str:
    """
    Export 'papers that cite' results to JSON file.

    Args:
        results: Query results from batch_query_citing_papers
        output_file: Output JSON file path
        papers_cache_builder: Optional PapersCacheBuilder for fetching titles
        description: Description for the output
        no_backup: Whether to skip backing up existing file

    Returns:
        Path to the created JSON file
    """
    print(f"\nSaving results to: {output_file}")

    # Backup existing file
    if not no_backup and os.path.exists(output_file):
        backup_path = output_file.replace(".json", f"_backup_{int(time.time())}.json")
        os.rename(output_file, backup_path)
        print(f"Backed up existing file to: {backup_path}")

    # Collect all paper IDs for batch title lookup
    all_paper_ids = set()
    for key in results.keys():
        if key == "query_stats":
            continue
        all_paper_ids.add(key)
        for citing_info in results[key].get("citing_papers", []):
            all_paper_ids.add(citing_info["citingcorpusid"])

    # Batch fetch titles
    paper_titles = {}
    if papers_cache_builder and all_paper_ids:
        try:
            ids_list = list(all_paper_ids)
            batch_size = 1000
            for i in range(0, len(ids_list), batch_size):
                batch_ids = ids_list[i : i + batch_size]
                placeholders = ",".join(["?" for _ in batch_ids])
                query = f"""
                    SELECT corpusid, title
                    FROM papers_optimized
                    WHERE corpusid IN ({placeholders})
                """
                batch_results = papers_cache_builder.con.execute(
                    query, batch_ids
                ).fetchall()
                for row in batch_results:
                    corpusid, title = row
                    paper_titles[corpusid] = title if title else "Unknown Title"
        except Exception as e:
            print(f"[WARN] Error querying paper titles: {e}")

    # Build output structure
    output_data = {
        "description": description,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cited_papers": 0,
        "total_citing_records": 0,
        "papers": [],
    }

    for cited_id in results.keys():
        if cited_id == "query_stats":
            continue

        cited_title = paper_titles.get(cited_id, "Unknown Title")
        citing_papers_info = results[cited_id].get("citing_papers", [])

        paper_entry = {
            "cited_corpusid": cited_id,
            "cited_title": cited_title,
            "citing_count": len(citing_papers_info),
            "citing_papers": [],
        }

        for citing_info in citing_papers_info:
            citing_id = citing_info["citingcorpusid"]
            citing_title = paper_titles.get(citing_id, "Unknown Title")

            # Handle numpy arrays
            contexts = citing_info.get("contexts", [])
            if contexts is None:
                contexts_list = []
            elif hasattr(contexts, "tolist"):
                contexts_list = contexts.tolist()
            else:
                contexts_list = list(contexts) if contexts is not None else []

            intents = citing_info.get("intents", [])
            if intents is None:
                intents_list = []
            elif hasattr(intents, "tolist"):
                intents_list = intents.tolist()
            else:
                intents_list = list(intents) if intents is not None else []

            citing_entry = {
                "citing_corpusid": citing_id,
                "citing_title": citing_title,
                "contexts": contexts_list,
                "intents": intents_list,
                "isinfluential": bool(citing_info.get("isinfluential", False)),
            }
            paper_entry["citing_papers"].append(citing_entry)

        output_data["papers"].append(paper_entry)
        output_data["total_cited_papers"] += 1
        output_data["total_citing_records"] += len(citing_papers_info)

    # Add query stats
    if "query_stats" in results:
        output_data["query_stats"] = results["query_stats"]

    # Write JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(f"[OK] JSON file saved: {output_file}")
    print(
        f"   Total {output_data['total_cited_papers']}  cited papers, {output_data['total_citing_records']}  citation records"
    )

    return output_file
