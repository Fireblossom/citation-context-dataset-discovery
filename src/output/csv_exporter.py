"""
CSV Export functions for citation contexts
"""

import csv
import os
from typing import Dict, Any


def export_contexts_to_csv(
    results: Dict[str, Any], papers_cache_builder, csv_output_file: str
) -> str:
    """
    Export citation contexts to CSV for auditing.

    CSV format: citing_paper_title, cited_paper_title, citation_context

    Args:
        results: Query results from batch_query_contexts
        papers_cache_builder: PapersCacheBuilder for fetching titles
        csv_output_file: Output CSV file path

    Returns:
        Path to the created CSV file
    """
    print(f"\nGenerating CSV audit file: {csv_output_file}")

    # Collect all paper IDs for batch title lookup
    all_citing_ids = set()
    all_cited_ids = set()

    for key in results.keys():
        if key == "query_stats":
            continue
        all_citing_ids.add(key)
        citation_details = results[key].get("citation_details", [])
        for detail in citation_details:
            cited_id = detail.get("citedcorpusid")
            if cited_id is not None:
                all_cited_ids.add(cited_id)

    # Batch fetch titles
    all_ids = list(all_citing_ids | all_cited_ids)
    paper_titles = {}

    if papers_cache_builder and all_ids:
        try:
            batch_size = 1000
            for i in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[i : i + batch_size]
                # Convert numpy types to native Python int for DuckDB compatibility
                batch_ids = [int(x) if hasattr(x, "item") else x for x in batch_ids]
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

    # Build CSV rows
    rows = []
    for citing_id in results.keys():
        if citing_id == "query_stats":
            continue

        citing_title = paper_titles.get(citing_id, "Unknown Title")
        citation_details = results[citing_id].get("citation_details", [])

        for detail in citation_details:
            cited_id = detail.get("citedcorpusid")
            cited_title = paper_titles.get(cited_id, "Unknown Title")
            contexts = detail.get("contexts", [])

            # Handle contexts that might be numpy arrays or lists
            if hasattr(contexts, "tolist"):
                contexts = contexts.tolist()
            elif contexts is None:
                contexts = []

            for ctx in contexts:
                if ctx and str(ctx).strip() and str(ctx).lower() != "null":
                    clean_ctx = str(ctx).replace("\n", " ").replace("\r", " ").strip()
                    rows.append([citing_title, cited_title, clean_ctx])

    # Write CSV
    try:
        with open(csv_output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["citing_paper_title", "cited_paper_title", "citation_context"]
            )
            writer.writerows(rows)
        print(f"[OK] CSV audit file saved: {csv_output_file} ({len(rows)} 条记录)")
    except Exception as e:
        print(f"[ERROR] Failed to save CSV file: {e}")

    return csv_output_file
