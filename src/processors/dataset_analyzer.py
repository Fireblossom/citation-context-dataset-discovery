"""
Dataset Analyzer - Analyzes extracted datasets by citation count.
"""

import json
from typing import List, Dict, Any, Optional
from collections import Counter

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def load_extraction_results(file_path: str) -> Dict[str, Any]:
    """Load dataset extraction results from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_citation_contexts(file_path: str) -> Dict[str, Any]:
    """Load citation contexts from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_citation_counts(citation_data: Dict[str, Any]) -> Dict[str, int]:
    """Analyze citation counts for each paper."""
    counts = {}

    if "papers" in citation_data:
        for paper in citation_data["papers"]:
            cited_id = paper.get("cited_corpusid") or paper.get("corpusid")
            if cited_id:
                citing_count = len(paper.get("citing_papers", []))
                citation_count = len(paper.get("citations", []))
                counts[str(cited_id)] = citing_count + citation_count

    return counts


def extract_datasets_with_metadata(
    extraction_results: Dict[str, Any],
    topic: str = None,
    validation_mode: str = "smart",
) -> List[Dict[str, Any]]:
    """Extract dataset information with metadata from extraction results."""
    datasets = []

    extractions = extraction_results.get("extractions", [])

    for extraction in extractions:
        extracted_datasets = extraction.get("datasets", [])

        for dataset_name in extracted_datasets:
            if not dataset_name or not dataset_name.strip():
                continue

            dataset_info = {
                "name": dataset_name.strip(),
                "context": extraction.get("context", ""),
                "citing_paper_id": extraction.get("citing_paper_id"),
                "cited_paper_id": extraction.get("cited_paper_id"),
                "confidence": extraction.get("confidence", 0.5),
                "citation_intent": extraction.get("citation_intent", []),
                "resource_type": extraction.get("resource_type", []),
                "description": extraction.get("descriptions", {}).get(dataset_name, ""),
            }

            datasets.append(dataset_info)

    return datasets


def normalize_dataset_name(name: str) -> str:
    """Normalize a dataset name for comparison."""
    import re

    normalized = name.lower().strip()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def merge_datasets_across_papers(
    datasets_info: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge datasets across papers by normalized name."""
    grouped = {}

    for ds in datasets_info:
        norm_name = normalize_dataset_name(ds["name"])

        if norm_name not in grouped:
            grouped[norm_name] = {
                "normalized_name": norm_name,
                "display_name": ds["name"],
                "name_variants": Counter(),
                "mentions": [],
                "citing_papers": set(),
                "cited_papers": set(),
                "total_confidence": 0.0,
            }

        group = grouped[norm_name]
        group["name_variants"][ds["name"]] += 1
        group["mentions"].append(ds)
        group["total_confidence"] += ds.get("confidence", 0.5)

        if ds.get("citing_paper_id"):
            group["citing_papers"].add(ds["citing_paper_id"])
        if ds.get("cited_paper_id"):
            group["cited_papers"].add(ds["cited_paper_id"])

    # Convert to list and select display name
    merged = []
    for norm_name, group in grouped.items():
        # Select most common variant as display name
        most_common = group["name_variants"].most_common(1)
        display_name = most_common[0][0] if most_common else group["display_name"]

        merged.append(
            {
                "name": display_name,
                "normalized_name": norm_name,
                "variants": list(group["name_variants"].keys()),
                "mention_count": len(group["mentions"]),
                "citing_paper_count": len(group["citing_papers"]),
                "cited_paper_count": len(group["cited_papers"]),
                "avg_confidence": group["total_confidence"] / len(group["mentions"])
                if group["mentions"]
                else 0,
                "mentions": group["mentions"],
            }
        )

    # Sort by mention count
    merged.sort(key=lambda x: x["mention_count"], reverse=True)

    return merged


def build_collapsed_table_rows(
    merged_groups: List[Dict[str, Any]],
    dataset_summaries: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Build collapsed table rows from merged groups."""
    rows = []

    for i, group in enumerate(merged_groups, 1):
        row = {
            "Rank": i,
            "Dataset": group["name"],
            "Mentions": group["mention_count"],
            "Citing Papers": group["citing_paper_count"],
            "Cited Papers": group["cited_paper_count"],
            "Avg Confidence": round(group["avg_confidence"], 2),
            "Variants": ", ".join(group["variants"][:5]),
        }

        # Add summary if available
        if dataset_summaries and group["name"] in dataset_summaries:
            row["Description"] = dataset_summaries[group["name"]]

        rows.append(row)

    return rows


def analyze_datasets(
    extraction_file: str,
    contexts_file: str,
    output_file: str = "analyzed_datasets.tsv",
    topic: str = "topic modeling",
    validation_mode: str = "smart",
) -> bool:
    """
    Main function to analyze datasets by citation count.

    Args:
        extraction_file: Path to dataset extraction results
        contexts_file: Path to citation contexts
        output_file: Output TSV file path
        topic: Research topic
        validation_mode: Validation mode

    Returns:
        True if successful
    """
    print("=== Dataset Analysis ===")
    print(f"Extraction results: {extraction_file}")
    print(f"Citation contexts: {contexts_file}")
    print(f"Output: {output_file}")

    try:
        # Load data
        extraction_results = load_extraction_results(extraction_file)
        citation_data = load_citation_contexts(contexts_file)

        # Analyze
        citation_counts = analyze_citation_counts(citation_data)
        print(f"Analyzed {len(citation_counts)}  papers citation counts")

        # Extract datasets
        datasets = extract_datasets_with_metadata(
            extraction_results, topic=topic, validation_mode=validation_mode
        )
        print(f"Extracted {len(datasets)}  dataset entries")

        # Merge
        merged = merge_datasets_across_papers(datasets)
        print(f"Merged into {len(merged)}  unique datasets")

        # Build table
        rows = build_collapsed_table_rows(merged)

        # Save
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv(output_file, sep="\t", index=False)

        print(f"[OK] Analysis complete! Results saved to: {output_file}")
        return True

    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return False
