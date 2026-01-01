"""
Dedup Utilities - Functions for deduplicating merged analysis tables.
"""

import os
import re
import shutil
from typing import Any, List, Optional

import pandas as pd


def extract_doi_from_text(text: str) -> Optional[str]:
    """Extract DOI from a text string."""
    if not text:
        return None
    s = str(text).strip()
    m = re.match(r"https?://doi\.org/(.+)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"10\.\d{4,9}/\S+", s)
    if m2:
        return m2.group(0).strip().rstrip(").,;")
    return None


def dedup_cell_urls_by_doi(cell_text: Any) -> str:
    """Deduplicate URLs in a cell by DOI."""
    if cell_text is None or str(cell_text).strip() == "" or str(cell_text) == "nan":
        return ""
    items = [p.strip() for p in str(cell_text).split(",") if p.strip()]
    seen = set()
    out_items: List[str] = []
    for it in items:
        m_suffix = re.search(r"\s*\(\+\d+\)$", it)
        suffix = m_suffix.group(0) if m_suffix else ""
        core = it[: it.rfind(suffix)].strip() if suffix else it
        doi = extract_doi_from_text(core)
        if doi:
            key = f"doi:{doi.lower()}"
        else:
            norm = core.rstrip(").,;").strip().lower()
            key = f"url:{norm}"
        if key in seen:
            continue
        seen.add(key)
        out_items.append((core + suffix).strip())
    return ", ".join(out_items)


def process_file(path: str, dry_run: bool = False) -> bool:
    """Process a single TSV file for deduplication."""
    try:
        df = pd.read_csv(path, sep="\t", header=0)
    except Exception as e:
        print(f"[ERROR] Failed to read: {path}: {e}")
        return False

    changed = False
    if "Citing Article" in df.columns:
        new_series = df["Citing Article"].apply(dedup_cell_urls_by_doi)
        if not new_series.equals(df["Citing Article"]):
            df["Citing Article"] = new_series
            changed = True
    if "Citied Article" in df.columns:
        new_series = df["Citied Article"].apply(dedup_cell_urls_by_doi)
        if not new_series.equals(df["Citied Article"]):
            df["Citied Article"] = new_series
            changed = True

    if not changed:
        print(f"[SKIP] No changes: {path}")
        return True

    if dry_run:
        print(f"[dry-run] Will update: {path}")
        return True

    bak = path + ".bak"
    try:
        if not os.path.exists(bak):
            shutil.copyfile(path, bak)
    except Exception:
        pass
    try:
        df.to_csv(path, sep="\t", index=False)
        print(f"[OK] Deduplicated: {path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write: {path}: {e}")
        return False


def walk_and_process(dirs: List[str], dry_run: bool = False) -> None:
    """Walk directories and process all merged analysis tables."""
    total = 0
    updated_or_ok = 0
    for base in dirs:
        for root, _dirs, files in os.walk(base):
            for fn in files:
                if not fn.endswith("_merged_analysis_table.tsv"):
                    continue
                total += 1
                path = os.path.join(root, fn)
                ok = process_file(path, dry_run=dry_run)
                if ok:
                    updated_or_ok += 1
    print(f"\nDone. Scanned {total}  files, successfully processed {updated_or_ok} .")
