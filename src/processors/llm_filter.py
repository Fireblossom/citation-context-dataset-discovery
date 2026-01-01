"""
LLM-based filtering for citation contexts and papers.
"""

import asyncio
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI
from tqdm import tqdm

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_API_KEY, DEFAULT_LLM_MODEL


@dataclass
class FilterResult:
    """Result of LLM filtering."""

    is_relevant: bool
    confidence: float
    reasoning: str


async def _score_single_paper(
    client: AsyncOpenAI,
    model: str,
    topic: str,
    title: str,
    abstract: str,
    semaphore: asyncio.Semaphore,
    expert_domain: Optional[str] = None,
) -> tuple:
    """Score a single paper for relevance."""

    domain_hint = f" You are an expert in {expert_domain}." if expert_domain else ""

    system_prompt = f"""You are a research paper relevance classifier.{domain_hint}
Determine if a paper is directly relevant to the research topic: "{topic}".
Return ONLY a JSON object with:
- "relevant": true/false
- "confidence": 0.0-1.0
- "reason": brief explanation"""

    paper_text = f"Title: {title}"
    if abstract and abstract.strip():
        paper_text += f"\nAbstract: {abstract[:500]}"

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": paper_text},
                ],
                temperature=0.1,
            )

            content = response.choices[0].message.content

            # Parse JSON response
            json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
            if json_match:
                import json

                result = json.loads(json_match.group())
                return (
                    result.get("relevant", False),
                    result.get("confidence", 0.0),
                    result.get("reason", ""),
                )

            # Fallback parsing
            is_relevant = (
                "true" in content.lower() or '"relevant": true' in content.lower()
            )
            return (is_relevant, 0.5, "Parsed from text")

        except Exception as e:
            return (False, 0.0, f"Error: {e}")


async def llm_prefilter_papers_async(
    papers: List[Dict],
    topic: str,
    min_confidence: float = 0.5,
    max_concurrent: int = 100,
    expert_domain: Optional[str] = None,
) -> List[Dict]:
    """
    Filter papers by relevance to a topic using LLM.

    Args:
        papers: List of papers with corpusId, title, abstract
        topic: Topic to filter for
        min_confidence: Minimum confidence threshold
        max_concurrent: Max concurrent LLM calls
        expert_domain: Optional domain for expert system prompt

    Returns:
        Filtered list of relevant papers
    """
    print(f"\n[LLM] Using LLM to filter {len(papers)}  papers for relevance...")

    client = AsyncOpenAI(
        base_url=DEFAULT_LLM_BASE_URL, api_key=DEFAULT_LLM_API_KEY, timeout=60.0
    )

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_paper(idx: int, paper: Dict):
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        is_relevant, confidence, reason = await _score_single_paper(
            client, DEFAULT_LLM_MODEL, topic, title, abstract, semaphore, expert_domain
        )

        return idx, is_relevant, confidence

    tasks = [process_paper(i, p) for i, p in enumerate(papers)]

    results = []
    for coro in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Filtering", unit="paper"
    ):
        idx, is_relevant, confidence = await coro
        results.append((idx, is_relevant, confidence))

    # Sort by original order
    results.sort(key=lambda x: x[0])

    # Filter papers
    filtered = []
    for idx, is_relevant, confidence in results:
        if is_relevant and confidence >= min_confidence:
            filtered.append(papers[idx])

    await client.close()

    print(
        f"[RESULT] LLM pre-filter results: {len(filtered)}/{len(papers)}  papers related to '{topic}' (confidence>={min_confidence})"
    )

    return filtered


def llm_prefilter_papers(
    papers: List[Dict],
    topic: str,
    min_confidence: float = 0.5,
    expert_domain: Optional[str] = None,
) -> List[Dict]:
    """Synchronous wrapper for LLM prefiltering."""
    return asyncio.run(
        llm_prefilter_papers_async(
            papers, topic, min_confidence, expert_domain=expert_domain
        )
    )
