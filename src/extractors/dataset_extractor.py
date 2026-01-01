"""
Dataset Extractor - Complete implementation

Extracts dataset names from citation contexts using LLM (OpenAI-compatible API).
"""

import json
import re
import time
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from openai import AsyncOpenAI
from tqdm import tqdm

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_MAX_CONCURRENT_REQUESTS,
)


@dataclass
class DatasetExtractionResult:
    """Result of dataset extraction from a citation context."""

    context_text: str
    citation_intent: List[str]
    resource_type: List[str]
    extracted_datasets: List[str]
    confidence_score: float
    reasoning: str
    processing_time: float
    citing_paper_id: Optional[str] = None
    cited_paper_id: Optional[str] = None
    dataset_descriptions: Optional[Dict[str, str]] = None


class DatasetExtractor:
    """
    Extracts dataset names from citation contexts using LLM.

    Usage:
        extractor = DatasetExtractor()
        extractor.initialize()
        results = extractor.extract_from_citation_contexts("contexts.json", "datasets.json")
    """

    def __init__(
        self,
        model_name: str = None,
        base_url: str = None,
        api_key: str = None,
        max_concurrent_requests: int = None,
        timeout: float = None,
        expert_domain: Optional[str] = None,
    ):
        self.model_name = model_name or DEFAULT_LLM_MODEL
        self.base_url = base_url or DEFAULT_LLM_BASE_URL
        self.api_key = api_key or DEFAULT_LLM_API_KEY
        self.max_concurrent_requests = (
            max_concurrent_requests or DEFAULT_MAX_CONCURRENT_REQUESTS
        )
        self.timeout = timeout or DEFAULT_LLM_TIMEOUT
        self.expert_domain = expert_domain
        self.client: Optional[AsyncOpenAI] = None
        self.current_topic: Optional[str] = None

    def initialize(self) -> bool:
        """Initialize the OpenAI client."""
        try:
            self.client = AsyncOpenAI(
                base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
            )
            return True
        except Exception as e:
            print(f"[ERROR] Client initialization failed: {e}")
            return False

    async def close(self) -> None:
        """Close the async client."""
        if self.client is not None:
            try:
                await self.client.close()
            except Exception:
                pass
            finally:
                self.client = None

    def get_extraction_prompt(self) -> str:
        """Get the extraction prompt template."""
        return """Analyze the following citation context and extract any datasets or data resources mentioned.

Citation context:
{context}

Metadata (if available):
{meta_block}

Research topic: {topic}

Extract and return a JSON object with:
{{
    "citation intent": ["background", "methodology", "result comparison", etc.],
    "resource type": ["dataset", "software", "database", "benchmark", etc.],
    "dataset labels": ["dataset name 1", "dataset name 2", ...],
    "dataset_descriptions": {{"dataset name": "brief description"}},
    "confidence": 0.0+1.0,
    "reasoning": "explanation of extraction"
}}

Important:
- Only extract ACTUAL dataset/resource names, not generic terms
- Include acronyms when mentioned
- If no datasets found, return empty "dataset labels" array
- Be conservative - only include if clearly a named data resource"""

    async def _extract_single_context(
        self,
        context_text: str,
        semaphore: asyncio.Semaphore,
        cited_title: Optional[str] = None,
        cited_abstract: Optional[str] = None,
        citing_paper_id: Optional[str] = None,
        cited_paper_id: Optional[str] = None,
    ) -> DatasetExtractionResult:
        """Extract datasets from a single context."""
        async with semaphore:
            start_time = time.time()

            try:
                # Build metadata block
                meta_lines = []
                if cited_title:
                    meta_lines.append(f"Title: {cited_title[:300]}")

                meta_block = "\n".join(meta_lines) if meta_lines else "(none)"

                prompt = self.get_extraction_prompt().format(
                    context=context_text[:2000],
                    meta_block=meta_block,
                    topic=self.current_topic or "",
                )

                # Build system role
                if self.expert_domain:
                    system_role = f"You are an expert in {self.expert_domain}. Extract datasets from citations. Respond with only valid JSON."
                else:
                    system_role = "You extract datasets from citations. Respond with only valid JSON."

                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                )

                response_text = response.choices[0].message.content.strip()
                processing_time = time.time() - start_time

                # Parse JSON response
                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    datasets = result.get("dataset labels", [])
                    citation_intent = result.get("citation intent", [])
                    resource_type = result.get("resource type", [])
                    dataset_descriptions = result.get("dataset_descriptions", {})
                    confidence = result.get("confidence", 0.5)
                    reasoning = result.get("reasoning", "")

                    if not isinstance(datasets, list):
                        datasets = []
                    if len(datasets) == 0 and confidence > 0.6:
                        confidence = 0.6
                else:
                    datasets, citation_intent, resource_type = [], [], []
                    dataset_descriptions = {}
                    confidence = 0.3
                    reasoning = f"JSON parse failed"

            except json.JSONDecodeError as e:
                datasets, citation_intent, resource_type = [], [], []
                dataset_descriptions = {}
                confidence = 0.3
                reasoning = f"JSON decode error: {e}"
                processing_time = time.time() - start_time

            except Exception as e:
                datasets, citation_intent, resource_type = [], [], []
                dataset_descriptions = {}
                confidence = 0.3
                reasoning = f"API error: {e}"
                processing_time = time.time() - start_time

            return DatasetExtractionResult(
                context_text=context_text,
                citation_intent=citation_intent,
                resource_type=resource_type,
                extracted_datasets=datasets,
                confidence_score=confidence,
                reasoning=reasoning,
                processing_time=processing_time,
                citing_paper_id=citing_paper_id,
                cited_paper_id=cited_paper_id,
                dataset_descriptions=dataset_descriptions,
            )

    async def _extract_batch_async(self, contexts: List[Any]) -> Dict[str, Any]:
        """Async batch extraction."""
        if not self.client:
            self.initialize()

        start_time = time.time()
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def run_with_index(idx: int, ctx):
            if isinstance(ctx, dict):
                ctx_text = ctx.get("context") or ctx.get("text") or ""
                title = ctx.get("cited_title") or ctx.get("title")
                abstract = ctx.get("cited_abstract")
                citing_id = ctx.get("citing_paper_id")
                cited_id = ctx.get("cited_paper_id")
            else:
                ctx_text = str(ctx)
                title, abstract, citing_id, cited_id = None, None, None, None

            result = await self._extract_single_context(
                ctx_text, semaphore, title, abstract, citing_id, cited_id
            )
            return idx, result

        tasks = [run_with_index(i, c) for i, c in enumerate(contexts)]
        results = [None] * len(contexts)
        successful = 0

        with tqdm(total=len(tasks), desc="Extracting", unit="ctx") as pbar:
            for coro in asyncio.as_completed(tasks):
                idx, result = await coro
                results[idx] = result
                if result.extracted_datasets:
                    successful += 1
                pbar.update(1)

        return {
            "total_processed": len(contexts),
            "successful_extractions": successful,
            "failed_extractions": len(contexts) - successful,
            "results": results,
            "total_processing_time": time.time() - start_time,
        }

    def extract_datasets_batch(self, contexts: List[Any]) -> Dict[str, Any]:
        """Synchronous wrapper for batch extraction."""
        return asyncio.run(self._extract_batch_async(contexts))

    def extract_from_citation_contexts(
        self,
        json_file_path: str,
        output_file: str = "extracted_datasets.json",
        filter_topic: bool = True,
        topic: str = "topic modeling",
    ) -> bool:
        """
        Extract datasets from a citation contexts JSON file.

        Args:
            json_file_path: Input JSON file with citation contexts
            output_file: Output JSON file for results
            filter_topic: Whether to filter by topic
            topic: Topic to filter for

        Returns:
            True if successful
        """
        print(f"=== Dataset Extraction ===")
        print(f"Input: {json_file_path}")
        print(f"Output: {output_file}")
        print(f"Topic: {topic}")

        self.current_topic = topic

        # Load input file
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load file: {e}")
            return False

        # Extract contexts
        contexts = []

        # Handle merged format (from --merge option)
        if "contexts" in data and isinstance(data["contexts"], list):
            for ctx_item in data["contexts"]:
                if isinstance(ctx_item, dict) and ctx_item.get("context"):
                    contexts.append(
                        {
                            "context": str(ctx_item["context"]),
                            "citing_paper_id": str(ctx_item.get("citing_paper_id"))
                            if ctx_item.get("citing_paper_id")
                            else None,
                            "cited_paper_id": str(ctx_item.get("cited_paper_id"))
                            if ctx_item.get("cited_paper_id")
                            else None,
                        }
                    )

        # Handle original format (papers with citations)
        elif "papers" in data:
            for paper in data["papers"]:
                citing_id = paper.get("corpusid") or paper.get("cited_corpusid")
                for citation in paper.get("citations", []) + paper.get(
                    "citing_papers", []
                ):
                    cited_id = citation.get("cited_corpusid") or citation.get(
                        "citing_corpusid"
                    )
                    for ctx in citation.get("contexts", []):
                        if ctx and str(ctx).strip():
                            contexts.append(
                                {
                                    "context": str(ctx),
                                    "citing_paper_id": str(citing_id)
                                    if citing_id
                                    else None,
                                    "cited_paper_id": str(cited_id)
                                    if cited_id
                                    else None,
                                }
                            )

        if not contexts:
            print("[ERROR] Not foundvalid contexts")
            return False

        print(f"Found {len(contexts)}  contexts")

        # Initialize client
        if not self.initialize():
            return False

        # Process
        try:
            results = self.extract_datasets_batch(contexts)
        finally:
            asyncio.run(self.close())

        # Build output
        output_data = {
            "source_file": json_file_path,
            "topic": topic,
            "total_contexts": results["total_processed"],
            "contexts_with_datasets": results["successful_extractions"],
            "processing_time": results["total_processing_time"],
            "extractions": [],
        }

        for result in results["results"]:
            if result and result.extracted_datasets:
                output_data["extractions"].append(
                    {
                        "context": result.context_text[:500],
                        "datasets": result.extracted_datasets,
                        "descriptions": result.dataset_descriptions,
                        "citation_intent": result.citation_intent,
                        "resource_type": result.resource_type,
                        "confidence": result.confidence_score,
                        "citing_paper_id": result.citing_paper_id,
                        "cited_paper_id": result.cited_paper_id,
                    }
                )

        # Save output
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"[OK] Extraction complete: {len(output_data['extractions'])}  results")
        print(f"Saved to: {output_file}")

        return True


# Alias for backward compatibility
DatasetExtractorOpenAI = DatasetExtractor


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract dataset names from citation contexts")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    parser.add_argument(
        "--output", "-o", default="extracted_datasets.json", help="Output JSON file"
    )
    parser.add_argument("--topic", "-t", default="topic modeling", help="Research topic")

    args = parser.parse_args()

    extractor = DatasetExtractor()
    success = extractor.extract_from_citation_contexts(
        args.input, args.output, topic=args.topic
    )

    return 0 if success else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
