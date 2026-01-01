"""
Base LLM Extractor class with shared async processing logic.
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
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
class ExtractionResult:
    """Base extraction result."""

    context_text: str
    extracted_items: List[str]
    confidence_score: float
    reasoning: str
    processing_time: float
    citing_paper_id: Optional[str] = None
    cited_paper_id: Optional[str] = None


class BaseLLMExtractor:
    """
    Base class for LLM-based extraction tasks.

    Provides:
    - Async client management
    - Batch processing with concurrency control
    - Progress tracking
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
        """Close the async client cleanly."""
        if self.client is not None:
            try:
                await self.client.close()
            except Exception:
                pass
            finally:
                self.client = None

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        semaphore: asyncio.Semaphore,
    ) -> Optional[str]:
        """
        Make a single LLM call with rate limiting.

        Args:
            system_prompt: System message
            user_prompt: User message
            semaphore: Concurrency limiter

        Returns:
            LLM response content or None on error
        """
        async with semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"LLM call error: {e}")
                return None

    async def _process_batch_async(
        self, items: List[Any], process_fn, desc: str = "Processing"
    ) -> List[Any]:
        """
        Process items in batch with async concurrency.

        Args:
            items: Items to process
            process_fn: Async function to process each item
            desc: Progress bar description

        Returns:
            List of results in same order as input
        """
        if not self.client:
            self.initialize()

        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def run_with_index(index: int, item):
            result = await process_fn(item, semaphore)
            return index, result

        tasks = [run_with_index(i, item) for i, item in enumerate(items)]

        results = [None] * len(items)

        for coro in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc=desc, unit="item"
        ):
            idx, result = await coro
            results[idx] = result

        return results

    def process_batch(
        self, items: List[Any], process_fn, desc: str = "Processing"
    ) -> List[Any]:
        """Synchronous wrapper for batch processing."""
        return asyncio.run(self._process_batch_async(items, process_fn, desc))

    def get_prompt_template(self) -> str:
        """Override in subclass to provide extraction prompt."""
        raise NotImplementedError("Subclass must implement get_prompt_template()")
