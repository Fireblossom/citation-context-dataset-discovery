# Extractors subpackage
# Lazy imports to avoid errors when openai is not installed


def __getattr__(name):
    if (
        name == "DatasetExtractor"
        or name == "DatasetExtractorOpenAI"
        or name == "DatasetExtractionResult"
    ):
        from .dataset_extractor import (
            DatasetExtractor,
            DatasetExtractorOpenAI,
            DatasetExtractionResult,
        )

        return {
            "DatasetExtractor": DatasetExtractor,
            "DatasetExtractorOpenAI": DatasetExtractorOpenAI,
            "DatasetExtractionResult": DatasetExtractionResult,
        }[name]
    elif name == "BaseLLMExtractor" or name == "ExtractionResult":
        from .base import BaseLLMExtractor, ExtractionResult

        return {
            "BaseLLMExtractor": BaseLLMExtractor,
            "ExtractionResult": ExtractionResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DatasetExtractor",
    "DatasetExtractorOpenAI",
    "DatasetExtractionResult",
    "BaseLLMExtractor",
    "ExtractionResult",
]
