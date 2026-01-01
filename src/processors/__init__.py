# Processors subpackage
# Lazy imports to avoid errors when openai is not installed


def __getattr__(name):
    if name == "llm_prefilter_papers":
        from .llm_filter import llm_prefilter_papers

        return llm_prefilter_papers
    elif name == "analyze_datasets":
        from .dataset_analyzer import analyze_datasets

        return analyze_datasets
    elif name == "DatasetAnalyzer":
        from .dataset_analyzer import analyze_datasets as DatasetAnalyzer

        return DatasetAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["llm_prefilter_papers", "analyze_datasets"]
