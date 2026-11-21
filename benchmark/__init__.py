"""Evaluation manifest schema and anticipation lead-time metrics."""

from benchmark.metrics import (
    ClipEvaluationResult,
    evaluate_clip_telemetry,
    format_markdown_report,
    summarize_benchmark,
)
from benchmark.schema import ClipAnnotation, load_manifest

__all__ = [
    "ClipAnnotation",
    "ClipEvaluationResult",
    "evaluate_clip_telemetry",
    "format_markdown_report",
    "load_manifest",
    "summarize_benchmark",
]
