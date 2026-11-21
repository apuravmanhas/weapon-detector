"""
Exact anticipation metrics grounded in real-time embedded latency budgets.

Core metric — Net Effective Lead Time:
    delta_t_net = t_first_shot_ms - (t_alert_frame_ms + inference_latency_ms)

Subtracting measured pipeline latency from video-timestamp lead time guarantees we never
overstate anticipation performance with unrealistic zero-latency assumptions.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmark.schema import ClipAnnotation
from src.states import FrameTelemetry, ThreatState


@dataclass
class ClipEvaluationResult:
    clip_id: str
    bucket: str
    camera_domain: str
    duration_s: float
    num_frames: int
    # Pre-shot anticipation metrics (populated for draw_to_fire)
    t_first_shot_ms: float | None = None
    first_draw_alert_ms: float | None = None
    first_aim_alert_ms: float | None = None
    net_lead_draw_ms: float | None = None
    net_lead_aim_ms: float | None = None
    detected_draw_before_shot: bool | None = None
    detected_aim_before_shot: bool | None = None
    # Hard-negative false escalation metrics (passive_carry & non_weapon_mimic)
    false_aim_frames: int = 0
    false_draw_frames: int = 0
    false_aim_events: int = 0
    false_draw_events: int = 0
    # Latency profile for this clip
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0


def _count_rising_edges(flags: list[bool]) -> int:
    events = 0
    prev = False
    for f in flags:
        if f and not prev:
            events += 1
        prev = f
    return events


def evaluate_clip_telemetry(
    annotation: ClipAnnotation,
    frames: list[FrameTelemetry],
    ignore_early_lead_s: float = 5.0,
) -> ClipEvaluationResult:
    """
    Evaluate one clip's causal FrameTelemetry sequence against its ground-truth annotation.

    For `draw_to_fire`, alerts are considered valid pre-shot anticipation if they occur within
    `[t_first_shot - ignore_early_lead_s, t_first_shot]` (or from `t_draw_start - 500 ms` onward)
    and strictly satisfy `t_alert + tau_infer < t_first_shot`.
    """
    if not frames:
        return ClipEvaluationResult(
            clip_id=annotation.clip_id,
            bucket=annotation.bucket,
            camera_domain=annotation.camera_domain,
            duration_s=0.0,
            num_frames=0,
            t_first_shot_ms=annotation.t_first_shot_ms,
        )

    latencies = np.array([f.latencies.total_ms for f in frames], dtype=np.float64)
    duration_s = max(0.001, (frames[-1].timestamp_ms - frames[0].timestamp_ms) / 1000.0)
    if len(frames) == 1:
        duration_s = 1.0 / 30.0

    res = ClipEvaluationResult(
        clip_id=annotation.clip_id,
        bucket=annotation.bucket,
        camera_domain=annotation.camera_domain,
        duration_s=duration_s,
        num_frames=len(frames),
        t_first_shot_ms=annotation.t_first_shot_ms,
        mean_latency_ms=float(np.mean(latencies)),
        p50_latency_ms=float(np.percentile(latencies, 50)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        p99_latency_ms=float(np.percentile(latencies, 99)),
    )

    if annotation.bucket == "draw_to_fire" and annotation.t_first_shot_ms is not None:
        t_shot = annotation.t_first_shot_ms
        window_start = (
            max(0.0, annotation.t_draw_start_ms - 500.0)
            if annotation.t_draw_start_ms is not None
            else max(0.0, t_shot - ignore_early_lead_s * 1000.0)
        )

        first_draw_frame: FrameTelemetry | None = None
        first_aim_frame: FrameTelemetry | None = None

        for ft in frames:
            if ft.timestamp_ms < window_start:
                continue
            if first_draw_frame is None and ft.max_state >= ThreatState.DRAWING_READYING:
                first_draw_frame = ft
            if first_aim_frame is None and ft.max_state >= ThreatState.AIMING_IMMINENT:
                first_aim_frame = ft

        if first_draw_frame is not None:
            res.first_draw_alert_ms = first_draw_frame.timestamp_ms
            res.net_lead_draw_ms = t_shot - (first_draw_frame.timestamp_ms + first_draw_frame.latencies.total_ms)
            res.detected_draw_before_shot = res.net_lead_draw_ms > 0.0
        else:
            res.detected_draw_before_shot = False

        if first_aim_frame is not None:
            res.first_aim_alert_ms = first_aim_frame.timestamp_ms
            res.net_lead_aim_ms = t_shot - (first_aim_frame.timestamp_ms + first_aim_frame.latencies.total_ms)
            res.detected_aim_before_shot = res.net_lead_aim_ms > 0.0
        else:
            res.detected_aim_before_shot = False
    else:
        aim_flags = [ft.max_state >= ThreatState.AIMING_IMMINENT for ft in frames]
        draw_flags = [ft.max_state >= ThreatState.DRAWING_READYING for ft in frames]
        res.false_aim_frames = sum(1 for v in aim_flags if v)
        res.false_draw_frames = sum(1 for v in draw_flags if v)
        res.false_aim_events = _count_rising_edges(aim_flags)
        res.false_draw_events = _count_rising_edges(draw_flags)

    return res


def _lead_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median_ms": None, "p25_ms": None, "p75_ms": None, "min_ms": None, "max_ms": None}
    arr = np.array(values, dtype=np.float64)
    return {
        "median_ms": round(float(np.median(arr)), 1),
        "p25_ms": round(float(np.percentile(arr, 25)), 1),
        "p75_ms": round(float(np.percentile(arr, 75)), 1),
        "min_ms": round(float(np.min(arr)), 1),
        "max_ms": round(float(np.max(arr)), 1),
    }


def summarize_benchmark(
    clip_results: list[ClipEvaluationResult],
    all_frame_latencies_ms: list[float] | None = None,
) -> dict[str, Any]:
    """Aggregate per-clip metrics into a publication-ready summary dictionary."""
    pos = [r for r in clip_results if r.bucket == "draw_to_fire"]
    passive = [r for r in clip_results if r.bucket == "passive_carry"]
    mimic = [r for r in clip_results if r.bucket == "non_weapon_mimic"]

    draw_before = [r for r in pos if r.detected_draw_before_shot]
    aim_before = [r for r in pos if r.detected_aim_before_shot]

    draw_leads = [r.net_lead_draw_ms for r in draw_before if r.net_lead_draw_ms is not None]
    aim_leads = [r.net_lead_aim_ms for r in aim_before if r.net_lead_aim_ms is not None]

    passive_total_frames = sum(r.num_frames for r in passive)
    passive_false_aim_frames = sum(r.false_aim_frames for r in passive)
    passive_false_aim_clips = sum(1 for r in passive if r.false_aim_events > 0)
    passive_hours = sum(r.duration_s for r in passive) / 3600.0

    mimic_total_frames = sum(r.num_frames for r in mimic)
    mimic_false_aim_frames = sum(r.false_aim_frames for r in mimic)
    mimic_false_aim_events = sum(r.false_aim_events for r in mimic)
    mimic_hours = sum(r.duration_s for r in mimic) / 3600.0

    if all_frame_latencies_ms:
        lat_arr = np.array(all_frame_latencies_ms, dtype=np.float64)
        latency_summary = {
            "mean_ms": round(float(np.mean(lat_arr)), 2),
            "p50_ms": round(float(np.percentile(lat_arr, 50)), 2),
            "p95_ms": round(float(np.percentile(lat_arr, 95)), 2),
            "p99_ms": round(float(np.percentile(lat_arr, 99)), 2),
            "avg_fps": round(1000.0 / max(1e-3, float(np.mean(lat_arr))), 1),
        }
    elif clip_results:
        means = [r.mean_latency_ms for r in clip_results if r.num_frames > 0]
        latency_summary = {
            "mean_ms": round(float(np.mean(means)), 2) if means else 0.0,
            "p50_ms": round(float(np.median([r.p50_latency_ms for r in clip_results if r.num_frames > 0])), 2) if means else 0.0,
            "p95_ms": round(float(np.max([r.p95_latency_ms for r in clip_results if r.num_frames > 0])), 2) if means else 0.0,
            "p99_ms": round(float(np.max([r.p99_latency_ms for r in clip_results if r.num_frames > 0])), 2) if means else 0.0,
            "avg_fps": round(1000.0 / max(1e-3, float(np.mean(means))), 1) if means else 0.0,
        }
    else:
        latency_summary = {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "avg_fps": 0.0}

    return {
        "counts": {
            "total_clips": len(clip_results),
            "draw_to_fire": len(pos),
            "passive_carry": len(passive),
            "non_weapon_mimic": len(mimic),
        },
        "pre_shot_anticipation": {
            "pre_alert_draw_rate": round(len(draw_before) / len(pos), 4) if pos else None,
            "imminent_aim_rate": round(len(aim_before) / len(pos), 4) if pos else None,
            "net_lead_draw": _lead_stats(draw_leads),
            "net_lead_aim": _lead_stats(aim_leads),
        },
        "passive_weapon_specificity": {
            "false_aim_clip_rate": round(passive_false_aim_clips / len(passive), 4) if passive else None,
            "false_aim_frame_rate": round(passive_false_aim_frames / passive_total_frames, 5)
            if passive_total_frames > 0
            else None,
            "false_aim_events_per_hour": round(sum(r.false_aim_events for r in passive) / passive_hours, 2)
            if passive_hours > 0
            else None,
        },
        "non_weapon_distractors": {
            "false_aim_frame_rate": round(mimic_false_aim_frames / mimic_total_frames, 5)
            if mimic_total_frames > 0
            else None,
            "false_aim_events_per_hour": round(mimic_false_aim_events / mimic_hours, 2)
            if mimic_hours > 0
            else None,
        },
        "latency_ms": latency_summary,
    }


def format_markdown_report(summary: dict[str, Any]) -> str:
    """Format benchmark summary as a concise, zero-fluff Markdown engineering table."""
    counts = summary["counts"]
    pre = summary["pre_shot_anticipation"]
    pas = summary["passive_weapon_specificity"]
    mim = summary["non_weapon_distractors"]
    lat = summary["latency_ms"]

    def _pct(v: float | None) -> str:
        return f"{v * 100:.1f}%" if v is not None else "N/A"

    def _ms(v: float | None) -> str:
        return f"{v:.1f} ms" if v is not None else "N/A"

    lines = [
        "## Pre-Firing Anticipation Benchmark Summary",
        "",
        f"- **Clips evaluated:** {counts['total_clips']} "
        f"(`draw_to_fire={counts['draw_to_fire']}`, `passive_carry={counts['passive_carry']}`, `non_weapon_mimic={counts['non_weapon_mimic']}`)",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Pre-Shot Alert Rate (`DRAWING+` before `t_shot`) | {_pct(pre['pre_alert_draw_rate'])} |",
        f"| Imminent Alert Rate (`AIMING` before `t_shot`) | {_pct(pre['imminent_aim_rate'])} |",
        f"| Median Net Lead Time (`DRAWING+`, minus latency) | {_ms(pre['net_lead_draw']['median_ms'])} |",
        f"| Median Net Lead Time (`AIMING`, minus latency) | {_ms(pre['net_lead_aim']['median_ms'])} |",
        f"| Passive-Weapon False Escalation Frame Rate | {_pct(pas['false_aim_frame_rate'])} |",
        f"| Non-Weapon Distractor False Aim Events / hr | {mim['false_aim_events_per_hour'] if mim['false_aim_events_per_hour'] is not None else 'N/A'} |",
        f"| Inference Latency (Mean / P50 / P95 / P99) | {lat['mean_ms']} / {lat['p50_ms']} / {lat['p95_ms']} / {lat['p99_ms']} ms ({lat['avg_fps']} FPS) |",
    ]
    return "\n".join(lines)
