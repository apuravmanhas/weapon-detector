import unittest

from benchmark.metrics import evaluate_clip_telemetry, format_markdown_report, summarize_benchmark
from benchmark.schema import ClipAnnotation
from src.states import FrameTelemetry, StageLatenciesMs, ThreatState


def _frame(idx: int, ts_ms: float, state: ThreatState, latency_ms: float = 45.0) -> FrameTelemetry:
    return FrameTelemetry(
        frame_index=idx,
        timestamp_ms=ts_ms,
        max_state=state,
        max_risk_score=0.9 if state == ThreatState.AIMING_IMMINENT else (0.55 if state == ThreatState.DRAWING_READYING else 0.1),
        assessments=[],
        persons=[],
        weapons=[],
        latencies=StageLatenciesMs(total_ms=latency_ms),
    )


class TestBenchmarkMetrics(unittest.TestCase):
    def test_net_lead_time_subtracts_inference_latency(self) -> None:
        ann = ClipAnnotation(
            clip_id="draw_01",
            video_path="dummy.mp4",
            bucket="draw_to_fire",
            t_draw_start_ms=1000.0,
            t_aim_start_ms=1450.0,
            t_first_shot_ms=2000.0,
        )
        frames = [
            _frame(0, 900.0, ThreatState.PASSIVE_PRESENT, latency_ms=45.0),
            _frame(1, 1100.0, ThreatState.DRAWING_READYING, latency_ms=45.0),
            _frame(2, 1500.0, ThreatState.AIMING_IMMINENT, latency_ms=45.0),
            _frame(3, 2000.0, ThreatState.AIMING_IMMINENT, latency_ms=45.0),
        ]
        res = evaluate_clip_telemetry(ann, frames)
        self.assertTrue(res.detected_draw_before_shot)
        self.assertTrue(res.detected_aim_before_shot)
        self.assertAlmostEqual(res.net_lead_draw_ms, 855.0, places=2)  # 2000 - (1100 + 45)
        self.assertAlmostEqual(res.net_lead_aim_ms, 455.0, places=2)   # 2000 - (1500 + 45)

    def test_late_alert_after_latency_deduction_counts_as_miss(self) -> None:
        ann = ClipAnnotation(
            clip_id="draw_late",
            video_path="dummy.mp4",
            bucket="draw_to_fire",
            t_first_shot_ms=2000.0,
        )
        # Alert fires on video frame 1970ms, but inference took 45ms -> operator sees alert at 2015ms (15ms AFTER shot!)
        frames = [
            _frame(0, 1900.0, ThreatState.PASSIVE_PRESENT, latency_ms=45.0),
            _frame(1, 1970.0, ThreatState.AIMING_IMMINENT, latency_ms=45.0),
        ]
        res = evaluate_clip_telemetry(ann, frames)
        self.assertFalse(res.detected_aim_before_shot)
        self.assertAlmostEqual(res.net_lead_aim_ms, -15.0, places=2)

    def test_passive_and_mimic_aggregation_and_report(self) -> None:
        ann_p = ClipAnnotation(clip_id="passive_01", video_path="p.mp4", bucket="passive_carry")
        frames_p = [_frame(i, i * 100.0, ThreatState.PASSIVE_PRESENT, 40.0) for i in range(20)]
        res_p = evaluate_clip_telemetry(ann_p, frames_p)
        self.assertEqual(res_p.false_aim_frames, 0)

        ann_m = ClipAnnotation(clip_id="mimic_01", video_path="m.mp4", bucket="non_weapon_mimic")
        frames_m = [_frame(i, i * 100.0, ThreatState.NO_WEAPON, 38.0) for i in range(20)]
        res_m = evaluate_clip_telemetry(ann_m, frames_m)
        self.assertEqual(res_m.false_aim_events, 0)

        summary = summarize_benchmark([res_p, res_m], all_frame_latencies_ms=[38.0, 40.0, 42.0])
        self.assertEqual(summary["passive_weapon_specificity"]["false_aim_frame_rate"], 0.0)
        self.assertEqual(summary["non_weapon_distractors"]["false_aim_events_per_hour"], 0.0)
        md = format_markdown_report(summary)
        self.assertIn("Pre-Firing Anticipation Benchmark Summary", md)


if __name__ == "__main__":
    unittest.main()
