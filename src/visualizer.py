"""
Minimal, clean engineering visualization for Pre-Firing Intent telemetry.

Color-coded strictly by ThreatState (BGR):
  - PASSIVE_PRESENT  -> Muted Cyan/Blue  (210, 175, 40)
  - DRAWING_READYING -> Amber            (0, 190, 255)
  - AIMING_IMMINENT  -> Red              (30, 30, 240)
Draws skeleton lines ONLY for persons actively associated with a weapon to keep frames uncluttered.
"""

import cv2
import numpy as np

from src.states import FrameTelemetry, PersonTrack, ThreatState

STATE_COLORS_BGR: dict[ThreatState, tuple[int, int, int]] = {
    ThreatState.NO_WEAPON: (120, 120, 120),
    ThreatState.PASSIVE_PRESENT: (210, 175, 40),
    ThreatState.DRAWING_READYING: (0, 190, 255),
    ThreatState.AIMING_IMMINENT: (30, 30, 240),
}

_UPPER_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (5, 6),   # shoulders
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12), # torso sides
    (11, 12),         # hips
)


def _draw_upper_skeleton(
    img: np.ndarray,
    person: PersonTrack,
    color: tuple[int, int, int],
    kpt_thresh: float = 0.25,
) -> None:
    kpts = person.keypoints
    if kpts is None or len(kpts) < 13:
        return
    for a, b in _UPPER_SKELETON_EDGES:
        if float(kpts[a, 2]) >= kpt_thresh and float(kpts[b, 2]) >= kpt_thresh:
            p1 = (int(round(float(kpts[a, 0]))), int(round(float(kpts[a, 1]))))
            p2 = (int(round(float(kpts[b, 0]))), int(round(float(kpts[b, 1]))))
            cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)
    for idx in (5, 6, 7, 8, 9, 10, 11, 12):
        if float(kpts[idx, 2]) >= kpt_thresh:
            pt = (int(round(float(kpts[idx, 0]))), int(round(float(kpts[idx, 1]))))
            cv2.circle(img, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)


def render_telemetry_frame(
    frame: np.ndarray,
    telemetry: FrameTelemetry,
    show_skeleton: bool = True,
    show_hud: bool = True,
    lead_countdown_ms: float | None = None,
) -> np.ndarray:
    """Render clean readiness boxes, bearer skeleton, and compact engineering latency bar."""
    out = frame.copy()
    fh, fw = out.shape[:2]
    long_edge = max(fh, fw)
    thick = max(2, min(4, round(0.0026 * long_edge)))
    font_scale = max(0.48, min(0.85, long_edge / 1550.0))
    font_thick = max(1, min(2, round(font_scale * 2.2)))

    persons_by_id = {p.person_id: p for p in telemetry.persons}

    for item in telemetry.assessments:
        color = STATE_COLORS_BGR[item.state]
        wx1, wy1, wx2, wy2 = item.weapon_box

        # Draw bearer upper-body skeleton first (behind box labels)
        if show_skeleton and item.person_id is not None and item.person_id in persons_by_id:
            _draw_upper_skeleton(out, persons_by_id[item.person_id], color)

        # Draw weapon box
        cv2.rectangle(out, (wx1, wy1), (wx2, wy2), color, thick)
        short_state = {
            ThreatState.PASSIVE_PRESENT: "PASSIVE",
            ThreatState.DRAWING_READYING: "DRAWING",
            ThreatState.AIMING_IMMINENT: "AIMING",
        }.get(item.state, item.state.label)
        label = f"{short_state} {item.risk_score:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

        top = wy1 - lh - 8 if wy1 >= lh + 10 else wy1
        bot = top + lh + 8
        cv2.rectangle(out, (wx1, top), (min(fw, wx1 + lw + 8), min(fh, bot)), color, -1)
        cv2.putText(
            out,
            label,
            (wx1 + 4, min(fh - 2, bot - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thick,
            cv2.LINE_AA,
        )

    if show_hud:
        # Compact bottom telemetry strip (two-line fallback on narrow portrait frames)
        base_strip_h = max(26, int(round(fh * 0.055)))
        hud_color = STATE_COLORS_BGR[telemetry.max_state]
        status_txt = f"STATE: {telemetry.max_state.label} | RISK: {telemetry.max_risk_score:.2f}"
        lat_txt = f"LAT: {telemetry.latencies.total_ms:.1f} ms"
        if lead_countdown_ms is not None:
            sign = "-" if lead_countdown_ms >= 0 else "+"
            lat_txt += f" | T{sign}{abs(lead_countdown_ms):.0f} ms"

        hud_scale = max(0.45, min(0.68, base_strip_h / 42.0))
        (sw, sh), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_SIMPLEX, hud_scale, 1)
        (rw, rh), _ = cv2.getTextSize(lat_txt, cv2.FONT_HERSHEY_SIMPLEX, hud_scale, 1)
        single_line = (sw + rw + 28) <= fw

        if single_line:
            strip_h = base_strip_h
        else:
            max_txt_w = max(1, max(sw, rw))
            if max_txt_w > fw - 20:
                hud_scale = max(0.32, hud_scale * (fw - 20) / max_txt_w)
                (sw, sh), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_SIMPLEX, hud_scale, 1)
                (rw, rh), _ = cv2.getTextSize(lat_txt, cv2.FONT_HERSHEY_SIMPLEX, hud_scale, 1)
            line_h = max(sh, rh, 12)
            strip_h = max(base_strip_h, 2 * line_h + 22)

        overlay = out.copy()
        cv2.rectangle(overlay, (0, fh - strip_h), (fw, fh), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.72, out, 0.28, 0.0, out)

        if single_line:
            y_text = fh - max(7, (strip_h - 14) // 2)
            cv2.putText(out, status_txt, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, hud_scale, hud_color, 1, cv2.LINE_AA)
            cv2.putText(
                out,
                lat_txt,
                (max(10, fw - rw - 10), y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                hud_scale,
                (225, 225, 225),
                1,
                cv2.LINE_AA,
            )
        else:
            line_h = max(sh, rh, 12)
            y1 = fh - strip_h + line_h + 7
            y2 = fh - 8
            cv2.putText(out, status_txt, (10, y1), cv2.FONT_HERSHEY_SIMPLEX, hud_scale, hud_color, 1, cv2.LINE_AA)
            cv2.putText(out, lat_txt, (10, y2), cv2.FONT_HERSHEY_SIMPLEX, hud_scale, (225, 225, 225), 1, cv2.LINE_AA)

    return out
