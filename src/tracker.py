from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import cv2
import numpy as np


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ix1 = max(float(box_a[0]), float(box_b[0]))
    iy1 = max(float(box_a[1]), float(box_b[1]))
    ix2 = min(float(box_a[2]), float(box_b[2]))
    iy2 = min(float(box_a[3]), float(box_b[3]))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = _box_area(box_a) + _box_area(box_b) - inter
    return inter / union if union > 0.0 else 0.0


@dataclass
class TrackState:
    """Internal kinematic, confidence, and static-confounder state for one tracked object."""

    track_id: int
    box: list[int]
    cx: float
    cy: float
    w: float
    h: float
    ema_conf: float
    last_conf: float
    cls_name: str
    hit_history: deque[int]
    anchor_box: list[int]
    vx: float = 0.0
    vy: float = 0.0
    verifier_ema: float | None = None
    verifier_label: str | None = None
    age: int = 1
    misses: int = 0
    motionless_frames: int = 0
    quarantined: bool = False


class WeaponTrackManager:
    """
    Velocity-aware multi-object tracker with confidence EMA, multi-frame Stage-2 verifier
    vote averaging, and motionless low-confidence confounder quarantine.
    """

    def __init__(
        self,
        min_hits: int = 1,
        window_size: int = 5,
        match_iou: float = 0.20,
        max_center_dist: float = 0.65,
        max_age: int = 6,
        ema_alpha: float = 0.45,
        high_conf_bypass: float = 0.80,
        static_suppress: bool = False,
        static_iou: float = 0.90,
        static_mad: float = 4.0,
        static_frames: int = 8,
        static_max_conf: float = 0.58,
        verify_thresh: float = 0.50,
        verifier_ema_alpha: float = 0.45,
    ):
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1.")
        if window_size < min_hits:
            raise ValueError("window_size must be >= min_hits.")
        if not 0.0 <= match_iou <= 1.0:
            raise ValueError("match_iou must be between 0.0 and 1.0.")
        if max_center_dist < 0.0:
            raise ValueError("max_center_dist must be >= 0.0.")
        if max_age < 0:
            raise ValueError("max_age must be >= 0.")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0.0, 1.0].")

        self.min_hits = min_hits
        self.window_size = window_size
        self.match_iou = match_iou
        self.max_center_dist = max_center_dist
        self.max_age = max_age
        self.ema_alpha = ema_alpha
        self.high_conf_bypass = high_conf_bypass

        self.static_suppress = static_suppress
        self.static_iou = static_iou
        self.static_mad = static_mad
        self.static_frames = max(1, static_frames)
        self.static_max_conf = static_max_conf

        self.verify_thresh = verify_thresh
        self.verifier_ema_alpha = verifier_ema_alpha

        self._next_id = 1
        self._tracks: dict[int, TrackState] = {}
        self._prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self._tracks.clear()
        self._prev_gray = None
        self._next_id = 1

    @property
    def active_tracks(self) -> dict[int, TrackState]:
        return self._tracks

    def _to_gray(self, frame: np.ndarray | None) -> np.ndarray | None:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3:
            if frame.shape[2] == 1:
                return frame[:, :, 0]
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return None

    def _crop_mad(self, prev_gray: np.ndarray | None, curr_gray: np.ndarray | None, box: Sequence[int]) -> float | None:
        if prev_gray is None or curr_gray is None or prev_gray.shape != curr_gray.shape:
            return None
        h, w = curr_gray.shape[:2]
        x1 = max(0, min(w, int(box[0])))
        y1 = max(0, min(h, int(box[1])))
        x2 = max(0, min(w, int(box[2])))
        y2 = max(0, min(h, int(box[3])))
        if x2 <= x1 or y2 <= y1:
            return None
        c1 = curr_gray[y1:y2, x1:x2].astype(np.int16)
        c0 = prev_gray[y1:y2, x1:x2].astype(np.int16)
        return float(np.mean(np.abs(c1 - c0)))

    def _match_detections(
        self, detections: list[dict[str, Any]]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        track_ids = list(self._tracks.keys())
        if not track_ids or not detections:
            return [], track_ids, list(range(len(detections)))

        candidates: list[tuple[float, int, int]] = []
        for tid in track_ids:
            tr = self._tracks[tid]
            pred_cx = tr.cx + tr.vx
            pred_cy = tr.cy + tr.vy
            pred_box = [
                pred_cx - tr.w * 0.5,
                pred_cy - tr.h * 0.5,
                pred_cx + tr.w * 0.5,
                pred_cy + tr.h * 0.5,
            ]
            tr_area = max(1.0, tr.w * tr.h)
            tr_diag = math.hypot(tr.w, tr.h)

            for d_idx, det in enumerate(detections):
                d_box = det["box"]
                d_w = max(1.0, float(d_box[2]) - float(d_box[0]))
                d_h = max(1.0, float(d_box[3]) - float(d_box[1]))
                d_cx = (float(d_box[0]) + float(d_box[2])) * 0.5
                d_cy = (float(d_box[1]) + float(d_box[3])) * 0.5

                best_iou = max(_compute_iou(pred_box, d_box), _compute_iou(tr.box, d_box))
                mean_diag = max(1.0, 0.5 * (tr_diag + math.hypot(d_w, d_h)))
                norm_dist = math.hypot(d_cx - pred_cx, d_cy - pred_cy) / mean_diag
                scale_ratio = (d_w * d_h) / tr_area

                iou_ok = best_iou >= self.match_iou
                dist_ok = (norm_dist <= self.max_center_dist) and (0.4 <= scale_ratio <= 2.5)
                if iou_ok or dist_ok:
                    score = best_iou + max(0.0, 1.0 - norm_dist)
                    candidates.append((score, tid, d_idx))

        candidates.sort(key=lambda item: item[0], reverse=True)
        matched_tids: set[int] = set()
        matched_didxs: set[int] = set()
        matches: list[tuple[int, int]] = []
        for _, tid, d_idx in candidates:
            if tid in matched_tids or d_idx in matched_didxs:
                continue
            matched_tids.add(tid)
            matched_didxs.add(d_idx)
            matches.append((tid, d_idx))

        unmatched_tracks = [tid for tid in track_ids if tid not in matched_tids]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_didxs]
        return matches, unmatched_tracks, unmatched_dets

    def _update_static_state(
        self,
        tr: TrackState,
        curr_gray: np.ndarray | None,
    ) -> None:
        if not self.static_suppress:
            tr.quarantined = False
            return

        anchor_iou = _compute_iou(tr.box, tr.anchor_box)
        mad = self._crop_mad(self._prev_gray, curr_gray, tr.box)

        if anchor_iou >= self.static_iou and mad is not None:
            if mad < self.static_mad:
                tr.motionless_frames += 1
            else:
                tr.anchor_box = list(tr.box)
                tr.motionless_frames = 0
                tr.quarantined = False
        elif anchor_iou < self.static_iou:
            tr.anchor_box = list(tr.box)
            tr.motionless_frames = 0
            tr.quarantined = False

        verifier_strong = tr.verifier_ema is not None and tr.verifier_ema >= 0.80
        if (
            tr.motionless_frames >= self.static_frames
            and tr.ema_conf < self.static_max_conf
            and not verifier_strong
        ):
            tr.quarantined = True
        else:
            tr.quarantined = False

    def update(
        self,
        detections: list[dict[str, Any]],
        frame: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """
        Match current detections against active tracks, update kinematics/EMA/static state,
        and return confirmed non-quarantined detections annotated with track metadata.
        """
        curr_gray = self._to_gray(frame) if self.static_suppress else None
        matches, unmatched_tracks, unmatched_dets = self._match_detections(detections)

        active_pairs: list[tuple[TrackState, dict[str, Any]]] = []

        for tid, d_idx in matches:
            tr = self._tracks[tid]
            det = detections[d_idx]
            d_box = [int(v) for v in det["box"]]
            conf = float(det["confidence"])

            d_w = float(max(1, d_box[2] - d_box[0]))
            d_h = float(max(1, d_box[3] - d_box[1]))
            d_cx = (d_box[0] + d_box[2]) * 0.5
            d_cy = (d_box[1] + d_box[3]) * 0.5

            step_frames = tr.misses + 1
            inst_vx = (d_cx - tr.cx) / step_frames
            inst_vy = (d_cy - tr.cy) / step_frames
            tr.vx = 0.65 * inst_vx + 0.35 * tr.vx
            tr.vy = 0.65 * inst_vy + 0.35 * tr.vy

            tr.box = d_box
            tr.cx = d_cx
            tr.cy = d_cy
            tr.w = d_w
            tr.h = d_h
            tr.last_conf = conf
            tr.ema_conf = self.ema_alpha * conf + (1.0 - self.ema_alpha) * tr.ema_conf
            tr.cls_name = str(det.get("class", tr.cls_name))
            tr.misses = 0
            tr.age += 1
            tr.hit_history.append(1)

            if det.get("verifier_prob") is not None:
                vp = float(det["verifier_prob"])
                if tr.verifier_ema is None:
                    tr.verifier_ema = vp
                else:
                    tr.verifier_ema = self.verifier_ema_alpha * vp + (1.0 - self.verifier_ema_alpha) * tr.verifier_ema
            if det.get("verifier_label") is not None:
                tr.verifier_label = str(det["verifier_label"])

            self._update_static_state(tr, curr_gray)
            active_pairs.append((tr, det))

        for tid in unmatched_tracks:
            tr = self._tracks[tid]
            tr.misses += 1
            tr.age += 1
            tr.vx *= 0.8
            tr.vy *= 0.8
            tr.cx += tr.vx
            tr.cy += tr.vy
            tr.hit_history.append(0)
            if tr.misses > self.max_age:
                del self._tracks[tid]

        for d_idx in unmatched_dets:
            det = detections[d_idx]
            d_box = [int(v) for v in det["box"]]
            conf = float(det["confidence"])
            d_w = float(max(1, d_box[2] - d_box[0]))
            d_h = float(max(1, d_box[3] - d_box[1]))
            d_cx = (d_box[0] + d_box[2]) * 0.5
            d_cy = (d_box[1] + d_box[3]) * 0.5

            history: deque[int] = deque([1], maxlen=self.window_size)
            v_prob = float(det["verifier_prob"]) if det.get("verifier_prob") is not None else None
            v_label = str(det["verifier_label"]) if det.get("verifier_label") is not None else None

            tid = self._next_id
            self._next_id += 1
            tr = TrackState(
                track_id=tid,
                box=d_box,
                cx=d_cx,
                cy=d_cy,
                w=d_w,
                h=d_h,
                ema_conf=conf,
                last_conf=conf,
                cls_name=str(det.get("class", "weapon")),
                hit_history=history,
                anchor_box=list(d_box),
                verifier_ema=v_prob,
                verifier_label=v_label,
            )
            self._tracks[tid] = tr
            active_pairs.append((tr, det))

        if self.static_suppress and curr_gray is not None:
            self._prev_gray = curr_gray

        confirmed: list[dict[str, Any]] = []
        for tr, det in active_pairs:
            if tr.quarantined:
                continue

            hits = sum(tr.hit_history)
            hits_ok = (self.min_hits <= 1) or (hits >= self.min_hits) or (tr.last_conf >= self.high_conf_bypass)
            if not hits_ok:
                continue

            if tr.verifier_ema is not None and tr.verifier_ema < self.verify_thresh and tr.last_conf < self.high_conf_bypass:
                continue

            out_det = dict(det)
            out_det["track_id"] = tr.track_id
            out_det["ema_confidence"] = round(tr.ema_conf, 4)
            if tr.verifier_ema is not None:
                out_det["verifier_ema"] = round(tr.verifier_ema, 4)
            confirmed.append(out_det)

        return confirmed
