"""
repair_queue/scoring.py – Explainable Multi-Factor Hydraulic Anomaly Priority Scoring.

Computes a deterministic, explainable priority score normalized strictly to
the [0.0, 100.0] range based on observed sensor evidence and asset criticality.

Component Decomposition (Total 100 points):
  1. Flow Anomaly Severity (S_flow, max 30 points):
     Proportional to positive z-score deviation: min(30.0, max(0.0, z_flow * 10.0)).
     z >= 3.0 yields the full 30.0 points. Zero/negative z-scores or missing telemetry yield 0.0.

  2. Pressure Anomaly Severity (S_pressure, max 30 points):
     Proportional to pressure drop deviation magnitude: min(30.0, max(0.0, |z_press| * 10.0)).
     |z| >= 3.0 yields the full 30.0 points. Zero deviation or missing telemetry yield 0.0.

  3. Cross-Sensor Detection Severity (S_severity, max 20 points):
     Reflects multi-sensor hydraulic signature agreement from detection engine:
       * 'critical' : 20.0 pts (Coincident flow surge and pressure drop)
       * 'high'     : 15.0 pts (Significant multi-sensor anomaly)
       * 'medium'   : 10.0 pts (Single-sensor anomaly)
       * 'low'      : 5.0 pts  (Minor unconfirmed variation)

  4. Persistence & Consecutive Observations (S_persistence, max 10 points):
     Proportional to sustained breach duration: min(10.0, consecutive * 2.0).
     5+ consecutive readings yield full 10.0 pts.

  5. Pipe Network Importance (S_pipe, max 10 points):
     Derived from pipe capacity (L/min): min(10.0, capacity_lpm / 100.0).
     Capacity >= 1000 L/min yields full 10.0 pts.

Priority Categories:
  - CRITICAL : 80.0 – 100.0 (Immediate emergency dispatch)
  - HIGH     : 60.0 – 79.9  (Urgent investigation required)
  - MEDIUM   : 35.0 – 59.9  (Scheduled field inspection)
  - LOW      : 0.0 – 34.9   (Routine maintenance / monitoring)

AUDIT DISCLAIMER:
  The priority score is a normalized mathematical ranking metric for maintenance dispatch.
  It does NOT quantify monetary losses, structural burst probabilities, or physical damage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

PRIORITY_THRESHOLDS = {
    "CRITICAL": (80.0, 100.0),
    "HIGH": (60.0, 79.9),
    "MEDIUM": (35.0, 59.9),
    "LOW": (0.0, 34.9),
}


@dataclass
class PriorityScore:
    """Represents the decomposed priority score and explainable audit trail."""
    total_score: float
    category: str
    flow_component: float
    pressure_component: float
    severity_component: float
    persistence_component: float
    pipe_importance_component: float
    breakdown: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


def _extract_consecutive_count(evidence: str) -> int:
    """Helper to extract consecutive abnormal readings count from evidence text."""
    if not evidence:
        return 3
    match = re.search(r"(\d+)\s+consecutive", evidence, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 3


def calculate_priority_score(
    leak_event: Mapping[str, Any] | Any | None = None,
    pipe: Mapping[str, Any] | Any | None = None,
    **kwargs: Any,
) -> PriorityScore:
    """
    Calculate an explainable priority score for a suspected leak event.

    Args:
        leak_event: Optional dict or sqlite3.Row of leak_events table.
        pipe: Optional dict or sqlite3.Row of pipes table for the affected pipe.
        **kwargs: Optional direct scoring parameters (flow_z_score, pressure_z_score,
                  event_severity, consecutive_abnormal, pipe_capacity_lpm).

    Returns:
        PriorityScore object with total score, category, breakdown, and explanation.
    """
    event_dict: dict[str, Any] = {}
    if leak_event is not None:
        event_dict = dict(leak_event)

    pipe_dict: dict[str, Any] = {}
    if pipe is not None:
        pipe_dict = dict(pipe)

    # Allow keyword arguments to populate or override fields
    if "flow_z_score" in kwargs:
        flow_raw = kwargs["flow_z_score"]
    elif "flow_anomaly_score" in kwargs:
        flow_raw = kwargs["flow_anomaly_score"]
    else:
        flow_raw = event_dict.get("flow_anomaly_score")

    if "pressure_z_score" in kwargs:
        press_raw = kwargs["pressure_z_score"]
    elif "pressure_anomaly_score" in kwargs:
        press_raw = kwargs["pressure_anomaly_score"]
    else:
        press_raw = event_dict.get("pressure_anomaly_score")

    if "event_severity" in kwargs:
        sev_raw = kwargs["event_severity"]
    elif "severity" in kwargs:
        sev_raw = kwargs["severity"]
    else:
        sev_raw = event_dict.get("severity", "medium")

    if "consecutive_abnormal" in kwargs:
        consec_raw = kwargs["consecutive_abnormal"]
    elif "consecutive_readings" in kwargs:
        consec_raw = kwargs["consecutive_readings"]
    else:
        consec_raw = event_dict.get("consecutive_readings")

    if "pipe_capacity_lpm" in kwargs:
        cap_raw = kwargs["pipe_capacity_lpm"]
    elif "capacity_lpm" in kwargs:
        cap_raw = kwargs["capacity_lpm"]
    else:
        cap_raw = pipe_dict.get("capacity_lpm")

    # 1. Flow Anomaly Severity Component (0 - 30 pts)
    flow_missing = False
    if flow_raw is None:
        flow_pts = 0.0
        flow_missing = True
        flow_note = "Telemetry unavailable (no flow meter on pipe)"
    else:
        flow_val = float(flow_raw)
        if flow_val <= 0.0:
            flow_pts = 0.0
            flow_note = f"Nominal flow (z = {flow_val:+.2f})"
        else:
            flow_pts = round(min(30.0, flow_val * 10.0), 1)
            flow_note = f"Flow surge (z = +{flow_val:.2f}, +{flow_pts:.1f} pts)"

    # 2. Pressure Anomaly Severity Component (0 - 30 pts)
    press_missing = False
    if press_raw is None:
        press_pts = 0.0
        press_missing = True
        press_note = "Telemetry unavailable (no pressure sensor on pipe)"
    else:
        press_val = float(press_raw)
        abs_p = abs(press_val)
        if abs_p <= 0.0:
            press_pts = 0.0
            press_note = f"Nominal pressure (z = {press_val:+.2f})"
        else:
            press_pts = round(min(30.0, abs_p * 10.0), 1)
            press_note = f"Pressure drop (z = -{abs_p:.2f}, +{press_pts:.1f} pts)"

    # 3. Detection Severity Rating (0 - 20 pts)
    sev_str = str(sev_raw or "medium").lower().strip()
    if sev_str == "critical":
        sev_pts = 20.0
        sev_note = "Critical multi-sensor coincidence"
    elif sev_str == "high":
        sev_pts = 15.0
        sev_note = "High multi-sensor anomaly"
    elif sev_str == "medium":
        sev_pts = 10.0
        sev_note = "Medium single-sensor anomaly"
    else:
        sev_pts = 5.0
        sev_note = "Low severity / unconfirmed variation"

    # 4. Persistence & Consecutive Observations (0 - 10 pts)
    if consec_raw is not None:
        try:
            consecutive = int(consec_raw)
        except (ValueError, TypeError):
            consecutive = 3
    else:
        evidence_text = str(event_dict.get("evidence_description", ""))
        consecutive = _extract_consecutive_count(evidence_text)

    if consecutive <= 0 or consec_raw is None and "consecutive_abnormal" in kwargs and kwargs["consecutive_abnormal"] is None:
        pers_pts = 0.0
        pers_note = "No consecutive readings recorded"
    else:
        pers_pts = round(min(10.0, max(0.0, consecutive * 2.0)), 1)
        pers_note = f"{consecutive} consecutive abnormal readings (+{pers_pts:.1f} pts)"

    # 5. Pipe Network Importance (0 - 10 pts)
    if cap_raw is not None:
        try:
            cap_val = float(cap_raw)
            pipe_pts = round(min(10.0, max(0.0, cap_val / 100.0)), 1)
            pipe_note = f"Pipe capacity {cap_val:.0f} L/min (+{pipe_pts:.1f} pts)"
        except (ValueError, TypeError):
            pipe_pts = 5.0
            pipe_note = "Standard default pipe weighting"
    else:
        if "pipe_capacity_lpm" in kwargs and kwargs["pipe_capacity_lpm"] is None:
            pipe_pts = 0.0
            pipe_note = "Pipe capacity unspecified (0 pts)"
        else:
            pipe_pts = 5.0
            pipe_note = "Standard default pipe weighting (capacity unspecified)"

    # Sum components and clamp to [0.0, 100.0]
    total_raw = flow_pts + press_pts + sev_pts + pers_pts + pipe_pts
    final_score = round(min(100.0, max(0.0, total_raw)), 1)

    # Classify Category
    if final_score >= 80.0:
        category = "CRITICAL"
    elif final_score >= 60.0:
        category = "HIGH"
    elif final_score >= 35.0:
        category = "MEDIUM"
    else:
        category = "LOW"

    breakdown = {
        # Standard decomposed keys
        "flow_anomaly_points": flow_pts,
        "flow_note": flow_note,
        "flow_telemetry_missing": flow_missing,
        "pressure_anomaly_points": press_pts,
        "pressure_note": press_note,
        "pressure_telemetry_missing": press_missing,
        "severity_points": sev_pts,
        "severity_note": sev_note,
        "persistence_points": pers_pts,
        "persistence_note": pers_note,
        "pipe_importance_points": pipe_pts,
        "pipe_note": pipe_note,
        "total_score": final_score,
        "category": category,
        # Direct formula aliases
        "flow_surge": flow_pts,
        "pressure_drop": press_pts,
        "event_severity": sev_pts,
        "persistence": pers_pts,
        "pipe_capacity": pipe_pts,
    }

    # Construct narrative explanation
    reasons = [
        f"Flow: +{flow_pts:.1f} pts ({flow_note})",
        f"Pressure: +{press_pts:.1f} pts ({press_note})",
        f"Severity: +{sev_pts:.1f} pts ({sev_note})",
        f"Persistence: +{pers_pts:.1f} pts ({pers_note})",
        f"Pipe Capacity: +{pipe_pts:.1f} pts ({pipe_note})",
    ]
    explanation = (
        f"Priority Score: {final_score}/100 ({category}). "
        + " | ".join(reasons)
        + ". Audit Disclaimer: Normalized operational maintenance ranking; does not model financial or structural damage."
    )

    return PriorityScore(
        total_score=final_score,
        category=category,
        flow_component=flow_pts,
        pressure_component=press_pts,
        severity_component=sev_pts,
        persistence_component=pers_pts,
        pipe_importance_component=pipe_pts,
        breakdown=breakdown,
        explanation=explanation,
    )
