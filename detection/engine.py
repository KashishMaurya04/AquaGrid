"""
detection/engine.py – Explainable Sliding-Window Leak Detection Engine for AquaGrid.

Implements an explainable statistical sliding-window anomaly detection algorithm:
  1. Separation of Observation Window from Baseline History:
     - The sliding observation window contains the most recent telemetry readings
       (default: 5 readings).
     - Baseline history consists strictly of readings recorded prior to the observation
       window.
  2. Simulation Scenario Provenance Validation:
     - Only verified 'normal' scenario readings are eligible for normal baseline calculations.
     - 'leak' scenario readings are strictly excluded from baseline calculations.
     - Unverified/legacy readings without known scenario provenance (scenario is NULL)
       are handled transparently and excluded from baseline calculations.
     - If fewer than min_baseline_samples (default: 5) verified normal readings exist in history,
       the sensor reports 'insufficient_data' with an explicit explanation.
  3. Independent Flow and Pressure Analysis:
     - Flow Meters: Detects statistically significant flow surges (+25% to +85%)
       above the sensor's established normal baseline.
     - Pressure Sensors: Detects statistically significant pressure drops (-20% to -60%)
       below the sensor's established normal baseline.
  4. False-Positive Suppression:
     - Enforces a consecutive abnormal observations requirement (default: 3) so single
       transient spikes or sensor jitter are rejected.
  5. Multi-Sensor Evidence Correlation:
     - A pipe is classified as 'suspected_leak' when coincident flow surge and pressure
       drop occur on the same pipe, or when a sustained persistent anomaly occurs on a pipe
       monitored by a single sensor type.
     - Does NOT assume 1 isolated unusual reading proves a physical leak.
  6. Measurement Unit Consistency:
     - Analyses are performed independently per-sensor in that sensor's native registered
       unit (L/min, m³/h, L/s, psi, bar, kPa) without cross-unit mixing.
  7. Transparent Evidence & Metrics:
     - Reports baseline sample count, observation window sample count, baseline average,
       current window average, statistical z-scores, and explainable evidence.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence


def _parse_iso(ts: str | None) -> datetime | None:
    """Safely parse ISO-8601 timestamp string into UTC datetime object."""
    if not ts:
        return None
    try:
        clean_ts = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _is_strictly_after(ts_a: str, ts_cutoff: str) -> bool:
    """Return True if ts_a is chronologically strictly after ts_cutoff."""
    dt_a = _parse_iso(ts_a)
    dt_c = _parse_iso(ts_cutoff)
    if dt_a is not None and dt_c is not None:
        return dt_a > dt_c
    return str(ts_a) > str(ts_cutoff)


@dataclass
class DetectorConfig:
    """Configurable thresholds and window parameters for leak detection."""
    # Sliding window and sample requirements
    window_size: int = 5
    min_baseline_samples: int = 5
    consecutive_abnormal_required: int = 3

    # Flow anomaly thresholds
    flow_ratio_threshold: float = 1.25       # Observed flow >= 1.25x baseline (+25%)
    flow_z_threshold: float = 2.5            # Z-score >= +2.5 standard deviations

    # Pressure anomaly thresholds
    pressure_ratio_threshold: float = 0.80   # Observed pressure <= 0.80x baseline (-20% drop)
    pressure_z_threshold: float = 2.0        # Z-score <= -2.0 standard deviations

    # Database event recording
    auto_record_events: bool = True


DEFAULT_DETECTOR_CONFIG = DetectorConfig()


@dataclass
class SensorAnalysisResult:
    """Detailed statistical anomaly analysis for an individual sensor."""
    sensor_id: int
    sensor_name: str
    sensor_type: str                         # "flow_meter" or "pressure_sensor"
    pipe_id: int
    pipe_name: str
    unit: str
    sample_count: int                        # Total readings recorded
    baseline_samples: int = 0                # Verified normal readings in baseline history
    window_samples: int = 0                  # Readings in current observation window
    unknown_provenance_samples: int = 0      # Historical readings with unknown scenario
    leak_scenario_samples: int = 0           # Historical readings with leak scenario excluded from baseline
    baseline_mean: float | None = None       # Normal baseline average
    baseline_std: float | None = None
    window_mean: float | None = None         # Current window average
    anomaly_score: float | None = None       # Statistical z-score
    deviation_ratio: float | None = None     # window_mean / baseline_mean
    consecutive_abnormal: int = 0
    is_abnormal: bool = False
    classification: str = "normal"           # "normal", "elevated_flow", "reduced_pressure", "insufficient_data", "inactive"
    evidence: str = ""
    threshold_used: str = ""


@dataclass
class PipeDiagnostic:
    """Pipe-level leak assessment correlating flow and pressure sensor evidence."""
    pipe_id: int
    pipe_name: str
    from_node_name: str
    to_node_name: str
    status: str                              # "suspected_leak", "normal", "insufficient_data"
    severity: str = "low"                    # "low", "medium", "high", "critical"
    evidence: str = ""
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    flow_result: SensorAnalysisResult | None = None
    pressure_result: SensorAnalysisResult | None = None

    flow_score: float | None = None
    pressure_score: float | None = None
    baseline_flow: float | None = None
    observed_flow: float | None = None
    baseline_flow_samples: int = 0
    window_flow_samples: int = 0

    baseline_pressure: float | None = None
    observed_pressure: float | None = None
    baseline_pressure_samples: int = 0
    window_pressure_samples: int = 0

    flow_unit: str | None = None
    pressure_unit: str | None = None

    is_simulated: bool = True
    event_id: int | None = None
    is_new_event: bool = False


class LeakDetector:
    """
    Sliding-window anomaly detection engine for municipal water networks.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DEFAULT_DETECTOR_CONFIG

    # ── Sensor-Level Sliding-Window Analysis ──────────────────────────────────

    def analyze_sensor(
        self,
        sensor: dict[str, Any] | Any,
        readings: Sequence[dict[str, Any] | Any],
        resolved_cutoff: str | None = None,
    ) -> SensorAnalysisResult:
        """
        Analyze recent readings for a single sensor using a fixed-size sliding window.

        Separates the active observation window from baseline history.
        Only verified 'normal' scenario readings are eligible for the normal baseline.
        Leak-scenario readings and unverified legacy readings are strictly excluded.
        If resolved_cutoff is provided, observation window and consecutive abnormal
        evaluations are strictly constrained to fresh post-resolution telemetry.

        Args:
            sensor: Sensor metadata (sensor_id, sensor_name, sensor_type, pipe_id, unit, status).
            readings: Raw readings from database (newest first or chronological).
            resolved_cutoff: ISO timestamp string of the latest resolved event on this pipe.

        Returns:
            SensorAnalysisResult with baseline, window statistics, z-score, and evidence.
        """
        sensor_id = int(sensor["sensor_id"])
        sensor_name = str(sensor["sensor_name"])
        sensor_type = str(sensor["sensor_type"])
        pipe_id = int(sensor["pipe_id"])
        pipe_name = str(sensor.get("pipe_name", f"Pipe #{pipe_id}"))
        unit = str(sensor["unit"])

        # Check sensor active status
        if sensor["status"] != "active":
            return SensorAnalysisResult(
                sensor_id=sensor_id,
                sensor_name=sensor_name,
                sensor_type=sensor_type,
                pipe_id=pipe_id,
                pipe_name=pipe_name,
                unit=unit,
                sample_count=len(readings),
                classification="inactive",
                evidence=f"Sensor #{sensor_id} is inactive.",
            )

        # Sort readings into chronological order (oldest to newest)
        chrono = sorted(readings, key=lambda r: (r.get("recorded_at", ""), r.get("reading_id", 0)))
        total_samples = len(chrono)

        if total_samples == 0:
            return SensorAnalysisResult(
                sensor_id=sensor_id,
                sensor_name=sensor_name,
                sensor_type=sensor_type,
                pipe_id=pipe_id,
                pipe_name=pipe_name,
                unit=unit,
                sample_count=0,
                classification="insufficient_data",
                evidence=f"No telemetry readings recorded for sensor #{sensor_id}.",
                threshold_used=f"min_samples={self.config.min_baseline_samples}",
            )

        # ── SEPARATE OBSERVATION WINDOW FROM BASELINE HISTORY ─────────────────
        if resolved_cutoff:
            fresh_readings = [r for r in chrono if _is_strictly_after(r.get("recorded_at", ""), resolved_cutoff)]

            # Baseline history: verified normal readings prior to or at resolved_cutoff
            normal_history = [
                r for r in chrono
                if r.get("scenario") == "normal" and not _is_strictly_after(r.get("recorded_at", ""), resolved_cutoff)
            ]
            # If fresh_readings has more than window_size readings, normal readings prior to the fresh window
            if len(fresh_readings) > self.config.window_size:
                extra_normal = [r for r in fresh_readings[:-self.config.window_size] if r.get("scenario") == "normal"]
                normal_history.extend(extra_normal)

            baseline_samples = len(normal_history)
            base_mean = statistics.mean([float(r["reading_value"]) for r in normal_history]) if normal_history else None
            base_std = statistics.stdev([float(r["reading_value"]) for r in normal_history]) if len(normal_history) > 1 else 0.0

            # Post-resolution telemetry freshness requirement
            if len(fresh_readings) < self.config.window_size:
                clean_cutoff = resolved_cutoff[:19].replace("T", " ")
                win_mean = statistics.mean([float(r["reading_value"]) for r in fresh_readings]) if fresh_readings else None
                return SensorAnalysisResult(
                    sensor_id=sensor_id,
                    sensor_name=sensor_name,
                    sensor_type=sensor_type,
                    pipe_id=pipe_id,
                    pipe_name=pipe_name,
                    unit=unit,
                    sample_count=total_samples,
                    baseline_samples=baseline_samples,
                    window_samples=len(fresh_readings),
                    baseline_mean=round(base_mean, 2) if base_mean is not None else None,
                    baseline_std=round(base_std, 2) if base_std is not None else None,
                    window_mean=round(win_mean, 2) if win_mean is not None else None,
                    classification="insufficient_data",
                    evidence=(
                        f"Insufficient fresh post-resolution telemetry ({len(fresh_readings)}/{self.config.window_size} required) "
                        f"following event resolution at {clean_cutoff}."
                    ),
                    threshold_used=f"min_fresh_samples={self.config.window_size}",
                )

            # We have at least window_size fresh post-resolution readings
            window_slice = fresh_readings[-self.config.window_size:]
            window_vals = [float(r["reading_value"]) for r in window_slice]
            win_mean = statistics.mean(window_vals)
            window_samples = len(window_vals)

            if baseline_samples < self.config.min_baseline_samples:
                return SensorAnalysisResult(
                    sensor_id=sensor_id,
                    sensor_name=sensor_name,
                    sensor_type=sensor_type,
                    pipe_id=pipe_id,
                    pipe_name=pipe_name,
                    unit=unit,
                    sample_count=total_samples,
                    baseline_samples=baseline_samples,
                    window_samples=window_samples,
                    window_mean=round(win_mean, 2),
                    classification="insufficient_data",
                    evidence=(
                        f"Insufficient verified normal baseline samples ({baseline_samples}/{self.config.min_baseline_samples} "
                        f"minimum required in history)."
                    ),
                    threshold_used=f"min_samples={self.config.min_baseline_samples}",
                )

            baseline_vals = [float(r["reading_value"]) for r in normal_history]
            base_mean = statistics.mean(baseline_vals)
            base_std = statistics.stdev(baseline_vals) if len(baseline_vals) > 1 else 0.0
            unknown_samples = 0
            leak_samples = 0

        else:
            # Window contains the most recent observations up to window_size
            window_slice = chrono[-self.config.window_size:]
            window_vals = [float(r["reading_value"]) for r in window_slice]
            win_mean = statistics.mean(window_vals)
            window_samples = len(window_vals)

            # Baseline history contains observations recorded strictly prior to the window
            history_slice = chrono[:-window_samples]

            # Partition history by simulation scenario provenance
            normal_history = [r for r in history_slice if r.get("scenario") == "normal"]
            leak_history = [r for r in history_slice if r.get("scenario") == "leak"]
            unknown_history = [r for r in history_slice if r.get("scenario") not in ("normal", "leak")]

            baseline_samples = len(normal_history)
            leak_samples = len(leak_history)
            unknown_samples = len(unknown_history)

            # Check if sufficient verified normal baseline samples exist
            if baseline_samples < self.config.min_baseline_samples:
                explanation_parts = [
                    f"Insufficient verified normal baseline samples ({baseline_samples}/{self.config.min_baseline_samples} "
                    f"minimum required in history prior to observation window)."
                ]
                if unknown_samples > 0:
                    explanation_parts.append(
                        f"{unknown_samples} historical reading(s) have unverified/unknown scenario provenance and cannot be used for baselining."
                    )
                if leak_samples > 0:
                    explanation_parts.append(
                        f"{leak_samples} leak-scenario reading(s) excluded from baseline."
                    )

                return SensorAnalysisResult(
                    sensor_id=sensor_id,
                    sensor_name=sensor_name,
                    sensor_type=sensor_type,
                    pipe_id=pipe_id,
                    pipe_name=pipe_name,
                    unit=unit,
                    sample_count=total_samples,
                    baseline_samples=baseline_samples,
                    window_samples=window_samples,
                    unknown_provenance_samples=unknown_samples,
                    leak_scenario_samples=leak_samples,
                    window_mean=round(win_mean, 2),
                    classification="insufficient_data",
                    evidence=" ".join(explanation_parts),
                    threshold_used=f"min_samples={self.config.min_baseline_samples}",
                )

            # Establish baseline statistics purely from verified normal history
            baseline_vals = [float(r["reading_value"]) for r in normal_history]
            base_mean = statistics.mean(baseline_vals)
            base_std = statistics.stdev(baseline_vals) if len(baseline_vals) > 1 else 0.0

        effective_std = base_std if base_std > 0.001 else max(0.01 * abs(base_mean), 0.5)
        deviation_ratio = win_mean / base_mean if base_mean != 0 else 1.0

        # ── FLOW METER ANOMALY EVALUATION ─────────────────────────────────────
        if sensor_type == "flow_meter":
            z_score = (win_mean - base_mean) / effective_std

            consecutive = 0
            for val in reversed(window_vals):
                val_z = (val - base_mean) / effective_std
                val_ratio = val / base_mean if base_mean != 0 else 1.0
                if val_ratio >= self.config.flow_ratio_threshold or val_z >= self.config.flow_z_threshold:
                    consecutive += 1
                else:
                    break

            is_abnormal = (
                consecutive >= self.config.consecutive_abnormal_required
                and (deviation_ratio >= self.config.flow_ratio_threshold or z_score >= self.config.flow_z_threshold)
            )

            threshold_str = (
                f"ratio>={self.config.flow_ratio_threshold:.2f}x OR z>={self.config.flow_z_threshold:.1f} "
                f"(min {self.config.consecutive_abnormal_required} consecutive)"
            )

            if is_abnormal:
                pct_surge = (deviation_ratio - 1.0) * 100.0
                evidence = (
                    f"Flow surge detected: window average {win_mean:.2f} {unit} (win={window_samples}) is "
                    f"+{pct_surge:.1f}% above verified baseline {base_mean:.2f} {unit} (n={baseline_samples} normal samples, "
                    f"z-score: +{z_score:.2f}, {consecutive} consecutive readings exceeding threshold)."
                )
                classification = "elevated_flow"
            else:
                evidence = (
                    f"Flow is within normal bounds: window average {win_mean:.2f} {unit} (win={window_samples}) "
                    f"vs verified baseline {base_mean:.2f} {unit} (n={baseline_samples} normal samples, z-score: {z_score:+.2f})."
                )
                classification = "normal"

        # ── PRESSURE SENSOR ANOMALY EVALUATION ─────────────────────────────────
        elif sensor_type == "pressure_sensor":
            z_score = (win_mean - base_mean) / effective_std

            consecutive = 0
            for val in reversed(window_vals):
                val_z = (val - base_mean) / effective_std
                val_ratio = val / base_mean if base_mean != 0 else 1.0
                if val_ratio <= self.config.pressure_ratio_threshold or val_z <= -self.config.pressure_z_threshold:
                    consecutive += 1
                else:
                    break

            is_abnormal = (
                consecutive >= self.config.consecutive_abnormal_required
                and (deviation_ratio <= self.config.pressure_ratio_threshold or z_score <= -self.config.pressure_z_threshold)
            )

            threshold_str = (
                f"ratio<={self.config.pressure_ratio_threshold:.2f}x OR z<=-{self.config.pressure_z_threshold:.1f} "
                f"(min {self.config.consecutive_abnormal_required} consecutive)"
            )

            if is_abnormal:
                pct_drop = (1.0 - deviation_ratio) * 100.0
                evidence = (
                    f"Pressure drop detected: window average {win_mean:.2f} {unit} (win={window_samples}) is "
                    f"-{pct_drop:.1f}% below verified baseline {base_mean:.2f} {unit} (n={baseline_samples} normal samples, "
                    f"z-score: {z_score:.2f}, {consecutive} consecutive readings below threshold)."
                )
                classification = "reduced_pressure"
            else:
                evidence = (
                    f"Pressure is within normal bounds: window average {win_mean:.2f} {unit} (win={window_samples}) "
                    f"vs verified baseline {base_mean:.2f} {unit} (n={baseline_samples} normal samples, z-score: {z_score:+.2f})."
                )
                classification = "normal"

        else:
            is_abnormal = False
            z_score = 0.0
            consecutive = 0
            threshold_str = "N/A"
            classification = "normal"
            evidence = f"Sensor type '{sensor_type}' is not monitored for hydraulic anomalies."

        return SensorAnalysisResult(
            sensor_id=sensor_id,
            sensor_name=sensor_name,
            sensor_type=sensor_type,
            pipe_id=pipe_id,
            pipe_name=pipe_name,
            unit=unit,
            sample_count=total_samples,
            baseline_samples=baseline_samples,
            window_samples=window_samples,
            unknown_provenance_samples=unknown_samples,
            leak_scenario_samples=leak_samples,
            baseline_mean=round(base_mean, 2),
            baseline_std=round(base_std, 2),
            window_mean=round(win_mean, 2),
            anomaly_score=round(z_score, 2),
            deviation_ratio=round(deviation_ratio, 3),
            consecutive_abnormal=consecutive,
            is_abnormal=is_abnormal,
            classification=classification,
            evidence=evidence,
            threshold_used=threshold_str,
        )

    # ── Pipe-Level Evidence Correlation ───────────────────────────────────────

    def diagnose_pipe(
        self,
        pipe: dict[str, Any] | Any,
        pipe_sensors: Sequence[dict[str, Any] | Any],
        readings_by_sensor: dict[int, Sequence[dict[str, Any] | Any]],
        resolved_cutoff: str | None = None,
    ) -> PipeDiagnostic:
        """
        Correlate flow and pressure sensor evidence across a monitored pipe.

        Args:
            pipe: Pipe row dict (id, name, from_node_name, to_node_name).
            pipe_sensors: List of sensors installed on this pipe.
            readings_by_sensor: Map of sensor_id -> list of raw readings.
            resolved_cutoff: ISO timestamp string of the latest resolved event on this pipe.

        Returns:
            PipeDiagnostic with correlated status, severity, and evidence explanation.
        """
        pipe_id = int(pipe["id"])
        pipe_name = str(pipe["name"])
        from_node = str(pipe.get("from_node_name", "From"))
        to_node = str(pipe.get("to_node_name", "To"))

        active_sensors = [s for s in pipe_sensors if s["status"] == "active"]

        # Case 1: No active sensors on this pipe
        if not active_sensors:
            return PipeDiagnostic(
                pipe_id=pipe_id,
                pipe_name=pipe_name,
                from_node_name=from_node,
                to_node_name=to_node,
                status="insufficient_data",
                severity="low",
                evidence=f"No active telemetry sensors installed on pipe '{pipe_name}' (#{pipe_id}).",
            )

        # Analyze each active sensor on this pipe
        sensor_results: list[SensorAnalysisResult] = []
        for s in active_sensors:
            s_readings = readings_by_sensor.get(int(s["sensor_id"]), [])
            res = self.analyze_sensor(s, s_readings, resolved_cutoff=resolved_cutoff)
            sensor_results.append(res)

        flow_res = next((r for r in sensor_results if r.sensor_type == "flow_meter"), None)
        press_res = next((r for r in sensor_results if r.sensor_type == "pressure_sensor"), None)

        flow_score = flow_res.anomaly_score if flow_res else None
        press_score = press_res.anomaly_score if press_res else None
        base_flow = flow_res.baseline_mean if flow_res else None
        obs_flow = flow_res.window_mean if flow_res else None
        flow_base_samples = flow_res.baseline_samples if flow_res else 0
        flow_win_samples = flow_res.window_samples if flow_res else 0

        base_press = press_res.baseline_mean if press_res else None
        obs_press = press_res.window_mean if press_res else None
        press_base_samples = press_res.baseline_samples if press_res else 0
        press_win_samples = press_res.window_samples if press_res else 0

        flow_unit = flow_res.unit if flow_res else None
        press_unit = press_res.unit if press_res else None

        # Case 2: Insufficient data on all sensors
        valid_results = [r for r in sensor_results if r.classification != "insufficient_data"]
        if not valid_results:
            sample_summary = "; ".join(f"{r.sensor_name}: {r.evidence}" for r in sensor_results)
            return PipeDiagnostic(
                pipe_id=pipe_id,
                pipe_name=pipe_name,
                from_node_name=from_node,
                to_node_name=to_node,
                status="insufficient_data",
                severity="low",
                evidence=f"Insufficient baseline reading history on pipe '{pipe_name}': {sample_summary}",
                flow_result=flow_res,
                pressure_result=press_res,
                baseline_flow=base_flow,
                observed_flow=obs_flow,
                baseline_flow_samples=flow_base_samples,
                window_flow_samples=flow_win_samples,
                baseline_pressure=base_press,
                observed_pressure=obs_press,
                baseline_pressure_samples=press_base_samples,
                window_pressure_samples=press_win_samples,
                flow_unit=flow_unit,
                pressure_unit=press_unit,
            )

        # Case 3: If one sensor on a dual-monitored pipe has insufficient baseline history
        if any(r.classification == "insufficient_data" for r in sensor_results):
            insufficient_reasons = "; ".join(
                f"{r.sensor_name}: {r.evidence}" for r in sensor_results if r.classification == "insufficient_data"
            )
            return PipeDiagnostic(
                pipe_id=pipe_id,
                pipe_name=pipe_name,
                from_node_name=from_node,
                to_node_name=to_node,
                status="insufficient_data",
                severity="low",
                evidence=f"Pending baseline verification on pipe '{pipe_name}': {insufficient_reasons}",
                flow_result=flow_res,
                pressure_result=press_res,
                flow_score=flow_score,
                pressure_score=press_score,
                baseline_flow=base_flow,
                observed_flow=obs_flow,
                baseline_flow_samples=flow_base_samples,
                window_flow_samples=flow_win_samples,
                baseline_pressure=base_press,
                observed_pressure=obs_press,
                baseline_pressure_samples=press_base_samples,
                window_pressure_samples=press_win_samples,
                flow_unit=flow_unit,
                pressure_unit=press_unit,
            )

        is_simulated = True

        # ── Cross-Sensor Correlation Logic ────────────────────────────────────
        flow_abnormal = (flow_res is not None and flow_res.is_abnormal)
        press_abnormal = (press_res is not None and press_res.is_abnormal)

        # Multi-sensor supporting evidence: Both Flow Surge AND Pressure Drop
        if flow_abnormal and press_abnormal:
            severity = "critical" if (flow_res.deviation_ratio >= 1.50 or press_res.deviation_ratio <= 0.60) else "high"
            evidence = (
                f"Coincident flow surge and pressure drop on pipe '{pipe_name}' (#{pipe_id}): "
                f"{flow_res.evidence} | {press_res.evidence} "
                f"Both hydraulic signatures persistently agree across {self.config.consecutive_abnormal_required} consecutive observations."
            )
            return PipeDiagnostic(
                pipe_id=pipe_id,
                pipe_name=pipe_name,
                from_node_name=from_node,
                to_node_name=to_node,
                status="suspected_leak",
                severity=severity,
                evidence=evidence,
                flow_result=flow_res,
                pressure_result=press_res,
                flow_score=flow_score,
                pressure_score=press_score,
                baseline_flow=base_flow,
                observed_flow=obs_flow,
                baseline_flow_samples=flow_base_samples,
                window_flow_samples=flow_win_samples,
                baseline_pressure=base_press,
                observed_pressure=obs_press,
                baseline_pressure_samples=press_base_samples,
                window_pressure_samples=press_win_samples,
                flow_unit=flow_unit,
                pressure_unit=press_unit,
                is_simulated=is_simulated,
            )

        # Single sensor anomaly: Persistent severe surge or drop on a pipe with single sensor type
        if flow_abnormal and press_res is None:
            severity = "high" if flow_res.deviation_ratio >= 1.50 else "medium"
            evidence = (
                f"Single-sensor leak anomaly on pipe '{pipe_name}' (#{pipe_id}): "
                f"{flow_res.evidence} (No pressure sensor installed on this pipe for cross-validation)."
            )
            return PipeDiagnostic(
                pipe_id=pipe_id,
                pipe_name=pipe_name,
                from_node_name=from_node,
                to_node_name=to_node,
                status="suspected_leak",
                severity=severity,
                evidence=evidence,
                flow_result=flow_res,
                pressure_result=press_res,
                flow_score=flow_score,
                pressure_score=press_score,
                baseline_flow=base_flow,
                observed_flow=obs_flow,
                baseline_flow_samples=flow_base_samples,
                window_flow_samples=flow_win_samples,
                baseline_pressure=base_press,
                observed_pressure=obs_press,
                baseline_pressure_samples=press_base_samples,
                window_pressure_samples=press_win_samples,
                flow_unit=flow_unit,
                pressure_unit=press_unit,
                is_simulated=is_simulated,
            )

        if press_abnormal and flow_res is None:
            severity = "high" if press_res.deviation_ratio <= 0.60 else "medium"
            evidence = (
                f"Single-sensor leak anomaly on pipe '{pipe_name}' (#{pipe_id}): "
                f"{press_res.evidence} (No flow meter installed on this pipe for cross-validation)."
            )
            return PipeDiagnostic(
                pipe_id=pipe_id,
                pipe_name=pipe_name,
                from_node_name=from_node,
                to_node_name=to_node,
                status="suspected_leak",
                severity=severity,
                evidence=evidence,
                flow_result=flow_res,
                pressure_result=press_res,
                flow_score=flow_score,
                pressure_score=press_score,
                baseline_flow=base_flow,
                observed_flow=obs_flow,
                baseline_flow_samples=flow_base_samples,
                window_flow_samples=flow_win_samples,
                baseline_pressure=base_press,
                observed_pressure=obs_press,
                baseline_pressure_samples=press_base_samples,
                window_pressure_samples=press_win_samples,
                flow_unit=flow_unit,
                pressure_unit=press_unit,
                is_simulated=is_simulated,
            )

        # Isolated anomaly on dual-monitored pipe without companion confirmation
        if flow_abnormal and not press_abnormal:
            evidence = (
                f"Isolated flow surge on pipe '{pipe_name}' without supporting pressure drop: "
                f"{flow_res.evidence} Pressure remains normal ({press_res.window_mean} {press_res.unit}). "
                f"Classified as unconfirmed operational variation."
            )
            return PipeDiagnostic(
                pipe_id=pipe_id,
                pipe_name=pipe_name,
                from_node_name=from_node,
                to_node_name=to_node,
                status="normal",
                severity="low",
                evidence=evidence,
                flow_result=flow_res,
                pressure_result=press_res,
                flow_score=flow_score,
                pressure_score=press_score,
                baseline_flow=base_flow,
                observed_flow=obs_flow,
                baseline_flow_samples=flow_base_samples,
                window_flow_samples=flow_win_samples,
                baseline_pressure=base_press,
                observed_pressure=obs_press,
                baseline_pressure_samples=press_base_samples,
                window_pressure_samples=press_win_samples,
                flow_unit=flow_unit,
                pressure_unit=press_unit,
                is_simulated=is_simulated,
            )

        if press_abnormal and not flow_abnormal:
            evidence = (
                f"Isolated pressure drop on pipe '{pipe_name}' without supporting flow surge: "
                f"{press_res.evidence} Flow remains normal ({flow_res.window_mean} {flow_res.unit}). "
                f"Classified as unconfirmed operational variation."
            )
            return PipeDiagnostic(
                pipe_id=pipe_id,
                pipe_name=pipe_name,
                from_node_name=from_node,
                to_node_name=to_node,
                status="normal",
                severity="low",
                evidence=evidence,
                flow_result=flow_res,
                pressure_result=press_res,
                flow_score=flow_score,
                pressure_score=press_score,
                baseline_flow=base_flow,
                observed_flow=obs_flow,
                baseline_flow_samples=flow_base_samples,
                window_flow_samples=flow_win_samples,
                baseline_pressure=base_press,
                observed_pressure=obs_press,
                baseline_pressure_samples=press_base_samples,
                window_pressure_samples=press_win_samples,
                flow_unit=flow_unit,
                pressure_unit=press_unit,
                is_simulated=is_simulated,
            )

        # Normal operation: Both or all sensors are within baseline bounds
        normal_parts: list[str] = []
        if flow_res:
            normal_parts.append(
                f"Flow: {flow_res.window_mean:.1f} {flow_res.unit} (baseline {flow_res.baseline_mean:.1f}, n={flow_res.baseline_samples})"
            )
        if press_res:
            normal_parts.append(
                f"Pressure: {press_res.window_mean:.1f} {press_res.unit} (baseline {press_res.baseline_mean:.1f}, n={press_res.baseline_samples})"
            )

        return PipeDiagnostic(
            pipe_id=pipe_id,
            pipe_name=pipe_name,
            from_node_name=from_node,
            to_node_name=to_node,
            status="normal",
            severity="low",
            evidence=f"Operating within normal statistical baseline bounds ({', '.join(normal_parts)}).",
            flow_result=flow_res,
            pressure_result=press_res,
            flow_score=flow_score,
            pressure_score=press_score,
            baseline_flow=base_flow,
            observed_flow=obs_flow,
            baseline_flow_samples=flow_base_samples,
            window_flow_samples=flow_win_samples,
            baseline_pressure=base_press,
            observed_pressure=obs_press,
            baseline_pressure_samples=press_base_samples,
            window_pressure_samples=press_win_samples,
            flow_unit=flow_unit,
            pressure_unit=press_unit,
            is_simulated=is_simulated,
        )

    # ── Network-Wide Detection Execution ──────────────────────────────────────

    def run_detection_network(
        self,
        db_module: Any,
        auto_record: bool | None = None,
    ) -> list[PipeDiagnostic]:
        """
        Run anomaly detection across all registered pipes in the database.
        Optionally records newly identified leak events in the leak_events table.

        Args:
            db_module: The database module (e.g. database.py).
            auto_record: If True, persist newly detected leaks to SQLite.
                         If False, compute diagnostics in memory (strictly read-only).
                         If None, defaults to self.config.auto_record_events.

        Returns:
            List of PipeDiagnostic objects for every monitored pipe.
        """
        all_pipes = db_module.get_all_pipes()
        all_sensors = db_module.get_all_sensors()

        # Group sensors by pipe_id
        sensors_by_pipe: dict[int, list[dict[str, Any]]] = {}
        for s in all_sensors:
            p_id = int(s["pipe_id"])
            sensors_by_pipe.setdefault(p_id, []).append(dict(s))

        # Fetch recent readings for all active sensors (limit=100 per sensor)
        readings_by_sensor: dict[int, list[dict[str, Any]]] = {}
        for s in all_sensors:
            if s["status"] == "active":
                s_id = int(s["sensor_id"])
                readings = db_module.get_readings_for_sensor(s_id, limit=100)
                readings_by_sensor[s_id] = [dict(r) for r in readings]

        diagnostics: list[PipeDiagnostic] = []

        for p in all_pipes:
            p_id = int(p["id"])
            p_sensors = sensors_by_pipe.get(p_id, [])

            resolved_cutoff = None
            if hasattr(db_module, "get_latest_resolved_leak_event"):
                latest_res = db_module.get_latest_resolved_leak_event(p_id)
                if latest_res:
                    resolved_cutoff = latest_res["resolved_at"] or latest_res["detected_at"]

            diag = self.diagnose_pipe(dict(p), p_sensors, readings_by_sensor, resolved_cutoff=resolved_cutoff)

            # Map active event_id if an open/investigating event already exists for this pipe
            if hasattr(db_module, "get_open_leak_events"):
                try:
                    for ev in db_module.get_open_leak_events():
                        if int(ev["pipe_id"]) == p_id:
                            diag.event_id = int(ev["event_id"])
                            break
                except Exception:
                    pass

            # Automatically persist suspected leaks to database if recording is enabled
            should_record = auto_record if auto_record is not None else self.config.auto_record_events
            if diag.status == "suspected_leak" and should_record:
                base_str_parts = []
                if diag.baseline_flow is not None:
                    base_str_parts.append(f"Flow: {diag.baseline_flow} {diag.flow_unit or ''} (n={diag.baseline_flow_samples})")
                if diag.baseline_pressure is not None:
                    base_str_parts.append(f"Pressure: {diag.baseline_pressure} {diag.pressure_unit or ''} (n={diag.baseline_pressure_samples})")

                obs_str_parts = []
                if diag.observed_flow is not None:
                    obs_str_parts.append(f"Flow: {diag.observed_flow} {diag.flow_unit or ''} (win={diag.window_flow_samples})")
                if diag.observed_pressure is not None:
                    obs_str_parts.append(f"Pressure: {diag.observed_pressure} {diag.pressure_unit or ''} (win={diag.window_pressure_samples})")

                event_id, is_new = db_module.record_leak_event(
                    pipe_id=diag.pipe_id,
                    severity=diag.severity,
                    evidence_description=diag.evidence,
                    flow_anomaly_score=diag.flow_score,
                    pressure_anomaly_score=diag.pressure_score,
                    baseline_values=", ".join(base_str_parts),
                    observed_values=", ".join(obs_str_parts),
                    is_simulated=diag.is_simulated,
                )
                diag.event_id = event_id
                diag.is_new_event = is_new

            diagnostics.append(diag)

        return diagnostics


def build_detection_result_message(diagnostics: Sequence[PipeDiagnostic]) -> tuple[str, str]:
    """
    Generate an accurate, human-readable flash message and category from detection diagnostics.

    Distinguishes three detection outcomes:
    1. SUSPECTED LEAK: If any pipe meets anomaly criteria, return an appropriate warning.
    2. INSUFFICIENT DATA (all): If every monitored pipe has insufficient telemetry, return:
       "Detection completed. Insufficient telemetry is available to determine network health."
    3. MIXED (normal + insufficient data): If some pipes are normal and others have insufficient data,
       communicate both counts accurately with a warning alert.
    4. NORMAL (all): If all monitored pipes are verified normal, return a success notice.

    Returns:
        (message_string, category_string) where category is 'warning', 'success', or 'info'.
    """
    if not diagnostics:
        return (
            "Detection completed: No monitored pipes found in the network.",
            "warning",
        )

    leaks = [d for d in diagnostics if d.status == "suspected_leak"]
    normal_pipes = [d for d in diagnostics if d.status == "normal"]
    insufficient_pipes = [d for d in diagnostics if d.status == "insufficient_data"]

    leak_count = len(leaks)
    normal_count = len(normal_pipes)
    insufficient_count = len(insufficient_pipes)
    total = len(diagnostics)

    # 1. Any pipe has a suspected leak -> Warning alert (not success)
    if leak_count > 0:
        leak_str = f"{leak_count} suspected leak anomaly event{'s' if leak_count != 1 else ''}"
        if insufficient_count > 0:
            msg = (
                f"Detection warning: {leak_str} detected across monitored pipes "
                f"({insufficient_count} pipe{'s' if insufficient_count != 1 else ''} pending telemetry)."
            )
        else:
            msg = f"Detection warning: {leak_str} detected across monitored pipes."
        return (msg, "warning")

    # 2. Every monitored pipe has insufficient data
    if insufficient_count == total:
        return (
            "Detection completed. Insufficient telemetry is available to determine network health.",
            "warning",
        )

    # 3. Mixed: Some pipes normal, others have insufficient data
    if normal_count > 0 and insufficient_count > 0:
        n_str = f"{normal_count} pipe{'s' if normal_count != 1 else ''}"
        i_str = f"{insufficient_count} pipe{'s' if insufficient_count != 1 else ''}"
        has_have = "has" if insufficient_count == 1 else "have"
        msg = (
            f"Detection completed: {n_str} operating within normal baseline bounds; "
            f"{i_str} {has_have} insufficient telemetry to determine health."
        )
        return (msg, "warning")

    # 4. All monitored pipes are verified normal (zero leaks, zero insufficient)
    if normal_count == total:
        return (
            "Detection completed: All active monitored pipes are operating within expected statistical baseline bounds.",
            "success",
        )

    return (
        f"Detection completed: {normal_count} normal, {insufficient_count} pending telemetry.",
        "info",
    )
