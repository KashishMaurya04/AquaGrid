"""
tests/test_detection_statistical_audit.py – Day 3: Leak Detection Algorithm Validation
and Statistical Reliability Audit Test Suite.

Covers:
  - Part 2: Normal behavior validation (stable constant telemetry, Gaussian fluctuations, near-threshold readings,
            changing operating conditions, single-sensor & dual-sensor pipes).
  - Part 3: Anomaly detection validation (sustained flow/pressure, coincident surge+drop, 1-spike & 2-spike suppression,
            intermittent chattering suppression, persistent anomalies, gradual onset).
  - Part 4: Statistical reliability (baseline sample thresholds, zero std dev fallback, missing telemetry,
            extreme single outliers, post-resolution fresh window requirements, mixed sensor availability, unknown provenance).
  - Part 5: Explainability (evidence string details, baseline/window sample counts, z-scores, severity rationale).
  - Part 6: Live database integrity safety check.
"""

import os
import sys
import pytest
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detection.engine import LeakDetector, DetectorConfig, SensorAnalysisResult, PipeDiagnostic
from simulation.engine import SensorSimulator


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def detector():
    return LeakDetector()


@pytest.fixture()
def sim():
    return SensorSimulator(seed=42)


@pytest.fixture()
def sample_pipe():
    return {
        "id": 1,
        "name": "MainTrunk-1",
        "from_node_name": "Reservoir-Alpha",
        "to_node_name": "Junction-Beta",
    }


@pytest.fixture()
def sample_flow_sensor():
    return {
        "sensor_id": 101,
        "sensor_name": "FM-Trunk-101",
        "sensor_type": "flow_meter",
        "pipe_id": 1,
        "unit": "L/min",
        "status": "active",
    }


@pytest.fixture()
def sample_pressure_sensor():
    return {
        "sensor_id": 202,
        "sensor_name": "PS-Trunk-202",
        "sensor_type": "pressure_sensor",
        "pipe_id": 1,
        "unit": "psi",
        "status": "active",
    }


# ═════════════════════════════════════════════════════════════════════════════
#  PART 2: NORMAL BEHAVIOUR VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

class TestNormalBehaviourValidation:
    def test_stable_constant_telemetry_produces_zero_leaks(self, detector, sample_flow_sensor, sample_pipe):
        """Constant normal telemetry has zero variance, effective_std fallback prevents division by zero."""
        readings = [
            {"reading_id": i, "sensor_id": 101, "reading_value": 350.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(15)
        ]
        res = detector.analyze_sensor(sample_flow_sensor, readings)
        assert res.classification == "normal"
        assert res.is_abnormal is False
        assert res.baseline_std == 0.0
        assert res.anomaly_score == 0.0

        diag = detector.diagnose_pipe(sample_pipe, [sample_flow_sensor], {101: readings})
        assert diag.status == "normal"
        assert diag.severity == "low"

    def test_normal_telemetry_with_gaussian_fluctuations(self, detector, sim, sample_flow_sensor, sample_pressure_sensor, sample_pipe):
        """Normal telemetry with realistic Gaussian variations (N=30) never flags an anomaly."""
        flow_readings = [
            {"reading_id": i, "sensor_id": 101, "reading_value": sim.generate_reading_for_sensor(sample_flow_sensor).reading_value, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(30)
        ]
        press_readings = [
            {"reading_id": i, "sensor_id": 202, "reading_value": sim.generate_reading_for_sensor(sample_pressure_sensor).reading_value, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(30)
        ]

        res_f = detector.analyze_sensor(sample_flow_sensor, flow_readings)
        res_p = detector.analyze_sensor(sample_pressure_sensor, press_readings)
        assert res_f.is_abnormal is False
        assert res_p.is_abnormal is False

        diag = detector.diagnose_pipe(sample_pipe, [sample_flow_sensor, sample_pressure_sensor], {101: flow_readings, 202: press_readings})
        assert diag.status == "normal"
        assert diag.severity == "low"

    def test_readings_near_configured_thresholds_rejected(self, sample_flow_sensor, sample_pressure_sensor):
        """
        Flow surge of +20% (below +25% threshold) and pressure drop of -15% (above -20% drop threshold)
        are rejected when variance is controlled.
        """
        # Detector configured with ratio-based test thresholds
        cfg = DetectorConfig(flow_z_threshold=100.0, pressure_z_threshold=100.0)
        det = LeakDetector(cfg)

        # Baseline 350 L/min, window 420 L/min (420/350 = 1.20x < 1.25x)
        flow_readings = [
            {"reading_id": i, "sensor_id": 101, "reading_value": 350.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(10)
        ] + [
            {"reading_id": 10 + j, "sensor_id": 101, "reading_value": 420.0, "recorded_at": f"2026-11-10T08:{10+j:02d}:00Z", "scenario": "normal"}
            for j in range(5)
        ]
        res_f = det.analyze_sensor(sample_flow_sensor, flow_readings)
        assert res_f.is_abnormal is False

        # Baseline 52.0 psi, window 44.2 psi (44.2/52.0 = 0.85x > 0.80x)
        press_readings = [
            {"reading_id": i, "sensor_id": 202, "reading_value": 52.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(10)
        ] + [
            {"reading_id": 10 + j, "sensor_id": 202, "reading_value": 44.2, "recorded_at": f"2026-11-10T08:{10+j:02d}:00Z", "scenario": "normal"}
            for j in range(5)
        ]
        res_p = det.analyze_sensor(sample_pressure_sensor, press_readings)
        assert res_p.is_abnormal is False

    def test_changing_normal_operating_conditions(self, detector, sample_flow_sensor):
        """Gradual variation within expected baseline bounds (e.g. 340 to 360 L/min) remains normal."""
        # Simulated diurnal wave within normal limits
        readings = [
            {"reading_id": i, "sensor_id": 101, "reading_value": 350.0 + (i % 5) * 2.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(25)
        ]
        res = detector.analyze_sensor(sample_flow_sensor, readings)
        assert res.classification == "normal"
        assert res.is_abnormal is False

    def test_single_sensor_pipes_normal(self, detector, sample_flow_sensor, sample_pressure_sensor, sample_pipe):
        """Single-monitored pipes (flow-only or pressure-only) operate cleanly in normal conditions."""
        flow_readings = [
            {"reading_id": i, "sensor_id": 101, "reading_value": 350.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(15)
        ]
        diag_f = detector.diagnose_pipe(sample_pipe, [sample_flow_sensor], {101: flow_readings})
        assert diag_f.status == "normal"
        assert diag_f.flow_result is not None
        assert diag_f.pressure_result is None

        press_readings = [
            {"reading_id": i, "sensor_id": 202, "reading_value": 52.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(15)
        ]
        diag_p = detector.diagnose_pipe(sample_pipe, [sample_pressure_sensor], {202: press_readings})
        assert diag_p.status == "normal"
        assert diag_p.flow_result is None
        assert diag_p.pressure_result is not None


# ═════════════════════════════════════════════════════════════════════════════
#  PART 3: ANOMALY DETECTION VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

class TestAnomalyDetectionValidation:
    def test_sustained_high_flow_and_low_pressure(self, detector, sample_flow_sensor, sample_pressure_sensor, sample_pipe):
        """Sustained high flow (+70%) and low pressure (-50%) triggers suspected_leak with critical severity."""
        flow_readings = [
            {"reading_id": i, "sensor_id": 101, "reading_value": 350.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(10)
        ] + [
            {"reading_id": 10 + j, "sensor_id": 101, "reading_value": 595.0, "recorded_at": f"2026-11-10T08:{10+j:02d}:00Z", "scenario": "leak"}
            for j in range(5)
        ]
        press_readings = [
            {"reading_id": i, "sensor_id": 202, "reading_value": 52.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(10)
        ] + [
            {"reading_id": 10 + j, "sensor_id": 202, "reading_value": 26.0, "recorded_at": f"2026-11-10T08:{10+j:02d}:00Z", "scenario": "leak"}
            for j in range(5)
        ]

        diag = detector.diagnose_pipe(sample_pipe, [sample_flow_sensor, sample_pressure_sensor], {101: flow_readings, 202: press_readings})
        assert diag.status == "suspected_leak"
        assert diag.severity == "critical"
        assert "Coincident flow surge and pressure drop" in diag.evidence
        assert diag.observed_flow == 595.0
        assert diag.observed_pressure == 26.0

    def test_short_isolated_single_spike_suppressed(self, detector, sample_flow_sensor):
        """A single 1-reading spike followed by normal readings has consecutive=1 < 3 and is suppressed."""
        readings = [
            {"reading_id": i, "sensor_id": 101, "reading_value": 350.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(14)
        ] + [
            {"reading_id": 14, "sensor_id": 101, "reading_value": 650.0, "recorded_at": "2026-11-10T08:14:00Z", "scenario": "normal"}
        ]
        res = detector.analyze_sensor(sample_flow_sensor, readings)
        assert res.consecutive_abnormal == 1
        assert res.is_abnormal is False
        assert res.classification == "normal"

    def test_short_isolated_two_reading_spike_suppressed(self, detector, sample_flow_sensor):
        """A 2-reading spike has consecutive=2 < 3 and is suppressed."""
        readings = [
            {"reading_id": i, "sensor_id": 101, "reading_value": 350.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(13)
        ] + [
            {"reading_id": 13, "sensor_id": 101, "reading_value": 600.0, "recorded_at": "2026-11-10T08:13:00Z", "scenario": "normal"},
            {"reading_id": 14, "sensor_id": 101, "reading_value": 600.0, "recorded_at": "2026-11-10T08:14:00Z", "scenario": "normal"},
        ]
        res = detector.analyze_sensor(sample_flow_sensor, readings)
        assert res.consecutive_abnormal == 2
        assert res.is_abnormal is False

    def test_intermittent_chattering_anomalies_suppressed(self, detector, sample_flow_sensor):
        """
        Alternating abnormal/normal readings ([600, 350, 600, 350, 600]) breaks the backward scan
        at the first normal sample, giving consecutive=1 < 3.
        """
        readings = [
            {"reading_id": i, "sensor_id": 101, "reading_value": 350.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(10)
        ] + [
            {"reading_id": 10, "sensor_id": 101, "reading_value": 600.0, "recorded_at": "2026-11-10T08:10:00Z", "scenario": "normal"},
            {"reading_id": 11, "sensor_id": 101, "reading_value": 350.0, "recorded_at": "2026-11-10T08:11:00Z", "scenario": "normal"},
            {"reading_id": 12, "sensor_id": 101, "reading_value": 600.0, "recorded_at": "2026-11-10T08:12:00Z", "scenario": "normal"},
            {"reading_id": 13, "sensor_id": 101, "reading_value": 350.0, "recorded_at": "2026-11-10T08:13:00Z", "scenario": "normal"},
            {"reading_id": 14, "sensor_id": 101, "reading_value": 600.0, "recorded_at": "2026-11-10T08:14:00Z", "scenario": "normal"},
        ]
        res = detector.analyze_sensor(sample_flow_sensor, readings)
        assert res.consecutive_abnormal == 1
        assert res.is_abnormal is False

    def test_persistent_anomaly_at_required_threshold(self, detector, sample_flow_sensor):
        """Exactly 3 consecutive abnormal readings ([350, 350, 600, 600, 600]) satisfies consecutive >= 3."""
        readings = [
            {"reading_id": i, "sensor_id": 101, "reading_value": 350.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(10)
        ] + [
            {"reading_id": 10, "sensor_id": 101, "reading_value": 350.0, "recorded_at": "2026-11-10T08:10:00Z", "scenario": "normal"},
            {"reading_id": 11, "sensor_id": 101, "reading_value": 350.0, "recorded_at": "2026-11-10T08:11:00Z", "scenario": "normal"},
            {"reading_id": 12, "sensor_id": 101, "reading_value": 600.0, "recorded_at": "2026-11-10T08:12:00Z", "scenario": "leak"},
            {"reading_id": 13, "sensor_id": 101, "reading_value": 600.0, "recorded_at": "2026-11-10T08:13:00Z", "scenario": "leak"},
            {"reading_id": 14, "sensor_id": 101, "reading_value": 600.0, "recorded_at": "2026-11-10T08:14:00Z", "scenario": "leak"},
        ]
        res = detector.analyze_sensor(sample_flow_sensor, readings)
        assert res.consecutive_abnormal == 3
        assert res.is_abnormal is True
        assert res.classification == "elevated_flow"

    def test_gradual_flow_and_pressure_anomaly_onset(self, detector, sim, sample_flow_sensor, sample_pressure_sensor):
        """Gradual onset reaching threshold for 3 consecutive readings is properly flagged."""
        # 15 baseline readings with natural noise
        base_f = [
            {"reading_id": i, "sensor_id": 101, "reading_value": sim.generate_reading_for_sensor(sample_flow_sensor).reading_value, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(15)
        ]
        # Window: 2 normal readings, followed by 3 progressive surge readings
        win_f = [
            {"reading_id": 15, "sensor_id": 101, "reading_value": 350.0, "recorded_at": "2026-11-10T08:15:00Z", "scenario": "normal"},
            {"reading_id": 16, "sensor_id": 101, "reading_value": 355.0, "recorded_at": "2026-11-10T08:16:00Z", "scenario": "normal"},
            {"reading_id": 17, "sensor_id": 101, "reading_value": 460.0, "recorded_at": "2026-11-10T08:17:00Z", "scenario": "leak"},
            {"reading_id": 18, "sensor_id": 101, "reading_value": 520.0, "recorded_at": "2026-11-10T08:18:00Z", "scenario": "leak"},
            {"reading_id": 19, "sensor_id": 101, "reading_value": 600.0, "recorded_at": "2026-11-10T08:19:00Z", "scenario": "leak"},
        ]
        res_f = detector.analyze_sensor(sample_flow_sensor, base_f + win_f)
        assert res_f.consecutive_abnormal == 3
        assert res_f.is_abnormal is True

        # Gradual pressure drop
        base_p = [
            {"reading_id": i, "sensor_id": 202, "reading_value": sim.generate_reading_for_sensor(sample_pressure_sensor).reading_value, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(15)
        ]
        win_p = [
            {"reading_id": 15, "sensor_id": 202, "reading_value": 52.0, "recorded_at": "2026-11-10T08:15:00Z", "scenario": "normal"},
            {"reading_id": 16, "sensor_id": 202, "reading_value": 51.5, "recorded_at": "2026-11-10T08:16:00Z", "scenario": "normal"},
            {"reading_id": 17, "sensor_id": 202, "reading_value": 38.0, "recorded_at": "2026-11-10T08:17:00Z", "scenario": "leak"},
            {"reading_id": 18, "sensor_id": 202, "reading_value": 30.0, "recorded_at": "2026-11-10T08:18:00Z", "scenario": "leak"},
            {"reading_id": 19, "sensor_id": 202, "reading_value": 24.0, "recorded_at": "2026-11-10T08:19:00Z", "scenario": "leak"},
        ]
        res_p = detector.analyze_sensor(sample_pressure_sensor, base_p + win_p)
        assert res_p.consecutive_abnormal == 3
        assert res_p.is_abnormal is True


# ═════════════════════════════════════════════════════════════════════════════
#  PART 4: STATISTICAL RELIABILITY
# ═════════════════════════════════════════════════════════════════════════════

class TestStatisticalReliability:
    def test_baseline_sample_requirements(self, detector, sample_flow_sensor):
        """Fewer than 5 verified normal samples in history returns insufficient_data."""
        # 8 total readings: window takes 5, history only has 3 (< 5 minimum)
        readings = [
            {"reading_id": i, "sensor_id": 101, "reading_value": 350.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(8)
        ]
        res = detector.analyze_sensor(sample_flow_sensor, readings)
        assert res.classification == "insufficient_data"
        assert res.baseline_samples == 3
        assert "Insufficient verified normal baseline samples" in res.evidence

    def test_zero_or_near_zero_standard_deviation_fallback(self, detector, sample_flow_sensor):
        """Constant readings produce baseline_std == 0.0; effective_std fallback prevents ZeroDivisionError."""
        readings = [
            {"reading_id": i, "sensor_id": 101, "reading_value": 350.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(15)
        ]
        res = detector.analyze_sensor(sample_flow_sensor, readings)
        assert res.baseline_std == 0.0
        assert res.anomaly_score == 0.0
        assert res.deviation_ratio == 1.0

    def test_missing_readings_handled_safely(self, detector, sample_flow_sensor):
        """An empty readings list returns insufficient_data with 0 sample_count."""
        res = detector.analyze_sensor(sample_flow_sensor, [])
        assert res.classification == "insufficient_data"
        assert res.sample_count == 0
        assert "No telemetry readings recorded" in res.evidence

    def test_extreme_single_outlier_suppression(self, detector, sample_flow_sensor):
        """A single extreme reading (e.g. 50,000 L/min) is suppressed by consecutive < 3 requirement."""
        readings = [
            {"reading_id": i, "sensor_id": 101, "reading_value": 350.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(14)
        ] + [
            {"reading_id": 14, "sensor_id": 101, "reading_value": 50000.0, "recorded_at": "2026-11-10T08:14:00Z", "scenario": "normal"}
        ]
        res = detector.analyze_sensor(sample_flow_sensor, readings)
        assert res.consecutive_abnormal == 1
        assert res.is_abnormal is False
        assert res.classification == "normal"

    def test_mixed_sensor_availability_on_dual_pipe_reports_insufficient_data(self, detector, sample_flow_sensor, sample_pressure_sensor, sample_pipe):
        """If one sensor on a dual pipe has insufficient data, the pipe diagnostic reports insufficient_data (never falsely normal)."""
        # Flow sensor has 20 valid readings
        flow_readings = [
            {"reading_id": i, "sensor_id": 101, "reading_value": 350.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(20)
        ]
        # Pressure sensor only has 7 readings (window takes 5, history has 2 < 5)
        press_readings = [
            {"reading_id": i, "sensor_id": 202, "reading_value": 52.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(7)
        ]

        diag = detector.diagnose_pipe(sample_pipe, [sample_flow_sensor, sample_pressure_sensor], {101: flow_readings, 202: press_readings})
        assert diag.status == "insufficient_data"
        assert "Pending baseline verification" in diag.evidence
        assert diag.severity == "low"

    def test_unknown_scenario_provenance_excluded_from_baseline(self, detector, sample_flow_sensor):
        """Readings with scenario=None (legacy/unverified) cannot serve as baseline samples."""
        readings = [
            {"reading_id": i, "sensor_id": 101, "reading_value": 350.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": None}
            for i in range(15)
        ]
        res = detector.analyze_sensor(sample_flow_sensor, readings)
        assert res.classification == "insufficient_data"
        assert res.baseline_samples == 0
        assert res.unknown_provenance_samples == 10
        assert "unverified/unknown scenario provenance" in res.evidence

    def test_post_resolution_freshness_cutoff_respected(self, detector, sample_flow_sensor):
        """Readings before or at resolved_cutoff cannot enter the post-resolution observation window."""
        res_time = "2026-11-10T08:10:00Z"

        # 10 readings before resolution, 3 readings after resolution (< 5 required)
        readings = [
            {"reading_id": i, "sensor_id": 101, "reading_value": 350.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(10)
        ] + [
            {"reading_id": 11, "sensor_id": 101, "reading_value": 600.0, "recorded_at": "2026-11-10T08:11:00Z", "scenario": "leak"},
            {"reading_id": 12, "sensor_id": 101, "reading_value": 600.0, "recorded_at": "2026-11-10T08:12:00Z", "scenario": "leak"},
            {"reading_id": 13, "sensor_id": 101, "reading_value": 600.0, "recorded_at": "2026-11-10T08:13:00Z", "scenario": "leak"},
        ]

        res = detector.analyze_sensor(sample_flow_sensor, readings, resolved_cutoff=res_time)
        assert res.classification == "insufficient_data"
        assert "Insufficient fresh post-resolution telemetry" in res.evidence
        assert res.window_samples == 3


# ═════════════════════════════════════════════════════════════════════════════
#  PART 5: EXPLAINABILITY VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

class TestExplainabilityValidation:
    def test_diagnostic_evidence_completeness(self, detector, sample_flow_sensor, sample_pressure_sensor, sample_pipe):
        """Verify that PipeDiagnostic contains complete, transparent evidence and metrics."""
        flow_readings = [
            {"reading_id": i, "sensor_id": 101, "reading_value": 350.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(10)
        ] + [
            {"reading_id": 10 + j, "sensor_id": 101, "reading_value": 595.0, "recorded_at": f"2026-11-10T08:{10+j:02d}:00Z", "scenario": "leak"}
            for j in range(5)
        ]
        press_readings = [
            {"reading_id": i, "sensor_id": 202, "reading_value": 52.0, "recorded_at": f"2026-11-10T08:{i:02d}:00Z", "scenario": "normal"}
            for i in range(10)
        ] + [
            {"reading_id": 10 + j, "sensor_id": 202, "reading_value": 26.0, "recorded_at": f"2026-11-10T08:{10+j:02d}:00Z", "scenario": "leak"}
            for j in range(5)
        ]

        diag = detector.diagnose_pipe(sample_pipe, [sample_flow_sensor, sample_pressure_sensor], {101: flow_readings, 202: press_readings})

        # Transparent metrics
        assert diag.observed_flow == 595.0
        assert diag.baseline_flow == 350.0
        assert diag.baseline_flow_samples == 10
        assert diag.window_flow_samples == 5
        assert diag.flow_unit == "L/min"
        assert diag.flow_score is not None

        assert diag.observed_pressure == 26.0
        assert diag.baseline_pressure == 52.0
        assert diag.baseline_pressure_samples == 10
        assert diag.window_pressure_samples == 5
        assert diag.pressure_unit == "psi"
        assert diag.pressure_score is not None

        # Human-readable evidence explanation
        assert "Coincident flow surge and pressure drop" in diag.evidence
        assert "Flow surge detected" in diag.evidence
        assert "Pressure drop detected" in diag.evidence
        assert "+70.0% above verified baseline" in diag.evidence
        assert "-50.0% below verified baseline" in diag.evidence


# ═════════════════════════════════════════════════════════════════════════════
#  PART 6: LIVE DATABASE INTEGRITY SAFETY CHECK
# ═════════════════════════════════════════════════════════════════════════════

def test_live_database_remains_unmodified():
    """Verify live database instance/aquagrid.db records were untouched during the audit."""
    import sqlite3
    live_db_path = os.path.join(PROJECT_ROOT, "instance", "aquagrid.db")
    assert os.path.isfile(live_db_path), f"Live DB not found at {live_db_path}"

    conn = sqlite3.connect(live_db_path)
    conn.row_factory = sqlite3.Row
    try:
        events = conn.execute("SELECT * FROM leak_events ORDER BY event_id ASC").fetchall()
        assert len(events) == 2
        assert events[0]["event_id"] == 1
        assert events[0]["status"] == "resolved"
        assert events[1]["event_id"] == 2
        assert events[1]["status"] == "open"

        tasks = conn.execute("SELECT * FROM repair_tasks ORDER BY task_id ASC").fetchall()
        assert len(tasks) == 2
        assert tasks[0]["task_id"] == 1
        assert tasks[0]["status"] == "Completed"
        assert tasks[1]["task_id"] == 2
        assert tasks[1]["status"] == "Pending"

        audits = conn.execute("SELECT * FROM lifecycle_audit_log").fetchall()
        assert len(audits) == 4
    finally:
        conn.close()

