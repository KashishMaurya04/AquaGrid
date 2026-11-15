"""
tests/test_detection.py – Unit and integration test suite for the Leak Detection Engine milestone.

Covers:
  1. Separation of observation window from baseline history.
  2. Simulation scenario provenance (normal vs leak vs unknown).
  3. Leak readings do not contaminate normal baselines.
  4. Normal readings are not incorrectly flagged.
  5. Persistent high-flow and low-pressure anomalies can be detected.
  6. Insufficient historical normal data produces an insufficient-data result.
  7. Unknown-provenance historical records are handled safely (excluded from baseline).
  8. Multi-sensor evidence correlation and single-sensor anomaly detection.
  9. False-positive suppression (transient single spikes suppressed).
 10. Independent measurement unit isolation (L/min, m³/h, psi, bar handled independently).
 11. Inactive sensors strictly excluded from detection.
 12. Database persistence in leak_events and is_simulated == 1 verification.
 13. Duplicate event prevention (updating open/investigating events instead of creating duplicates).
 14. Event lifecycle status updates ('open' -> 'investigating' -> 'resolved') and resolved_at timestamp.
 15. Network-wide detection execution (run_detection_network).
 16. Flask route integration:
     - GET /leak-detection loads cleanly with disclaimer and summary metrics (HTTP 200)
     - POST /leak-detection/run executes network detection and redirects (HTTP 302)
     - POST /leak-detection/events/<id>/status updates status and redirects (HTTP 302)
     - GET on POST-only endpoints returns 405 Method Not Allowed
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detection.engine import (
    LeakDetector,
    DetectorConfig,
    SensorAnalysisResult,
    PipeDiagnostic,
    DEFAULT_DETECTOR_CONFIG,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    """Isolate database tests with a fresh SQLite DB in a temporary directory."""
    import database as db_module

    test_db = str(tmp_path / "test_detection_aquagrid.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


@pytest.fixture()
def detection_network(tmp_db):
    """
    Sets up a 4-node, 3-pipe network with diverse sensor types and units:
      - Pipe 1: Tank Alpha -> Junction Beta (Dual monitored: Flow L/min + Pressure psi)
      - Pipe 2: Junction Beta -> Valve Gamma (Single monitored: Flow m³/h)
      - Pipe 3: Valve Gamma -> Tank Delta (Single monitored: Pressure bar + inactive Flow)
    """
    id_a = tmp_db.create_node("Tank Alpha", "tank", "Zone 1", "active")
    id_b = tmp_db.create_node("Junction Beta", "junction", "Zone 2", "active")
    id_c = tmp_db.create_node("Valve Gamma", "valve", "Zone 3", "active")
    id_d = tmp_db.create_node("Tank Delta", "tank", "Zone 4", "active")

    pipe_1 = tmp_db.create_pipe("Trunk-Main-1", id_a, id_b, "200.0", "500.0", "1.0", "active")
    pipe_2 = tmp_db.create_pipe("Branch-2", id_b, id_c, "150.0", "350.0", "1.5", "active")
    pipe_3 = tmp_db.create_pipe("Sub-Branch-3", id_c, id_d, "100.0", "200.0", "2.0", "active")

    # Sensors on Pipe 1 (Dual: Flow L/min + Pressure psi)
    s1_flow = tmp_db.create_sensor("FM-Trunk-1", pipe_1, "flow_meter", "L/min", "active")
    s1_press = tmp_db.create_sensor("PS-Trunk-1", pipe_1, "pressure_sensor", "psi", "active")

    # Sensors on Pipe 2 (Single: Flow m³/h)
    s2_flow = tmp_db.create_sensor("FM-Branch-2", pipe_2, "flow_meter", "m³/h", "active")

    # Sensors on Pipe 3 (Pressure bar + Inactive Flow L/min)
    s3_press = tmp_db.create_sensor("PS-SubBranch-3", pipe_3, "pressure_sensor", "bar", "active")
    s3_flow_inactive = tmp_db.create_sensor("FM-SubBranch-3-Inactive", pipe_3, "flow_meter", "L/min", "inactive")

    return {
        "nodes": [id_a, id_b, id_c, id_d],
        "pipe_1": pipe_1,
        "pipe_2": pipe_2,
        "pipe_3": pipe_3,
        "sensors": {
            "s1_flow": s1_flow,
            "s1_press": s1_press,
            "s2_flow": s2_flow,
            "s3_press": s3_press,
            "s3_flow_inactive": s3_flow_inactive,
        },
    }


@pytest.fixture()
def app_client(tmp_db, monkeypatch):
    """Flask test client wired to the isolated database."""
    import database as db_module
    import app as app_module

    monkeypatch.setattr(app_module, "db", db_module)

    app_module.app.config["TESTING"] = True
    app_module.app.config["WTF_CSRF_ENABLED"] = False
    app_module.app.config["SECRET_KEY"] = "test-detection-secret-key"

    with app_module.app.test_client() as client:
        yield client


# Helper to record a sequence of readings for a sensor
def record_readings(
    tmp_db,
    sensor_id: int,
    values: list[float],
    unit: str = "L/min",
    is_simulated: bool = True,
    scenario: str = "normal",
):
    for i, val in enumerate(values):
        tmp_db.record_sensor_reading(
            sensor_id=sensor_id,
            reading_value=val,
            reading_type=unit,
            is_simulated=is_simulated,
            scenario=scenario,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  1. DETECTOR CONFIGURATION UNIT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestDetectorConfig:
    def test_default_config_values(self):
        config = DEFAULT_DETECTOR_CONFIG
        assert config.window_size == 5
        assert config.min_baseline_samples == 5
        assert config.consecutive_abnormal_required == 3
        assert config.flow_ratio_threshold == 1.25
        assert config.flow_z_threshold == 2.5
        assert config.pressure_ratio_threshold == 0.80
        assert config.pressure_z_threshold == 2.0
        assert config.auto_record_events is True

    def test_custom_config_values(self):
        config = DetectorConfig(
            window_size=8,
            min_baseline_samples=10,
            consecutive_abnormal_required=4,
            flow_ratio_threshold=1.30,
            flow_z_threshold=3.0,
            pressure_ratio_threshold=0.75,
            pressure_z_threshold=2.5,
            auto_record_events=False,
        )
        assert config.window_size == 8
        assert config.min_baseline_samples == 10
        assert config.consecutive_abnormal_required == 4
        assert config.flow_ratio_threshold == 1.30
        assert config.flow_z_threshold == 3.0
        assert config.pressure_ratio_threshold == 0.75
        assert config.pressure_z_threshold == 2.5
        assert config.auto_record_events is False


# ═════════════════════════════════════════════════════════════════════════════
#  2. SENSOR-LEVEL SLIDING-WINDOW & BASELINE VALIDATION TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestSensorAnalysis:
    def test_normal_flow_readings_not_flagged(self):
        """10 normal flow readings (5 baseline history + 5 observation window) not flagged."""
        detector = LeakDetector()
        sensor = {
            "sensor_id": 1,
            "sensor_name": "FM-01",
            "sensor_type": "flow_meter",
            "pipe_id": 10,
            "unit": "L/min",
            "status": "active",
        }
        # 5 normal history readings + 5 normal window readings around 50 L/min
        history = [
            {"reading_id": i, "reading_value": 50.0 + (i % 3) * 0.5, "recorded_at": f"2026-11-01T10:0{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        window = [
            {"reading_id": 5 + i, "reading_value": 50.2 + (i % 2) * 0.3, "recorded_at": f"2026-11-01T10:1{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        result = detector.analyze_sensor(sensor, history + window)

        assert result.classification == "normal"
        assert result.is_abnormal is False
        assert result.baseline_samples == 5
        assert result.window_samples == 5
        assert result.baseline_mean is not None
        assert abs(result.baseline_mean - 50.5) < 1.0
        assert "within normal bounds" in result.evidence.lower()

    def test_normal_pressure_readings_not_flagged(self):
        """10 normal pressure readings (5 baseline history + 5 observation window) not flagged."""
        detector = LeakDetector()
        sensor = {
            "sensor_id": 2,
            "sensor_name": "PS-01",
            "sensor_type": "pressure_sensor",
            "pipe_id": 10,
            "unit": "psi",
            "status": "active",
        }
        history = [
            {"reading_id": i, "reading_value": 60.0 + (i % 2) * 0.4, "recorded_at": f"2026-11-01T10:0{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        window = [
            {"reading_id": 5 + i, "reading_value": 60.1 + (i % 2) * 0.3, "recorded_at": f"2026-11-01T10:1{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        result = detector.analyze_sensor(sensor, history + window)

        assert result.classification == "normal"
        assert result.is_abnormal is False
        assert result.baseline_samples == 5
        assert result.window_samples == 5
        assert "within normal bounds" in result.evidence.lower()

    def test_high_flow_surge_detected(self):
        """5 normal history readings + 5 surge window readings detected cleanly."""
        detector = LeakDetector()
        sensor = {
            "sensor_id": 1,
            "sensor_name": "FM-01",
            "sensor_type": "flow_meter",
            "pipe_id": 10,
            "unit": "L/min",
            "status": "active",
        }
        history = [
            {"reading_id": i, "reading_value": val, "recorded_at": f"2026-11-01T10:0{i}:00Z", "scenario": "normal"}
            for i, val in enumerate([50.0, 50.2, 49.8, 50.1, 49.9])
        ]
        window = [
            {"reading_id": 5 + i, "reading_value": val, "recorded_at": f"2026-11-01T10:1{i}:00Z", "scenario": "leak"}
            for i, val in enumerate([80.0, 81.0, 82.0, 81.5, 83.0])
        ]
        result = detector.analyze_sensor(sensor, history + window)

        assert result.classification == "elevated_flow"
        assert result.is_abnormal is True
        assert result.baseline_samples == 5
        assert result.window_samples == 5
        assert result.consecutive_abnormal >= 3
        assert result.anomaly_score is not None
        assert result.anomaly_score > 2.0
        assert "flow surge detected" in result.evidence.lower()
        assert result.unit == "L/min"

    def test_low_pressure_drop_detected(self):
        """5 normal history readings + 5 drop window readings detected cleanly."""
        detector = LeakDetector()
        sensor = {
            "sensor_id": 2,
            "sensor_name": "PS-01",
            "sensor_type": "pressure_sensor",
            "pipe_id": 10,
            "unit": "psi",
            "status": "active",
        }
        history = [
            {"reading_id": i, "reading_value": val, "recorded_at": f"2026-11-01T10:0{i}:00Z", "scenario": "normal"}
            for i, val in enumerate([60.0, 60.5, 59.5, 60.2, 59.8])
        ]
        window = [
            {"reading_id": 5 + i, "reading_value": val, "recorded_at": f"2026-11-01T10:1{i}:00Z", "scenario": "leak"}
            for i, val in enumerate([35.0, 34.0, 35.5, 33.5, 34.2])
        ]
        result = detector.analyze_sensor(sensor, history + window)

        assert result.classification == "reduced_pressure"
        assert result.is_abnormal is True
        assert result.baseline_samples == 5
        assert result.window_samples == 5
        assert result.consecutive_abnormal >= 3
        assert result.anomaly_score is not None
        assert result.anomaly_score < -2.0
        assert "pressure drop detected" in result.evidence.lower()
        assert result.unit == "psi"

    def test_leak_readings_do_not_contaminate_normal_baselines(self):
        """
        Verify that leak-scenario readings in history are strictly excluded from
        normal baseline calculations, preventing baseline inflation/deflation.
        """
        detector = LeakDetector()
        sensor = {
            "sensor_id": 1,
            "sensor_name": "FM-01",
            "sensor_type": "flow_meter",
            "pipe_id": 10,
            "unit": "L/min",
            "status": "active",
        }
        # 10 verified normal readings (~350 L/min)
        normal_history = [
            {"reading_id": i, "reading_value": 350.0 + (i % 3) * 0.5, "recorded_at": f"2026-11-01T10:{i:02d}:00Z", "scenario": "normal"}
            for i in range(10)
        ]
        # 5 leak scenario readings (~600 L/min) recorded during earlier leak simulation
        leak_history = [
            {"reading_id": 10 + i, "reading_value": 600.0 + (i % 2) * 2.0, "recorded_at": f"2026-11-01T10:{10+i:02d}:00Z", "scenario": "leak"}
            for i in range(5)
        ]
        # 5 current observation window readings (~351 L/min) after leak was repaired
        current_window = [
            {"reading_id": 15 + i, "reading_value": 351.0 + (i % 2) * 0.4, "recorded_at": f"2026-11-01T10:{15+i:02d}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        all_readings = normal_history + leak_history + current_window

        result = detector.analyze_sensor(sensor, all_readings)

        # Baseline MUST be ~350.5 L/min (from the 10 normal readings only), NOT contaminated to ~503 L/min!
        assert result.baseline_samples == 10
        assert result.leak_scenario_samples == 5
        assert result.window_samples == 5
        assert abs(result.baseline_mean - 350.5) < 1.5
        assert abs(result.window_mean - 351.2) < 1.0
        assert result.is_abnormal is False
        assert result.classification == "normal"

    def test_unknown_provenance_historical_records_handled_safely(self):
        """
        Verify that historical records without known scenario provenance (scenario is NULL/None)
        are handled transparently and NOT silently assumed to be normal.
        """
        detector = LeakDetector()
        sensor = {
            "sensor_id": 1,
            "sensor_name": "FM-Legacy",
            "sensor_type": "flow_meter",
            "pipe_id": 10,
            "unit": "L/min",
            "status": "active",
        }
        # 42 legacy readings with scenario=None (like the historical records in aquagrid.db)
        legacy_readings = [
            {"reading_id": i, "reading_value": 350.0 + (i % 3) * 0.5, "recorded_at": f"2026-11-01T09:{i:02d}:00Z", "scenario": None}
            for i in range(42)
        ]
        result = detector.analyze_sensor(sensor, legacy_readings)

        assert result.classification == "insufficient_data"
        assert result.is_abnormal is False
        assert result.baseline_samples == 0
        assert result.unknown_provenance_samples == 42 - detector.config.window_size
        assert "unverified/unknown scenario provenance" in result.evidence

    def test_insufficient_historical_normal_data_produces_insufficient_data(self):
        """
        When fewer than min_baseline_samples verified normal readings exist in history,
        the engine must produce an insufficient_data result.
        """
        detector = LeakDetector()
        sensor = {
            "sensor_id": 1,
            "sensor_name": "FM-01",
            "sensor_type": "flow_meter",
            "pipe_id": 10,
            "unit": "L/min",
            "status": "active",
        }
        # Only 3 normal readings in history + 5 window readings (3 < 5 required)
        history = [
            {"reading_id": i, "reading_value": 50.0, "recorded_at": f"2026-11-01T10:0{i}:00Z", "scenario": "normal"}
            for i in range(3)
        ]
        window = [
            {"reading_id": 10 + i, "reading_value": 50.0, "recorded_at": f"2026-11-01T10:1{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        result = detector.analyze_sensor(sensor, history + window)

        assert result.classification == "insufficient_data"
        assert result.baseline_samples == 3
        assert result.baseline_mean is None
        assert "insufficient verified normal baseline samples" in result.evidence.lower()

    def test_consecutive_abnormal_requirement_suppresses_transient_spike(self):
        """
        Verify false-positive suppression: A single transient spike followed by normal readings
        or an isolated 1-spike anomaly does NOT trigger is_abnormal=True because
        consecutive_abnormal_required is 3.
        """
        detector = LeakDetector()
        sensor = {
            "sensor_id": 1,
            "sensor_name": "FM-01",
            "sensor_type": "flow_meter",
            "pipe_id": 10,
            "unit": "L/min",
            "status": "active",
        }
        # 5 normal history readings
        history = [
            {"reading_id": i, "reading_value": 50.0, "recorded_at": f"2026-11-01T10:0{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        # Window with 1 spike of 90 L/min, but followed by 4 normal readings of 50 L/min
        window = [
            {"reading_id": 10, "reading_value": 90.0, "recorded_at": "2026-11-01T10:10:00Z", "scenario": "normal"},
            {"reading_id": 11, "reading_value": 50.0, "recorded_at": "2026-11-01T10:11:00Z", "scenario": "normal"},
            {"reading_id": 12, "reading_value": 50.1, "recorded_at": "2026-11-01T10:12:00Z", "scenario": "normal"},
            {"reading_id": 13, "reading_value": 49.9, "recorded_at": "2026-11-01T10:13:00Z", "scenario": "normal"},
            {"reading_id": 14, "reading_value": 50.0, "recorded_at": "2026-11-01T10:14:00Z", "scenario": "normal"},
        ]
        result = detector.analyze_sensor(sensor, history + window)

        assert result.is_abnormal is False
        assert result.classification == "normal"
        assert result.consecutive_abnormal < 3

    def test_inactive_sensor_excluded(self):
        """Inactive sensors return classification='inactive' and is_abnormal=False."""
        detector = LeakDetector()
        sensor = {
            "sensor_id": 3,
            "sensor_name": "FM-Inactive",
            "sensor_type": "flow_meter",
            "pipe_id": 10,
            "unit": "L/min",
            "status": "inactive",
        }
        readings = [
            {"reading_id": i, "reading_value": 150.0, "recorded_at": f"2026-11-01T10:{i:02d}:00Z", "scenario": "normal"}
            for i in range(10)
        ]
        result = detector.analyze_sensor(sensor, readings)

        assert result.classification == "inactive"
        assert result.is_abnormal is False
        assert "inactive" in result.evidence.lower()

    def test_different_measurement_units_handled_independently(self):
        """Verify sensors with different units (m³/h, bar) maintain their native unit and scales."""
        detector = LeakDetector()

        # Flow in m³/h
        sensor_m3h = {
            "sensor_id": 4,
            "sensor_name": "FM-m3h",
            "sensor_type": "flow_meter",
            "pipe_id": 12,
            "unit": "m³/h",
            "status": "active",
        }
        history_m3h = [
            {"reading_id": i, "reading_value": 12.0 + (i % 2) * 0.1, "recorded_at": f"2026-11-01T10:0{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        window_m3h = [
            {"reading_id": 5 + i, "reading_value": 18.0 + (i % 2) * 0.5, "recorded_at": f"2026-11-01T10:1{i}:00Z", "scenario": "leak"}
            for i in range(5)
        ]
        res_m3h = detector.analyze_sensor(sensor_m3h, history_m3h + window_m3h)
        assert res_m3h.unit == "m³/h"
        assert res_m3h.is_abnormal is True
        assert "m³/h" in res_m3h.evidence

        # Pressure in bar
        sensor_bar = {
            "sensor_id": 5,
            "sensor_name": "PS-bar",
            "sensor_type": "pressure_sensor",
            "pipe_id": 12,
            "unit": "bar",
            "status": "active",
        }
        history_bar = [
            {"reading_id": i, "reading_value": 4.2 - (i % 2) * 0.1, "recorded_at": f"2026-11-01T10:0{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        window_bar = [
            {"reading_id": 5 + i, "reading_value": 2.1 - (i % 2) * 0.1, "recorded_at": f"2026-11-01T10:1{i}:00Z", "scenario": "leak"}
            for i in range(5)
        ]
        res_bar = detector.analyze_sensor(sensor_bar, history_bar + window_bar)
        assert res_bar.unit == "bar"
        assert res_bar.is_abnormal is True
        assert "bar" in res_bar.evidence


# ═════════════════════════════════════════════════════════════════════════════
#  3. PIPE-LEVEL CORRELATION & MULTI-SENSOR DIAGNOSTIC TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestPipeDiagnostics:
    def test_coincident_surge_and_drop_triggers_high_severity_leak(self):
        """
        When both flow surge and pressure drop occur on the same pipe,
        the diagnostic classifies the pipe as suspected_leak with high/critical severity.
        """
        detector = LeakDetector()
        pipe = {"id": 1, "name": "Trunk-Main-1", "from_node_name": "Alpha", "to_node_name": "Beta"}
        sensors = [
            {"sensor_id": 1, "sensor_name": "FM-1", "sensor_type": "flow_meter", "pipe_id": 1, "unit": "L/min", "status": "active"},
            {"sensor_id": 2, "sensor_name": "PS-1", "sensor_type": "pressure_sensor", "pipe_id": 1, "unit": "psi", "status": "active"},
        ]
        flow_history = [
            {"reading_id": i, "reading_value": 50.0 + (i % 2) * 0.2, "recorded_at": f"2026-11-01T10:0{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        flow_window = [
            {"reading_id": 5 + i, "reading_value": 85.0 + (i % 2) * 1.0, "recorded_at": f"2026-11-01T10:1{i}:00Z", "scenario": "leak"}
            for i in range(5)
        ]
        press_history = [
            {"reading_id": i, "reading_value": 60.0 - (i % 2) * 0.2, "recorded_at": f"2026-11-01T10:0{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        press_window = [
            {"reading_id": 5 + i, "reading_value": 31.0 - (i % 2) * 0.5, "recorded_at": f"2026-11-01T10:1{i}:00Z", "scenario": "leak"}
            for i in range(5)
        ]
        readings = {
            1: flow_history + flow_window,
            2: press_history + press_window,
        }

        diag = detector.diagnose_pipe(pipe, sensors, readings)

        assert diag.status == "suspected_leak"
        assert diag.severity in ("high", "critical")
        assert "coincident flow surge and pressure drop" in diag.evidence.lower()
        assert diag.observed_flow is not None
        assert diag.observed_pressure is not None
        assert diag.flow_score > 0
        assert diag.pressure_score < 0
        assert diag.baseline_flow_samples == 5
        assert diag.window_flow_samples == 5
        assert diag.baseline_pressure_samples == 5
        assert diag.window_pressure_samples == 5

    def test_both_sensors_normal_produces_normal_status(self):
        """When flow and pressure are normal, pipe status is normal."""
        detector = LeakDetector()
        pipe = {"id": 1, "name": "Trunk-Main-1", "from_node_name": "Alpha", "to_node_name": "Beta"}
        sensors = [
            {"sensor_id": 1, "sensor_name": "FM-1", "sensor_type": "flow_meter", "pipe_id": 1, "unit": "L/min", "status": "active"},
            {"sensor_id": 2, "sensor_name": "PS-1", "sensor_type": "pressure_sensor", "pipe_id": 1, "unit": "psi", "status": "active"},
        ]
        readings = {
            1: [{"reading_id": i, "reading_value": 50.0 + (i % 2) * 0.4, "recorded_at": f"2026-11-01T10:{i:02d}:00Z", "scenario": "normal"}
                for i in range(10)],
            2: [{"reading_id": i, "reading_value": 60.0 - (i % 2) * 0.3, "recorded_at": f"2026-11-01T10:{i:02d}:00Z", "scenario": "normal"}
                for i in range(10)],
        }

        diag = detector.diagnose_pipe(pipe, sensors, readings)

        assert diag.status == "normal"
        assert diag.severity == "low"
        assert "operating within normal" in diag.evidence.lower()

    def test_isolated_flow_surge_without_pressure_drop_suppressed(self):
        """
        Multi-sensor suppression: If flow surges but the companion pressure sensor
        remains normal, do NOT flag as a leak (unconfirmed operational variation).
        """
        detector = LeakDetector()
        pipe = {"id": 1, "name": "Trunk-Main-1", "from_node_name": "Alpha", "to_node_name": "Beta"}
        sensors = [
            {"sensor_id": 1, "sensor_name": "FM-1", "sensor_type": "flow_meter", "pipe_id": 1, "unit": "L/min", "status": "active"},
            {"sensor_id": 2, "sensor_name": "PS-1", "sensor_type": "pressure_sensor", "pipe_id": 1, "unit": "psi", "status": "active"},
        ]
        flow_history = [
            {"reading_id": i, "reading_value": 50.0, "recorded_at": f"2026-11-01T10:0{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        flow_window = [
            {"reading_id": 5 + i, "reading_value": 85.0, "recorded_at": f"2026-11-01T10:1{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        press_history = [
            {"reading_id": i, "reading_value": 60.0, "recorded_at": f"2026-11-01T10:0{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        press_window = [
            {"reading_id": 5 + i, "reading_value": 60.1, "recorded_at": f"2026-11-01T10:1{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        readings = {
            1: flow_history + flow_window,
            2: press_history + press_window,
        }

        diag = detector.diagnose_pipe(pipe, sensors, readings)

        assert diag.status == "normal"
        assert "isolated flow surge" in diag.evidence.lower()
        assert "unconfirmed operational variation" in diag.evidence.lower()

    def test_isolated_pressure_drop_without_flow_surge_suppressed(self):
        """
        Multi-sensor suppression: If pressure drops but the companion flow meter
        remains normal, do NOT flag as a leak (unconfirmed operational variation).
        """
        detector = LeakDetector()
        pipe = {"id": 1, "name": "Trunk-Main-1", "from_node_name": "Alpha", "to_node_name": "Beta"}
        sensors = [
            {"sensor_id": 1, "sensor_name": "FM-1", "sensor_type": "flow_meter", "pipe_id": 1, "unit": "L/min", "status": "active"},
            {"sensor_id": 2, "sensor_name": "PS-1", "sensor_type": "pressure_sensor", "pipe_id": 1, "unit": "psi", "status": "active"},
        ]
        flow_history = [
            {"reading_id": i, "reading_value": 50.0, "recorded_at": f"2026-11-01T10:0{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        flow_window = [
            {"reading_id": 5 + i, "reading_value": 50.1, "recorded_at": f"2026-11-01T10:1{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        press_history = [
            {"reading_id": i, "reading_value": 60.0, "recorded_at": f"2026-11-01T10:0{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        press_window = [
            {"reading_id": 5 + i, "reading_value": 31.0, "recorded_at": f"2026-11-01T10:1{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        readings = {
            1: flow_history + flow_window,
            2: press_history + press_window,
        }

        diag = detector.diagnose_pipe(pipe, sensors, readings)

        assert diag.status == "normal"
        assert "isolated pressure drop" in diag.evidence.lower()
        assert "unconfirmed operational variation" in diag.evidence.lower()

    def test_single_sensor_pipe_with_flow_surge_flags_leak(self):
        """
        When a pipe has only a flow meter (no pressure sensor), sustained surge
        flags suspected_leak with medium severity and explains single-sensor status.
        """
        detector = LeakDetector()
        pipe = {"id": 2, "name": "Branch-2", "from_node_name": "Beta", "to_node_name": "Gamma"}
        sensors = [
            {"sensor_id": 3, "sensor_name": "FM-Single", "sensor_type": "flow_meter", "pipe_id": 2, "unit": "m³/h", "status": "active"},
        ]
        history = [
            {"reading_id": i, "reading_value": 10.0 + (i % 2) * 0.1, "recorded_at": f"2026-11-01T10:0{i}:00Z", "scenario": "normal"}
            for i in range(5)
        ]
        window = [
            {"reading_id": 5 + i, "reading_value": 14.5 + (i % 2) * 0.3, "recorded_at": f"2026-11-01T10:1{i}:00Z", "scenario": "leak"}
            for i in range(5)
        ]
        readings = {3: history + window}

        diag = detector.diagnose_pipe(pipe, sensors, readings)

        assert diag.status == "suspected_leak"
        assert diag.severity in ("medium", "high")
        assert "single-sensor leak anomaly" in diag.evidence.lower()
        assert "no pressure sensor installed" in diag.evidence.lower()

    def test_pipe_with_no_active_sensors(self):
        """Pipes with no active sensors return status='insufficient_data'."""
        detector = LeakDetector()
        pipe = {"id": 99, "name": "Unmonitored-Pipe", "from_node_name": "A", "to_node_name": "B"}
        sensors = []
        readings = {}

        diag = detector.diagnose_pipe(pipe, sensors, readings)

        assert diag.status == "insufficient_data"
        assert "no active telemetry sensors" in diag.evidence.lower()


# ═════════════════════════════════════════════════════════════════════════════
#  4. DATABASE PERSISTENCE, SCENARIO VALIDATION & LIFECYCLE
# ═════════════════════════════════════════════════════════════════════════════

class TestDatabasePersistenceAndLifecycle:
    def test_record_sensor_reading_scenario_validation(self, tmp_db, detection_network):
        sensor_id = detection_network["sensors"]["s1_flow"]

        # Normal scenario
        r1 = tmp_db.record_sensor_reading(sensor_id, 350.0, scenario="normal")
        assert r1 > 0

        # Leak scenario
        r2 = tmp_db.record_sensor_reading(sensor_id, 600.0, scenario="leak")
        assert r2 > 0

        # None scenario (legacy/unspecified)
        r3 = tmp_db.record_sensor_reading(sensor_id, 350.0, scenario=None)
        assert r3 > 0

        # Invalid scenario raises ValueError
        with pytest.raises(ValueError, match="Invalid simulation scenario"):
            tmp_db.record_sensor_reading(sensor_id, 350.0, scenario="invalid_scenario")

    def test_record_leak_event_creates_persisted_event(self, tmp_db, detection_network):
        pipe_id = detection_network["pipe_1"]

        event_id, is_new = tmp_db.record_leak_event(
            pipe_id=pipe_id,
            severity="high",
            evidence_description="Test high flow surge and pressure drop.",
            flow_anomaly_score=3.14,
            pressure_anomaly_score=-2.85,
            baseline_values="Flow: 50.0 L/min, Pressure: 60.0 psi",
            observed_values="Flow: 85.0 L/min, Pressure: 32.0 psi",
            is_simulated=True,
        )

        assert is_new is True
        assert event_id > 0

        event = tmp_db.get_leak_event_by_id(event_id)
        assert event is not None
        assert event["pipe_id"] == pipe_id
        assert event["severity"] == "high"
        assert event["status"] == "open"
        assert event["is_simulated"] == 1
        assert event["flow_anomaly_score"] == pytest.approx(3.14)
        assert event["pressure_anomaly_score"] == pytest.approx(-2.85)

    def test_deduplication_prevents_duplicate_open_events(self, tmp_db, detection_network):
        """
        Calling record_leak_event multiple times while an event is 'open' or 'investigating'
        updates the existing event without creating a duplicate row.
        """
        pipe_id = detection_network["pipe_1"]

        # First detection: creates new event
        event_id_1, is_new_1 = tmp_db.record_leak_event(
            pipe_id=pipe_id,
            severity="medium",
            evidence_description="Initial anomaly observation.",
            flow_anomaly_score=2.6,
        )
        assert is_new_1 is True

        # Second detection: anomaly continues
        event_id_2, is_new_2 = tmp_db.record_leak_event(
            pipe_id=pipe_id,
            severity="high",
            evidence_description="Updated anomaly with escalated severity.",
            flow_anomaly_score=3.5,
        )
        assert is_new_2 is False
        assert event_id_2 == event_id_1

        # Verify only 1 open event exists for this pipe
        events = tmp_db.get_open_leak_events()
        pipe_events = [e for e in events if e["pipe_id"] == pipe_id]
        assert len(pipe_events) == 1
        assert pipe_events[0]["severity"] == "high"
        assert pipe_events[0]["evidence_description"] == "Updated anomaly with escalated severity."

    def test_resolved_event_allows_new_event_creation(self, tmp_db, detection_network):
        """Once an event is resolved, subsequent anomalies create a new event."""
        pipe_id = detection_network["pipe_1"]

        # First event
        event_id_1, is_new_1 = tmp_db.record_leak_event(
            pipe_id=pipe_id,
            severity="medium",
            evidence_description="First event.",
        )
        assert is_new_1 is True

        # Mark as resolved
        tmp_db.update_leak_event_status(event_id_1, "resolved")
        resolved_ev = tmp_db.get_leak_event_by_id(event_id_1)
        assert resolved_ev["status"] == "resolved"
        assert resolved_ev["resolved_at"] is not None

        # Subsequent anomaly creates a fresh event
        event_id_2, is_new_2 = tmp_db.record_leak_event(
            pipe_id=pipe_id,
            severity="high",
            evidence_description="New anomaly after repair.",
        )
        assert is_new_2 is True
        assert event_id_2 != event_id_1

    def test_update_leak_event_status_validations(self, tmp_db, detection_network):
        pipe_id = detection_network["pipe_1"]
        event_id, _ = tmp_db.record_leak_event(pipe_id=pipe_id, severity="low", evidence_description="Test")

        # Transition open -> investigating
        tmp_db.update_leak_event_status(event_id, "investigating")
        ev = tmp_db.get_leak_event_by_id(event_id)
        assert ev["status"] == "investigating"
        assert ev["resolved_at"] is None

        # Transition investigating -> resolved
        tmp_db.update_leak_event_status(event_id, "resolved")
        ev = tmp_db.get_leak_event_by_id(event_id)
        assert ev["status"] == "resolved"
        assert ev["resolved_at"] is not None

        # Invalid status raises ValueError
        with pytest.raises(ValueError, match="Invalid event status"):
            tmp_db.update_leak_event_status(event_id, "invalid_status")

        # Non-existent event raises LookupError
        with pytest.raises(LookupError, match="not found"):
            tmp_db.update_leak_event_status(99999, "resolved")

    def test_summary_counter_helpers(self, tmp_db, detection_network):
        pipe_1 = detection_network["pipe_1"]
        pipe_2 = detection_network["pipe_2"]

        assert tmp_db.count_open_leak_events() == 0
        assert tmp_db.count_pipes_under_investigation() == 0
        assert tmp_db.count_total_leak_events() == 0

        # Record events on 2 different pipes
        e1, _ = tmp_db.record_leak_event(pipe_1, "high", "Pipe 1 anomaly")
        e2, _ = tmp_db.record_leak_event(pipe_2, "medium", "Pipe 2 anomaly")

        assert tmp_db.count_open_leak_events() == 2
        assert tmp_db.count_pipes_under_investigation() == 2
        assert tmp_db.count_total_leak_events() == 2

        # Mark pipe 1 resolved
        tmp_db.update_leak_event_status(e1, "resolved")
        assert tmp_db.count_open_leak_events() == 1
        assert tmp_db.count_pipes_under_investigation() == 1
        assert tmp_db.count_total_leak_events() == 2


# ═════════════════════════════════════════════════════════════════════════════
#  5. NETWORK-WIDE DETECTION EXECUTION INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════

class TestNetworkDetectionExecution:
    def test_run_detection_network_end_to_end(self, tmp_db, detection_network):
        """
        Verify run_detection_network evaluates all pipes, detects anomalies on pipe 1,
        keeps pipe 2 normal, and handles pipe 3 with insufficient baseline history.
        """
        s1_flow = detection_network["sensors"]["s1_flow"]
        s1_press = detection_network["sensors"]["s1_press"]
        s2_flow = detection_network["sensors"]["s2_flow"]

        # Pipe 1: 5 normal history readings + 5 leak window readings
        record_readings(tmp_db, s1_flow, [50.0, 50.1, 49.9, 50.2, 49.8], unit="L/min", scenario="normal")
        record_readings(tmp_db, s1_flow, [85.0, 86.0, 87.0, 86.5, 87.2], unit="L/min", scenario="leak")

        record_readings(tmp_db, s1_press, [60.0, 60.2, 59.8, 60.1, 59.9], unit="psi", scenario="normal")
        record_readings(tmp_db, s1_press, [32.0, 31.0, 30.5, 31.2, 30.8], unit="psi", scenario="leak")

        # Pipe 2: 5 normal history readings + 5 normal window readings
        record_readings(tmp_db, s2_flow, [15.0, 15.1, 14.9, 15.0, 15.2], unit="m³/h", scenario="normal")
        record_readings(tmp_db, s2_flow, [15.1, 14.8, 15.0, 15.2, 14.9], unit="m³/h", scenario="normal")

        detector = LeakDetector()
        diagnostics = detector.run_detection_network(tmp_db)

        assert len(diagnostics) == 3

        # Pipe 1 should be suspected_leak
        diag_1 = next(d for d in diagnostics if d.pipe_id == detection_network["pipe_1"])
        assert diag_1.status == "suspected_leak"
        assert diag_1.severity in ("high", "critical")
        assert diag_1.event_id is not None
        assert diag_1.is_new_event is True
        assert diag_1.baseline_flow_samples == 5
        assert diag_1.window_flow_samples == 5

        # Pipe 2 should be normal
        diag_2 = next(d for d in diagnostics if d.pipe_id == detection_network["pipe_2"])
        assert diag_2.status == "normal"
        assert diag_2.severity == "low"
        assert diag_2.baseline_flow_samples == 5
        assert diag_2.window_flow_samples == 5

        # Pipe 3 should be insufficient_data (no readings recorded for s3)
        diag_3 = next(d for d in diagnostics if d.pipe_id == detection_network["pipe_3"])
        assert diag_3.status == "insufficient_data"

        # Database leak_events record verified
        events = tmp_db.get_all_leak_events()
        assert len(events) == 1
        assert events[0]["pipe_id"] == detection_network["pipe_1"]
        assert events[0]["status"] == "open"
        assert events[0]["is_simulated"] == 1


# ═════════════════════════════════════════════════════════════════════════════
#  6. FLASK ROUTE INTEGRATION TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestDetectionRoutes:
    def test_get_leak_detection_page_renders_successfully(self, app_client, detection_network):
        res = app_client.get("/leak-detection")
        assert res.status_code == 200

        html = res.data.decode("utf-8")
        assert "Leak Detection Engine" in html
        assert "Hydraulic Anomaly Simulation Notice" in html
        assert "Suspected Leak Events" in html
        assert "Run Anomaly Detection Engine" in html
        assert "Monitored Pipes Diagnostic Overview" in html
        assert "Suspected Leak Events Log" in html
        assert "AQUAGRID · MCA PROJECT PROTOTYPE" in html

    def test_post_leak_detection_run_redirects_and_flashes(self, app_client, detection_network):
        res = app_client.post("/leak-detection/run", follow_redirects=True)
        assert res.status_code == 200

        html = res.data.decode("utf-8")
        assert "Detection completed" in html
        assert "Insufficient telemetry is available to determine network health" in html
        assert "/leak-detection" in res.request.path

    def test_post_leak_detection_update_status(self, app_client, tmp_db, detection_network):
        pipe_id = detection_network["pipe_1"]
        event_id, _ = tmp_db.record_leak_event(
            pipe_id=pipe_id,
            severity="high",
            evidence_description="Initial leak detected.",
        )

        res = app_client.post(
            f"/leak-detection/events/{event_id}/status",
            data={"status": "investigating"},
            follow_redirects=True,
        )
        assert res.status_code == 200

        html = res.data.decode("utf-8")
        assert f"Leak event #{event_id} status updated to &#39;investigating&#39;." in html

        ev = tmp_db.get_leak_event_by_id(event_id)
        assert ev["status"] == "investigating"

    def test_post_leak_detection_update_status_invalid_value(self, app_client, tmp_db, detection_network):
        pipe_id = detection_network["pipe_1"]
        event_id, _ = tmp_db.record_leak_event(pipe_id=pipe_id, severity="high", evidence_description="Test")

        res = app_client.post(
            f"/leak-detection/events/{event_id}/status",
            data={"status": "bogus_status"},
            follow_redirects=True,
        )
        assert res.status_code == 200
        html = res.data.decode("utf-8")
        assert "Invalid event status" in html

    def test_get_on_post_endpoints_returns_405(self, app_client):
        res_run = app_client.get("/leak-detection/run")
        assert res_run.status_code == 405

        res_status = app_client.get("/leak-detection/events/1/status")
        assert res_status.status_code == 405
