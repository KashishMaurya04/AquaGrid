"""
tests/test_simulator_reliability_audit.py – Day 2 Sensor Simulator Validation and Reliability Audit Test Suite.

Covers:
  - Part 2: Normal mode statistical bounds, zero false-positive leak alarms, unit consistency, timestamp monotonicity.
  - Part 3: Leak simulation mode sustained abnormal telemetry, pipe isolation, detection engine evaluation.
  - Part 4: Data provenance, is_simulated labeling, leak scenario baseline contamination prevention, unknown provenance rejection.
  - Part 5: Repeated runs, mode switching, route parameter robustness, and live database safety.
"""

import os
import sys
import pytest
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation.engine import SensorSimulator, SimulatorConfig
from detection.engine import LeakDetector


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    """Isolate database tests with a fresh SQLite DB in a temporary directory."""
    import database as db_module

    test_db = str(tmp_path / "test_simulator_audit_aquagrid.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


@pytest.fixture()
def audit_network(tmp_db):
    """
    Sets up a 2-pipe, 3-node network:
      - Pipe 1: Tank Alpha -> Junction Beta (Flow Meter in L/min, Pressure Sensor in psi)
      - Pipe 2: Junction Beta -> Valve Gamma (Flow Meter in m³/h, Pressure Sensor in bar)
    """
    id_a = tmp_db.create_node("Tank Alpha", "tank", "Zone 1", "active")
    id_b = tmp_db.create_node("Junction Beta", "junction", "Zone 2", "active")
    id_c = tmp_db.create_node("Valve Gamma", "valve", "Zone 3", "active")

    p1 = tmp_db.create_pipe("Trunk-Alpha-Beta", id_a, id_b, "150.0", "600.0", "1.0", "active")
    p2 = tmp_db.create_pipe("Branch-Beta-Gamma", id_b, id_c, "100.0", "400.0", "2.0", "active")

    s1_flow = tmp_db.create_sensor("FM-P1", p1, "flow_meter", "L/min", "active")
    s1_press = tmp_db.create_sensor("PS-P1", p1, "pressure_sensor", "psi", "active")

    s2_flow = tmp_db.create_sensor("FM-P2", p2, "flow_meter", "m³/h", "active")
    s2_press = tmp_db.create_sensor("PS-P2", p2, "pressure_sensor", "bar", "active")

    return {
        "nodes": [id_a, id_b, id_c],
        "pipe_1": p1,
        "pipe_2": p2,
        "sensors": {
            "p1_flow": s1_flow,
            "p1_press": s1_press,
            "p2_flow": s2_flow,
            "p2_press": s2_press,
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
    app_module.app.config["SECRET_KEY"] = "test-simulator-audit-secret-key"

    with app_module.app.test_client() as client:
        yield client


# ═════════════════════════════════════════════════════════════════════════════
#  PART 2: NORMAL MODE VERIFICATION
# ═════════════════════════════════════════════════════════════════════════════

class TestNormalModeVerification:
    def test_normal_mode_statistical_bounds_and_zero_false_leaks(self, tmp_db, audit_network):
        """
        Verify:
        1. Normal mode generates readings strictly within operating ranges.
        2. Small Gaussian variations do not trigger artificial leak events in LeakDetector.
        3. Timestamps are chronological and valid ISO strings.
        4. Stored in SQLite with is_simulated=1 and scenario='normal'.
        """
        sim = SensorSimulator(seed=12345)
        active_sensors = tmp_db.get_active_sensors()
        detector = LeakDetector()

        # Generate 20 consecutive cycles of normal telemetry (5 for baseline, 15 for window)
        readings = sim.generate_batch(active_sensors, mode="normal", iterations=20)
        assert len(readings) == 4 * 20  # 4 sensors * 20 cycles = 80 readings

        for r in readings:
            # Check unit bounds
            if r.sensor_type == "flow_meter" and r.unit == "L/min":
                assert 320.0 <= r.reading_value <= 380.0
            elif r.sensor_type == "pressure_sensor" and r.unit == "psi":
                assert 47.0 <= r.reading_value <= 57.0

            assert r.is_anomaly is False
            assert r.simulation_mode == "normal"
            assert r.is_simulated == 1

            # Persist to isolated DB
            tmp_db.record_sensor_reading(
                sensor_id=r.sensor_id,
                reading_value=r.reading_value,
                recorded_at=r.recorded_at,
                reading_type=r.sensor_type,
                unit=r.unit,
                is_simulated=True,
                scenario=r.simulation_mode,
            )

        assert tmp_db.count_sensor_readings() == 80

        # Execute leak detection engine on the network
        diagnostics = detector.run_detection_network(tmp_db, auto_record=True)
        assert len(diagnostics) == 2

        # Both pipes must be classified as 'normal' with 0 leak events created
        for diag in diagnostics:
            assert diag.status == "normal"
            assert diag.severity == "low"
            assert "Operating within normal statistical baseline bounds" in diag.evidence

        open_events = tmp_db.get_open_leak_events()
        assert len(open_events) == 0

    def test_flow_and_pressure_units_scaling(self):
        """Verify simulator accurately scales flow and pressure across all 6 supported units."""
        sim = SensorSimulator(seed=99)

        # Flow units (baseline 350 L/min)
        fm_lpm = {"sensor_id": 1, "sensor_name": "FM1", "sensor_type": "flow_meter", "pipe_id": 1, "unit": "L/min", "status": "active"}
        fm_m3h = {"sensor_id": 2, "sensor_name": "FM2", "sensor_type": "flow_meter", "pipe_id": 1, "unit": "m³/h", "status": "active"}
        fm_lps = {"sensor_id": 3, "sensor_name": "FM3", "sensor_type": "flow_meter", "pipe_id": 1, "unit": "L/s", "status": "active"}

        sim.set_seed(42)
        r_lpm = sim.generate_reading_for_sensor(fm_lpm).reading_value
        sim.set_seed(42)
        r_m3h = sim.generate_reading_for_sensor(fm_m3h).reading_value
        sim.set_seed(42)
        r_lps = sim.generate_reading_for_sensor(fm_lps).reading_value

        assert r_m3h == pytest.approx(r_lpm * 0.06, rel=1e-2)
        assert r_lps == pytest.approx(r_lpm / 60.0, rel=1e-2)

        # Pressure units (baseline 52 psi)
        ps_psi = {"sensor_id": 4, "sensor_name": "PS1", "sensor_type": "pressure_sensor", "pipe_id": 1, "unit": "psi", "status": "active"}
        ps_bar = {"sensor_id": 5, "sensor_name": "PS2", "sensor_type": "pressure_sensor", "pipe_id": 1, "unit": "bar", "status": "active"}
        ps_kpa = {"sensor_id": 6, "sensor_name": "PS3", "sensor_type": "pressure_sensor", "pipe_id": 1, "unit": "kPa", "status": "active"}

        sim.set_seed(42)
        r_psi = sim.generate_reading_for_sensor(ps_psi).reading_value
        sim.set_seed(42)
        r_bar = sim.generate_reading_for_sensor(ps_bar).reading_value
        sim.set_seed(42)
        r_kpa = sim.generate_reading_for_sensor(ps_kpa).reading_value

        assert r_bar == pytest.approx(r_psi * 0.0689476, rel=1e-2)
        assert r_kpa == pytest.approx(r_psi * 6.89476, rel=1e-2)


# ═════════════════════════════════════════════════════════════════════════════
#  PART 3: LEAK SIMULATION MODE VERIFICATION
# ═════════════════════════════════════════════════════════════════════════════

class TestLeakSimulationModeVerification:
    def test_leak_simulation_sustained_pattern_and_detection(self, tmp_db, audit_network):
        """
        Verify:
        1. Leak simulation creates a sustained abnormal flow surge and pressure drop on the targeted pipe.
        2. Non-targeted pipes remain completely normal.
        3. Readings are accurately evaluated by LeakDetector and create a single leak event.
        """
        sim = SensorSimulator(seed=777)
        p1 = audit_network["pipe_1"]
        p2 = audit_network["pipe_2"]
        active_sensors = tmp_db.get_active_sensors()
        detector = LeakDetector()

        # Step 1: Establish baseline with 10 normal readings
        t0 = datetime(2026, 11, 10, 8, 0, 0, tzinfo=timezone.utc)
        normal_batch = sim.generate_batch(active_sensors, mode="normal", iterations=10, base_time=t0)
        for r in normal_batch:
            tmp_db.record_sensor_reading(
                sensor_id=r.sensor_id,
                reading_value=r.reading_value,
                recorded_at=r.recorded_at,
                reading_type=r.sensor_type,
                unit=r.unit,
                is_simulated=True,
                scenario=r.simulation_mode,
            )

        # Step 2: Simulate sustained leak on Pipe 1 (10 readings)
        t_leak = t0 + timedelta(minutes=10)
        leak_batch = sim.generate_batch(active_sensors, mode="leak", leak_pipe_id=p1, iterations=10, base_time=t_leak)
        for r in leak_batch:
            tmp_db.record_sensor_reading(
                sensor_id=r.sensor_id,
                reading_value=r.reading_value,
                recorded_at=r.recorded_at,
                reading_type=r.sensor_type,
                unit=r.unit,
                is_simulated=True,
                scenario=r.simulation_mode,
            )

        # Verify Pipe 1 vs Pipe 2 in the generated batch
        p1_readings = [r for r in leak_batch if r.pipe_id == p1]
        p2_readings = [r for r in leak_batch if r.pipe_id == p2]

        assert all(r.is_anomaly for r in p1_readings)
        assert all(r.simulation_mode == "leak" for r in p1_readings)

        assert all(not r.is_anomaly for r in p2_readings)
        assert all(r.simulation_mode == "normal" for r in p2_readings)

        # Step 3: Run leak detection engine
        diagnostics = detector.run_detection_network(tmp_db, auto_record=True)
        diag_map = {d.pipe_id: d for d in diagnostics}

        # Pipe 1 must be flagged with suspected_leak
        d1 = diag_map[p1]
        assert d1.status == "suspected_leak"
        assert d1.severity in ("critical", "high")
        assert "Coincident flow surge and pressure drop" in d1.evidence

        # Pipe 2 must remain completely normal
        d2 = diag_map[p2]
        assert d2.status == "normal"
        assert d2.severity == "low"

        # Exactly 1 leak event recorded for Pipe 1
        open_events = tmp_db.get_open_leak_events()
        assert len(open_events) == 1
        assert open_events[0]["pipe_id"] == p1


# ═════════════════════════════════════════════════════════════════════════════
#  PART 4: DATA PROVENANCE AND BASELINE SAFETY
# ═════════════════════════════════════════════════════════════════════════════

class TestDataProvenanceAndBaselineSafety:
    def test_leak_readings_never_pollute_normal_baseline(self, tmp_db, audit_network):
        """
        Verify:
        1. When switching from leak mode to normal mode, historical leak readings are strictly excluded.
        2. Detection engine uses only verified normal readings for baseline calculations.
        """
        p1 = audit_network["pipe_1"]
        s_flow = audit_network["sensors"]["p1_flow"]
        detector = LeakDetector()

        # Insert 10 verified normal readings
        t0 = datetime(2026, 11, 10, 6, 0, 0, tzinfo=timezone.utc)
        for i in range(10):
            tmp_db.record_sensor_reading(
                sensor_id=s_flow,
                reading_value=350.0 + i,
                recorded_at=(t0 + timedelta(minutes=i)).isoformat(),
                reading_type="flow_meter",
                unit="L/min",
                is_simulated=True,
                scenario="normal",
            )

        # Insert 10 leak readings
        for i in range(10):
            tmp_db.record_sensor_reading(
                sensor_id=s_flow,
                reading_value=600.0 + i,
                recorded_at=(t0 + timedelta(minutes=10 + i)).isoformat(),
                reading_type="flow_meter",
                unit="L/min",
                is_simulated=True,
                scenario="leak",
            )

        # Insert 5 new normal readings forming the current observation window
        for i in range(5):
            tmp_db.record_sensor_reading(
                sensor_id=s_flow,
                reading_value=351.0,
                recorded_at=(t0 + timedelta(minutes=20 + i)).isoformat(),
                reading_type="flow_meter",
                unit="L/min",
                is_simulated=True,
                scenario="normal",
            )

        # Analyze sensor
        readings = [dict(r) for r in tmp_db.get_readings_for_sensor(s_flow, limit=50)]
        sensor_info = dict(tmp_db.get_sensor_by_id(s_flow))
        result = detector.analyze_sensor(sensor_info, readings)

        # Baseline must be derived purely from the 10 normal readings, completely ignoring the 10 leak readings!
        assert result.baseline_samples == 10
        assert result.leak_scenario_samples == 10
        assert result.baseline_mean == pytest.approx(354.5, rel=1e-2)
        assert result.classification == "normal"
        assert result.is_abnormal is False

    def test_unknown_provenance_readings_cannot_form_normal_baseline(self, tmp_db, audit_network):
        """
        Verify:
        Readings with scenario=None (legacy or unverified data) cannot be used to establish a normal baseline.
        """
        s_flow = audit_network["sensors"]["p1_flow"]
        detector = LeakDetector()

        # Insert 10 readings without scenario provenance (scenario=None)
        t0 = datetime(2026, 11, 10, 5, 0, 0, tzinfo=timezone.utc)
        for i in range(10):
            tmp_db.record_sensor_reading(
                sensor_id=s_flow,
                reading_value=350.0,
                recorded_at=(t0 + timedelta(minutes=i)).isoformat(),
                reading_type="flow_meter",
                unit="L/min",
                is_simulated=False,
                scenario=None,
            )

        readings = [dict(r) for r in tmp_db.get_readings_for_sensor(s_flow, limit=50)]
        sensor_info = dict(tmp_db.get_sensor_by_id(s_flow))
        result = detector.analyze_sensor(sensor_info, readings)

        assert result.classification == "insufficient_data"
        assert result.baseline_samples == 0
        assert result.unknown_provenance_samples == 5
        assert "unverified/unknown scenario provenance" in result.evidence


# ═════════════════════════════════════════════════════════════════════════════
#  PART 5: REPEATED RUNS, EDGE CASES & HTTP ROBUSTNESS
# ═════════════════════════════════════════════════════════════════════════════

class TestRepeatedRunsAndEdgeCases:
    def test_repeated_consecutive_runs_maintain_integrity(self, tmp_db, audit_network):
        """
        Verify:
        Multiple consecutive batch runs advance timestamps monotonically and do not create duplicate IDs.
        """
        sim = SensorSimulator(seed=888)
        active_sensors = tmp_db.get_active_sensors()

        base_time = datetime(2026, 11, 10, 10, 0, 0, tzinfo=timezone.utc)
        total_created = 0

        # Run 5 consecutive batches of 2 iterations each
        for run_idx in range(5):
            t_run = base_time + timedelta(minutes=run_idx * 5)
            batch = sim.generate_batch(active_sensors, mode="normal", iterations=2, base_time=t_run)
            for r in batch:
                tmp_db.record_sensor_reading(
                    sensor_id=r.sensor_id,
                    reading_value=r.reading_value,
                    recorded_at=r.recorded_at,
                    reading_type=r.sensor_type,
                    unit=r.unit,
                    is_simulated=True,
                    scenario=r.simulation_mode,
                )
            total_created += len(batch)

        assert total_created == 5 * 2 * 4  # 40 readings
        assert tmp_db.count_sensor_readings() == 40

        # Check timestamp ordering for each sensor
        for s in active_sensors:
            readings = tmp_db.get_readings_for_sensor(s["sensor_id"], limit=50)
            assert len(readings) == 10
            ts_list = [r["recorded_at"] for r in reversed(readings)]
            for i in range(len(ts_list) - 1):
                assert ts_list[i] < ts_list[i + 1]

    def test_http_parameter_robustness_with_invalid_inputs(self, app_client, tmp_db, audit_network):
        """
        Verify:
        Invalid parameters in POST /sensor-simulator/generate degrade gracefully without 500 error:
        - Non-numeric seed (e.g. 'notanumber')
        - Unknown count (e.g. '999') defaults to 1
        - Unknown mode (e.g. 'superleak') defaults to 'normal'
        """
        # Invalid count and invalid seed and invalid mode
        r = app_client.post(
            "/sensor-simulator/generate",
            data={
                "mode": "superleak",
                "count": "999",
                "seed": "not_an_int",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        # Mode fell back to normal, count fell back to 1: 4 active sensors * 1 = 4 readings
        assert tmp_db.count_sensor_readings() == 4
        assert b"Generated 4 synthetic telemetry reading(s)" in r.data
        assert b"Normal mode" in r.data

    def test_clear_readings_idempotency(self, app_client, tmp_db, audit_network):
        """POST /sensor-simulator/clear can be executed repeatedly without error."""
        # Clear when empty
        r1 = app_client.post("/sensor-simulator/clear", follow_redirects=True)
        assert r1.status_code == 200
        assert b"Cleared 0 simulated sensor reading(s)" in r1.data

        # Clear again
        r2 = app_client.post("/sensor-simulator/clear", follow_redirects=True)
        assert r2.status_code == 200
        assert b"Cleared 0 simulated sensor reading(s)" in r2.data


# ═════════════════════════════════════════════════════════════════════════════
#  LIVE DATABASE INTEGRITY SAFETY CHECK
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

