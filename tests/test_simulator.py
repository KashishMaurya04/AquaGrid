"""
tests/test_simulator.py – Unit and integration test suite for the Sensor Simulator milestone.

Covers:
  1. Normal flow reading generation (range, fluctuation, units L/min, m³/h, L/s)
  2. Normal pressure reading generation (range, fluctuation, units psi, bar, kPa)
  3. Reproducibility with configurable random seed
  4. Leak scenario flow surge generation (abnormal surge pattern on target pipe)
  5. Leak scenario pressure drop generation (abnormal drop pattern on target pipe)
  6. Unaffected sensors on other pipes remain normal during leak scenario
  7. Inactive sensors strictly excluded from reading generation
  8. Pipe selection validation in leak scenario (missing pipe, non-existent pipe, pipe without active sensors)
  9. Handling when no active sensors exist in the network
 10. Database persistence and verification of is_simulated == 1
 11. Database query helpers (get_latest_sensor_readings, get_latest_reading_by_type, clear_sensor_readings)
 12. HTTP route integration:
     - GET /sensor-simulator loads cleanly with summary cards and notice
     - POST /sensor-simulator/generate (single reading and 10-reading batch)
     - POST /sensor-simulator/clear removes readings safely
     - GET on state-changing endpoints returns 405 Method Not Allowed
     - Synthetic demonstration disclaimer verification
     - Sidebar navigation link present across all pages
 13. Existing node, pipe, sensor, graph, and route functionality remains unaffected
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation.engine import SensorSimulator, SimulatorConfig, GeneratedReading


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    """Isolate database tests with a fresh SQLite DB in a temporary directory."""
    import database as db_module

    test_db = str(tmp_path / "test_simulator_aquagrid.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


@pytest.fixture()
def simulated_network(tmp_db):
    """
    Sets up a 3-node, 2-pipe network with active and inactive sensors:
      - Pipe 1: Tank Alpha -> Junction Beta
        * Sensor 1: Flow Meter (active, L/min)
        * Sensor 2: Pressure Sensor (active, psi)
        * Sensor 3: Flow Meter (inactive, L/min)
      - Pipe 2: Junction Beta -> Valve Gamma
        * Sensor 4: Flow Meter (active, m³/h)
        * Sensor 5: Pressure Sensor (active, bar)
    """
    id_a = tmp_db.create_node("Tank Alpha", "tank", "Zone 1", "active")
    id_b = tmp_db.create_node("Junction Beta", "junction", "Zone 2", "active")
    id_c = tmp_db.create_node("Valve Gamma", "valve", "Zone 3", "active")

    pipe_1 = tmp_db.create_pipe("Main-Trunk-1", id_a, id_b, "200.0", "600.0", "1.0", "active")
    pipe_2 = tmp_db.create_pipe("Sub-Branch-2", id_b, id_c, "150.0", "400.0", "2.0", "active")

    s1 = tmp_db.create_sensor("FM-Pipe1-Active", pipe_1, "flow_meter", "L/min", "active")
    s2 = tmp_db.create_sensor("PS-Pipe1-Active", pipe_1, "pressure_sensor", "psi", "active")
    s3 = tmp_db.create_sensor("FM-Pipe1-Inactive", pipe_1, "flow_meter", "L/min", "inactive")

    s4 = tmp_db.create_sensor("FM-Pipe2-Active", pipe_2, "flow_meter", "m³/h", "active")
    s5 = tmp_db.create_sensor("PS-Pipe2-Active", pipe_2, "pressure_sensor", "bar", "active")

    return {
        "nodes": [id_a, id_b, id_c],
        "pipe_1": pipe_1,
        "pipe_2": pipe_2,
        "sensors": {
            "s1_flow_active": s1,
            "s2_press_active": s2,
            "s3_flow_inactive": s3,
            "s4_flow_m3h": s4,
            "s5_press_bar": s5,
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
    app_module.app.config["SECRET_KEY"] = "test-simulator-secret-key"

    with app_module.app.test_client() as client:
        yield client


# ═════════════════════════════════════════════════════════════════════════════
#  1. SIMULATOR ENGINE UNIT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestSimulatorEngine:
    def test_normal_flow_reading_generation(self):
        """Flow reading is centered around 350 L/min baseline with small natural fluctuation."""
        sim = SensorSimulator(seed=123)
        sensor = {
            "sensor_id": 1,
            "sensor_name": "FM-01",
            "sensor_type": "flow_meter",
            "pipe_id": 10,
            "pipe_name": "Pipe-10",
            "unit": "L/min",
            "status": "active",
        }
        reading = sim.generate_reading_for_sensor(sensor, is_leak_affected=False)

        assert reading.sensor_id == 1
        assert reading.sensor_type == "flow_meter"
        assert reading.unit == "L/min"
        assert reading.is_simulated == 1
        assert reading.is_anomaly is False
        assert reading.simulation_mode == "normal"
        # Baseline is 350.0 with std dev 7.0; reading should be well within [325, 375]
        assert 325.0 <= reading.reading_value <= 375.0

    def test_normal_pressure_reading_generation(self):
        """Pressure reading is centered around 52 psi baseline with small fluctuation."""
        sim = SensorSimulator(seed=456)
        sensor = {
            "sensor_id": 2,
            "sensor_name": "PS-01",
            "sensor_type": "pressure_sensor",
            "pipe_id": 10,
            "pipe_name": "Pipe-10",
            "unit": "psi",
            "status": "active",
        }
        reading = sim.generate_reading_for_sensor(sensor, is_leak_affected=False)

        assert reading.sensor_id == 2
        assert reading.sensor_type == "pressure_sensor"
        assert reading.unit == "psi"
        assert reading.is_anomaly is False
        assert reading.simulation_mode == "normal"
        # Baseline is 52.0 with std dev 1.1; reading should be within [48.0, 56.0]
        assert 48.0 <= reading.reading_value <= 56.0

    def test_flow_unit_conversions(self):
        """Simulator converts flow into m³/h and L/s accurately."""
        sim = SensorSimulator(seed=789)
        base = {
            "sensor_id": 3,
            "sensor_name": "FM-M3H",
            "sensor_type": "flow_meter",
            "pipe_id": 10,
            "pipe_name": "Pipe-10",
            "unit": "m³/h",
            "status": "active",
        }
        reading_m3h = sim.generate_reading_for_sensor(base, is_leak_affected=False)
        # ~350 L/min * 0.06 = ~21.0 m³/h
        assert 19.0 <= reading_m3h.reading_value <= 23.0

        base["unit"] = "L/s"
        reading_lps = sim.generate_reading_for_sensor(base, is_leak_affected=False)
        # ~350 L/min / 60 = ~5.83 L/s
        assert 5.0 <= reading_lps.reading_value <= 6.8

    def test_pressure_unit_conversions(self):
        """Simulator converts pressure into bar and kPa accurately."""
        sim = SensorSimulator(seed=101)
        base = {
            "sensor_id": 4,
            "sensor_name": "PS-Bar",
            "sensor_type": "pressure_sensor",
            "pipe_id": 10,
            "pipe_name": "Pipe-10",
            "unit": "bar",
            "status": "active",
        }
        reading_bar = sim.generate_reading_for_sensor(base, is_leak_affected=False)
        # 52 psi * 0.0689476 = ~3.58 bar
        assert 3.2 <= reading_bar.reading_value <= 4.0

        base["unit"] = "kPa"
        reading_kpa = sim.generate_reading_for_sensor(base, is_leak_affected=False)
        # 52 psi * 6.89476 = ~358.5 kPa
        assert 330.0 <= reading_kpa.reading_value <= 390.0

    def test_reproducibility_with_seed(self):
        """Identical seeds produce exact identical reading sequences."""
        sensor = {
            "sensor_id": 1,
            "sensor_name": "FM-Repro",
            "sensor_type": "flow_meter",
            "pipe_id": 1,
            "unit": "L/min",
            "status": "active",
        }
        sim1 = SensorSimulator(seed=9999)
        sim2 = SensorSimulator(seed=9999)

        r1 = [sim1.generate_reading_for_sensor(sensor).reading_value for _ in range(5)]
        r2 = [sim2.generate_reading_for_sensor(sensor).reading_value for _ in range(5)]
        assert r1 == r2

        # Different seed produces variation
        sim3 = SensorSimulator(seed=1111)
        r3 = [sim3.generate_reading_for_sensor(sensor).reading_value for _ in range(5)]
        assert r1 != r3

    def test_leak_scenario_flow_surge_and_pressure_drop(self):
        """
        In leak mode:
        - Flow meter surges to ~1.7x baseline (550 - 650 L/min)
        - Pressure sensor drops by ~26 psi (down to 21 - 31 psi)
        - is_anomaly is marked True
        """
        sim = SensorSimulator(seed=42)

        flow_sensor = {
            "sensor_id": 1,
            "sensor_name": "FM-Leak",
            "sensor_type": "flow_meter",
            "pipe_id": 5,
            "unit": "L/min",
            "status": "active",
        }
        reading_flow_leak = sim.generate_reading_for_sensor(flow_sensor, is_leak_affected=True)
        assert reading_flow_leak.is_anomaly is True
        assert reading_flow_leak.simulation_mode == "leak"
        assert reading_flow_leak.reading_value > 520.0  # conspicuous surge

        press_sensor = {
            "sensor_id": 2,
            "sensor_name": "PS-Leak",
            "sensor_type": "pressure_sensor",
            "pipe_id": 5,
            "unit": "psi",
            "status": "active",
        }
        reading_press_leak = sim.generate_reading_for_sensor(press_sensor, is_leak_affected=True)
        assert reading_press_leak.is_anomaly is True
        assert reading_press_leak.simulation_mode == "leak"
        assert reading_press_leak.reading_value < 35.0  # conspicuous pressure drop

    def test_leak_scenario_unaffected_sensors_stay_normal(self):
        """In batch leak generation, sensors on other pipes remain at normal baseline."""
        sim = SensorSimulator(seed=777)
        sensors = [
            {"sensor_id": 1, "sensor_name": "FM-P1", "sensor_type": "flow_meter", "pipe_id": 1, "unit": "L/min", "status": "active"},
            {"sensor_id": 2, "sensor_name": "FM-P2", "sensor_type": "flow_meter", "pipe_id": 2, "unit": "L/min", "status": "active"},
        ]
        # Simulate leak specifically on pipe 1
        batch = sim.generate_batch(active_sensors=sensors, mode="leak", leak_pipe_id=1, iterations=1)
        r_pipe1 = next(r for r in batch if r.pipe_id == 1)
        r_pipe2 = next(r for r in batch if r.pipe_id == 2)

        assert r_pipe1.is_anomaly is True
        assert r_pipe1.reading_value > 520.0

        assert r_pipe2.is_anomaly is False
        assert 330.0 <= r_pipe2.reading_value <= 370.0

    def test_inactive_sensor_raises_value_error(self):
        """Direct call to generate_reading_for_sensor on inactive sensor raises ValueError."""
        sim = SensorSimulator()
        inactive_sensor = {
            "sensor_id": 99,
            "sensor_name": "FM-Inactive",
            "sensor_type": "flow_meter",
            "pipe_id": 1,
            "unit": "L/min",
            "status": "inactive",
        }
        with pytest.raises(ValueError, match="inactive"):
            sim.generate_reading_for_sensor(inactive_sensor)

    def test_generate_batch_excludes_inactive_sensors(self):
        """Batch generation automatically skips inactive sensors."""
        sim = SensorSimulator()
        sensors = [
            {"sensor_id": 1, "sensor_name": "Active", "sensor_type": "flow_meter", "pipe_id": 1, "unit": "L/min", "status": "active"},
            {"sensor_id": 2, "sensor_name": "Inactive", "sensor_type": "flow_meter", "pipe_id": 1, "unit": "L/min", "status": "inactive"},
        ]
        batch = sim.generate_batch(active_sensors=sensors, mode="normal", iterations=1)
        assert len(batch) == 1
        assert batch[0].sensor_id == 1

    def test_generate_batch_leak_mode_requires_pipe_id(self):
        """Calling generate_batch in leak mode without leak_pipe_id raises ValueError."""
        sim = SensorSimulator()
        sensors = [{"sensor_id": 1, "sensor_name": "S1", "sensor_type": "flow_meter", "pipe_id": 1, "unit": "L/min", "status": "active"}]
        with pytest.raises(ValueError, match="leak_pipe_id is required"):
            sim.generate_batch(sensors, mode="leak", leak_pipe_id=None)


# ═════════════════════════════════════════════════════════════════════════════
#  2. DATABASE PERSISTENCE & QUERY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

class TestSimulatorDatabaseIntegration:
    def test_readings_persistence_and_is_simulated_flag(self, tmp_db, simulated_network):
        """Readings stored in sensor_readings have is_simulated == 1 and correct fields."""
        s1 = simulated_network["sensors"]["s1_flow_active"]

        reading_id = tmp_db.record_sensor_reading(
            sensor_id=s1,
            reading_value=351.42,
            recorded_at="2026-11-01T10:00:00Z",
            reading_type="flow_meter",
            unit="L/min",
            is_simulated=True,
        )
        assert reading_id > 0

        readings = tmp_db.get_readings_for_sensor(s1)
        assert len(readings) == 1
        r = readings[0]
        assert r["sensor_id"] == s1
        assert r["reading_value"] == pytest.approx(351.42)
        assert r["is_simulated"] == 1
        assert r["unit"] == "L/min"

    def test_get_active_sensors(self, tmp_db, simulated_network):
        """get_active_sensors returns only active sensors with joined metadata."""
        active = tmp_db.get_active_sensors()
        # In fixture: 4 active sensors (s1, s2, s4, s5) and 1 inactive (s3)
        assert len(active) == 4
        sensor_ids = [s["sensor_id"] for s in active]
        assert simulated_network["sensors"]["s3_flow_inactive"] not in sensor_ids

        # Filter by pipe_1
        active_p1 = tmp_db.get_active_sensors(pipe_id=simulated_network["pipe_1"])
        assert len(active_p1) == 2  # s1 and s2

    def test_count_active_sensors(self, tmp_db, simulated_network):
        assert tmp_db.count_active_sensors() == 4

    def test_get_latest_sensor_readings_joined(self, tmp_db, simulated_network):
        """get_latest_sensor_readings joins sensor, pipe, and endpoint nodes."""
        s1 = simulated_network["sensors"]["s1_flow_active"]
        s2 = simulated_network["sensors"]["s2_press_active"]

        tmp_db.record_sensor_reading(s1, 350.0, recorded_at="2026-11-01T10:00:00Z")
        tmp_db.record_sensor_reading(s2, 52.0, recorded_at="2026-11-01T10:01:00Z")

        latest = tmp_db.get_latest_sensor_readings(limit=10)
        assert len(latest) == 2
        # Newest first
        assert latest[0]["sensor_id"] == s2
        assert latest[0]["pipe_name"] == "Main-Trunk-1"
        assert latest[0]["from_node_name"] == "Tank Alpha"
        assert latest[0]["to_node_name"] == "Junction Beta"

    def test_get_latest_reading_by_type(self, tmp_db, simulated_network):
        s1 = simulated_network["sensors"]["s1_flow_active"]
        s2 = simulated_network["sensors"]["s2_press_active"]

        tmp_db.record_sensor_reading(s1, 354.2, recorded_at="2026-11-01T10:00:00Z")
        tmp_db.record_sensor_reading(s2, 51.8, recorded_at="2026-11-01T10:05:00Z")

        latest_flow = tmp_db.get_latest_reading_by_type("flow_meter")
        assert latest_flow is not None
        assert latest_flow["sensor_id"] == s1
        assert latest_flow["reading_value"] == pytest.approx(354.2)

        latest_press = tmp_db.get_latest_reading_by_type("pressure_sensor")
        assert latest_press is not None
        assert latest_press["sensor_id"] == s2
        assert latest_press["reading_value"] == pytest.approx(51.8)

    def test_clear_sensor_readings(self, tmp_db, simulated_network):
        s1 = simulated_network["sensors"]["s1_flow_active"]
        tmp_db.record_sensor_reading(s1, 350.0)
        tmp_db.record_sensor_reading(s1, 352.0)
        assert tmp_db.count_sensor_readings() == 2

        deleted = tmp_db.clear_sensor_readings()
        assert deleted == 2
        assert tmp_db.count_sensor_readings() == 0


# ═════════════════════════════════════════════════════════════════════════════
#  3. HTTP ROUTES & INTEGRATION TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestSimulatorRoutes:
    def test_simulator_page_loads_cleanly(self, app_client, tmp_db, simulated_network):
        """GET /sensor-simulator returns 200 with notice, controls, and summary cards."""
        r = app_client.get("/sensor-simulator")
        assert r.status_code == 200
        assert b"Sensor Simulator" in r.data
        assert b"Synthetic Demonstration Data Notice" in r.data
        assert b"All readings generated by this simulator are synthetic demonstration data" in r.data
        assert b"Total Readings Generated" in r.data
        assert b"Active Sensors" in r.data
        assert b"Generate Reading" in r.data
        assert b"Generate 10 Readings" in r.data

    def test_generate_single_reading_normal_mode(self, app_client, tmp_db, simulated_network):
        """POST /sensor-simulator/generate with count=1 creates 1 reading per active sensor."""
        r = app_client.post(
            "/sensor-simulator/generate",
            data={
                "mode": "normal",
                "count": "1",
                "seed": "42",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        # 4 active sensors -> 4 readings
        assert tmp_db.count_sensor_readings() == 4
        assert b"Generated 4 synthetic telemetry reading(s)" in r.data
        assert b"Synthetic" in r.data

    def test_generate_10_readings_batch(self, app_client, tmp_db, simulated_network):
        """POST /sensor-simulator/generate with count=10 creates 10 readings per active sensor."""
        r = app_client.post(
            "/sensor-simulator/generate",
            data={
                "mode": "normal",
                "count": "10",
                "seed": "100",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        # 4 active sensors * 10 iterations = 40 readings
        assert tmp_db.count_sensor_readings() == 40
        assert b"Generated 40 synthetic telemetry reading(s)" in r.data

    def test_generate_leak_scenario_success(self, app_client, tmp_db, simulated_network):
        """POST /sensor-simulator/generate with mode=leak triggers anomaly on target pipe."""
        p1 = simulated_network["pipe_1"]
        r = app_client.post(
            "/sensor-simulator/generate",
            data={
                "mode": "leak",
                "pipe_id": p1,
                "count": "1",
                "seed": "55",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        assert b"simulated leak anomaly on pipe" in r.data.lower()
        assert tmp_db.count_sensor_readings() == 4

        # Verify readings for Pipe 1 sensors show surge/drop
        s1 = simulated_network["sensors"]["s1_flow_active"]
        s1_reading = tmp_db.get_readings_for_sensor(s1)[0]["reading_value"]
        assert s1_reading > 520.0  # Leak surge

    def test_leak_mode_missing_pipe_flashes_error(self, app_client, tmp_db, simulated_network):
        """POST /sensor-simulator/generate with mode=leak without pipe_id flashes error."""
        r = app_client.post(
            "/sensor-simulator/generate",
            data={
                "mode": "leak",
                "pipe_id": "",
                "count": "1",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        assert b"select a monitored pipe" in r.data.lower()
        assert tmp_db.count_sensor_readings() == 0

    def test_leak_mode_nonexistent_pipe_flashes_error(self, app_client, tmp_db, simulated_network):
        """POST /sensor-simulator/generate with non-existent pipe_id flashes error."""
        r = app_client.post(
            "/sensor-simulator/generate",
            data={
                "mode": "leak",
                "pipe_id": "9999",
                "count": "1",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        assert b"does not exist" in r.data.lower()
        assert tmp_db.count_sensor_readings() == 0

    def test_leak_mode_pipe_without_active_sensors_flashes_error(self, app_client, tmp_db):
        """POST /sensor-simulator/generate on a pipe that has no active sensors flashes error."""
        id_a = tmp_db.create_node("T1", "tank", "Z1", "active")
        id_b = tmp_db.create_node("J1", "junction", "Z2", "active")
        bare_pipe = tmp_db.create_pipe("BarePipe", id_a, id_b, "100", "200", "1.0", "active")

        # Create one active sensor on a different pipe so network has active sensors
        other_pipe = tmp_db.create_pipe("OtherPipe", id_b, id_a, "100", "200", "1.0", "active")
        tmp_db.create_sensor("ActiveSensor", other_pipe, "flow_meter", "L/min", "active")

        r = app_client.post(
            "/sensor-simulator/generate",
            data={
                "mode": "leak",
                "pipe_id": bare_pipe,
                "count": "1",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        assert b"no active sensors installed" in r.data.lower()
        assert tmp_db.count_sensor_readings() == 0

    def test_generate_when_no_active_sensors_flashes_error(self, app_client, tmp_db):
        """When the entire network has 0 active sensors, error is flashed."""
        r = app_client.post(
            "/sensor-simulator/generate",
            data={
                "mode": "normal",
                "count": "1",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        assert b"no active sensors found" in r.data.lower()
        assert tmp_db.count_sensor_readings() == 0

    def test_generate_get_not_allowed(self, app_client):
        """GET /sensor-simulator/generate returns 405 Method Not Allowed."""
        r = app_client.get("/sensor-simulator/generate")
        assert r.status_code == 405

    def test_clear_readings_via_post(self, app_client, tmp_db, simulated_network):
        """POST /sensor-simulator/clear removes all simulated readings."""
        s1 = simulated_network["sensors"]["s1_flow_active"]
        tmp_db.record_sensor_reading(s1, 350.0)
        assert tmp_db.count_sensor_readings() == 1

        r = app_client.post("/sensor-simulator/clear", follow_redirects=True)
        assert r.status_code == 200
        assert b"Cleared 1 simulated sensor reading(s)" in r.data
        assert tmp_db.count_sensor_readings() == 0

    def test_clear_get_not_allowed(self, app_client):
        """GET /sensor-simulator/clear returns 405 Method Not Allowed."""
        r = app_client.get("/sensor-simulator/clear")
        assert r.status_code == 405

    def test_sidebar_sensors_simulator_link_present(self, app_client):
        """Verify navigation sidebar links to /sensor-simulator from all platform pages."""
        for path in ["/", "/network", "/pipes", "/sensors", "/sensor-simulator", "/graph", "/route"]:
            r = app_client.get(path)
            assert r.status_code == 200
            assert b"/sensor-simulator" in r.data
            assert b"Sensor Simulator" in r.data

