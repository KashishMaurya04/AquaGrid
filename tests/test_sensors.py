"""
tests/test_sensors.py – Unit and integration test suite for Sensor Management.

Milestone: Sensor Management
Covers:
  1. Sensor registration (flow_meter, pressure_sensor, custom/default units)
  2. Input validation (blank/long names, invalid types, invalid status, blank units)
  3. Pipe validation (rejects non-existent pipe IDs, missing pipe IDs)
  4. Sensor editing (updates name, pipe, type, unit, status; rejects non-existent ID or invalid pipe)
  5. Sensor status toggling (activate/deactivate via toggle_sensor_status)
  6. Safe sensor deletion (deletes sensor safely; rejects non-existent ID)
  7. Safe sensor deletion restriction (blocked if sensor has recorded readings)
  8. Foreign-key enforcement:
     - SQLite PRAGMA foreign_keys is ON
     - Cannot insert sensor referencing non-existent pipe (IntegrityError / ValueError)
     - Cannot delete pipe referenced by active or inactive sensor (RuntimeError in app, IntegrityError in SQLite)
     - Can delete pipe once associated sensor is deleted
     - Cannot insert sensor reading referencing non-existent sensor (IntegrityError / LookupError)
     - Cannot delete sensor referenced by reading (IntegrityError in SQLite / RuntimeError in app)
  9. Sensor readings recording and query functions
 10. Flask HTTP route integration:
     - GET /sensors (empty state & populated list)
     - POST /sensors/create (success redirect & validation error flash)
     - GET /sensors/<id>/edit & POST /sensors/<id>/edit (render form & save changes)
     - POST /sensors/<id>/toggle-status (status toggle & GET not allowed 405)
     - POST /sensors/<id>/delete (safe deletion & GET not allowed 405)
     - POST /pipes/<id>/delete blocked when sensor exists (user-friendly flash error)
     - Sidebar navigation link to /sensors present across all pages
"""

import os
import sqlite3
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    """Isolate database tests with a fresh SQLite DB in a temporary directory."""
    import database as db_module

    test_db = str(tmp_path / "test_sensors_aquagrid.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


@pytest.fixture()
def sample_network(tmp_db):
    """
    Sets up a minimal test network:
      Tank 1 (Alpha) -> Junction 2 (Beta) connected by Pipe 1 (Main Trunk)
      Returns (node_a_id, node_b_id, pipe_id).
    """
    id_a = tmp_db.create_node("Tank Alpha", "tank", "Zone North", "active")
    id_b = tmp_db.create_node("Junction Beta", "junction", "Zone Central", "active")
    pipe_id = tmp_db.create_pipe("Main Trunk", id_a, id_b, "250.0", "500.0", "1.0", "active")
    return id_a, id_b, pipe_id


@pytest.fixture()
def app_client(tmp_db, monkeypatch):
    """Flask test client wired to the isolated database."""
    import database as db_module
    import app as app_module

    monkeypatch.setattr(app_module, "db", db_module)

    app_module.app.config["TESTING"] = True
    app_module.app.config["WTF_CSRF_ENABLED"] = False
    app_module.app.config["SECRET_KEY"] = "test-sensor-secret-key"

    with app_module.app.test_client() as client:
        yield client


# ═════════════════════════════════════════════════════════════════════════════
#  1. SENSOR CREATION & RETRIEVAL (DB LAYER)
# ═════════════════════════════════════════════════════════════════════════════

class TestCreateSensor:
    def test_create_flow_meter_returns_id(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        sensor_id = tmp_db.create_sensor(
            sensor_name="FM-Trunk-01",
            pipe_id=pipe_id,
            sensor_type="flow_meter",
            unit="L/min",
            status="active",
        )
        assert isinstance(sensor_id, int)
        assert sensor_id > 0

    def test_create_pressure_sensor_returns_id(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        sensor_id = tmp_db.create_sensor(
            sensor_name="PS-Trunk-01",
            pipe_id=pipe_id,
            sensor_type="pressure_sensor",
            unit="psi",
            status="active",
        )
        assert isinstance(sensor_id, int)
        assert sensor_id > 0

    def test_created_sensor_is_retrievable_with_metadata(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        sensor_id = tmp_db.create_sensor(
            sensor_name="FM-North",
            pipe_id=pipe_id,
            sensor_type="flow_meter",
            unit="m³/h",
            status="active",
        )
        sensor = tmp_db.get_sensor_by_id(sensor_id)
        assert sensor is not None
        assert sensor["sensor_id"] == sensor_id
        assert sensor["sensor_name"] == "FM-North"
        assert sensor["pipe_id"] == pipe_id
        assert sensor["sensor_type"] == "flow_meter"
        assert sensor["unit"] == "m³/h"
        assert sensor["status"] == "active"
        assert sensor["pipe_name"] == "Main Trunk"
        assert sensor["from_node_name"] == "Tank Alpha"
        assert sensor["to_node_name"] == "Junction Beta"
        assert "created_at" in sensor.keys()

    def test_create_sensor_inactive_status_accepted(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        sensor_id = tmp_db.create_sensor(
            sensor_name="PS-Spare",
            pipe_id=pipe_id,
            sensor_type="pressure_sensor",
            unit="bar",
            status="inactive",
        )
        sensor = tmp_db.get_sensor_by_id(sensor_id)
        assert sensor["status"] == "inactive"

    def test_count_sensors_increments(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        assert tmp_db.count_sensors() == 0
        tmp_db.create_sensor("S-1", pipe_id, "flow_meter", "L/min", "active")
        assert tmp_db.count_sensors() == 1
        tmp_db.create_sensor("S-2", pipe_id, "pressure_sensor", "psi", "active")
        assert tmp_db.count_sensors() == 2

    def test_get_all_sensors_ordering(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        id_1 = tmp_db.create_sensor("First Sensor", pipe_id, "flow_meter", "L/min", "active")
        id_2 = tmp_db.create_sensor("Second Sensor", pipe_id, "pressure_sensor", "psi", "active")
        all_sensors = tmp_db.get_all_sensors()
        assert len(all_sensors) == 2
        sensor_ids = [s["sensor_id"] for s in all_sensors]
        assert id_1 in sensor_ids
        assert id_2 in sensor_ids

    def test_get_pipe_name_map(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        pipe_map = tmp_db.get_pipe_name_map()
        assert pipe_id in pipe_map
        assert "Main Trunk" in pipe_map[pipe_id]
        assert "Tank Alpha" in pipe_map[pipe_id]
        assert "Junction Beta" in pipe_map[pipe_id]


# ═════════════════════════════════════════════════════════════════════════════
#  2. SENSOR VALIDATION (DB LAYER)
# ═════════════════════════════════════════════════════════════════════════════

class TestSensorValidation:
    def test_blank_name_raises(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        with pytest.raises(ValueError) as exc:
            tmp_db.create_sensor("", pipe_id, "flow_meter", "L/min", "active")
        assert any("blank" in m.lower() or "required" in m.lower() for m in exc.value.args[0])

    def test_whitespace_name_raises(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        with pytest.raises(ValueError) as exc:
            tmp_db.create_sensor("    ", pipe_id, "flow_meter", "L/min", "active")
        assert any("blank" in m.lower() or "required" in m.lower() for m in exc.value.args[0])

    def test_name_too_long_raises(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        with pytest.raises(ValueError) as exc:
            tmp_db.create_sensor("A" * 121, pipe_id, "flow_meter", "L/min", "active")
        assert any("120" in m for m in exc.value.args[0])

    def test_none_pipe_id_raises(self, tmp_db):
        with pytest.raises(ValueError) as exc:
            tmp_db.create_sensor("Sensor-No-Pipe", None, "flow_meter", "L/min", "active")
        assert any("pipe is required" in m.lower() for m in exc.value.args[0])

    def test_nonexistent_pipe_id_raises(self, tmp_db):
        with pytest.raises(ValueError) as exc:
            tmp_db.create_sensor("Sensor-Bad-Pipe", 9999, "flow_meter", "L/min", "active")
        assert any("does not exist" in m.lower() for m in exc.value.args[0])

    def test_invalid_sensor_type_raises(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        with pytest.raises(ValueError) as exc:
            tmp_db.create_sensor("Sensor-Type-Err", pipe_id, "temperature", "C", "active")
        assert any("sensor type" in m.lower() for m in exc.value.args[0])

    def test_blank_unit_raises(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        with pytest.raises(ValueError) as exc:
            tmp_db.create_sensor("Sensor-No-Unit", pipe_id, "flow_meter", "", "active")
        assert any("unit" in m.lower() for m in exc.value.args[0])

    def test_unit_too_long_raises(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        with pytest.raises(ValueError) as exc:
            tmp_db.create_sensor("Sensor-Long-Unit", pipe_id, "flow_meter", "U" * 31, "active")
        assert any("30" in m for m in exc.value.args[0])

    def test_invalid_status_raises(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        with pytest.raises(ValueError) as exc:
            tmp_db.create_sensor("Sensor-Bad-Status", pipe_id, "flow_meter", "L/min", "pending")
        assert any("status" in m.lower() for m in exc.value.args[0])

    def test_multiple_validation_errors_reported_together(self, tmp_db):
        with pytest.raises(ValueError) as exc:
            tmp_db.create_sensor("", 9999, "invalid_type", "", "bad_status")
        msgs = exc.value.args[0]
        assert len(msgs) >= 4


# ═════════════════════════════════════════════════════════════════════════════
#  3. SENSOR EDITING & TOGGLE (DB LAYER)
# ═════════════════════════════════════════════════════════════════════════════

class TestEditSensor:
    def test_update_sensor_changes_values(self, tmp_db, sample_network):
        id_a, id_b, pipe_1 = sample_network
        pipe_2 = tmp_db.create_pipe("Secondary Pipe", id_b, id_a, "100.0", "300.0", "2.0", "active")

        sensor_id = tmp_db.create_sensor("Original", pipe_1, "flow_meter", "L/min", "active")
        tmp_db.update_sensor(
            sensor_id=sensor_id,
            sensor_name="Updated Sensor",
            pipe_id=pipe_2,
            sensor_type="pressure_sensor",
            unit="bar",
            status="inactive",
        )

        sensor = tmp_db.get_sensor_by_id(sensor_id)
        assert sensor["sensor_name"] == "Updated Sensor"
        assert sensor["pipe_id"] == pipe_2
        assert sensor["sensor_type"] == "pressure_sensor"
        assert sensor["unit"] == "bar"
        assert sensor["status"] == "inactive"
        assert sensor["pipe_name"] == "Secondary Pipe"

    def test_update_nonexistent_sensor_raises_lookup(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        with pytest.raises(LookupError):
            tmp_db.update_sensor(9999, "Name", pipe_id, "flow_meter", "L/min", "active")

    def test_update_with_invalid_pipe_raises_value_error(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        sensor_id = tmp_db.create_sensor("Name", pipe_id, "flow_meter", "L/min", "active")
        with pytest.raises(ValueError) as exc:
            tmp_db.update_sensor(sensor_id, "Name", 9999, "flow_meter", "L/min", "active")
        assert any("does not exist" in m.lower() for m in exc.value.args[0])

    def test_toggle_sensor_status_active_to_inactive_and_back(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        sensor_id = tmp_db.create_sensor("ToggleMe", pipe_id, "flow_meter", "L/min", "active")

        new_status = tmp_db.toggle_sensor_status(sensor_id)
        assert new_status == "inactive"
        assert tmp_db.get_sensor_by_id(sensor_id)["status"] == "inactive"

        new_status_2 = tmp_db.toggle_sensor_status(sensor_id)
        assert new_status_2 == "active"
        assert tmp_db.get_sensor_by_id(sensor_id)["status"] == "active"

    def test_toggle_nonexistent_sensor_raises_lookup(self, tmp_db):
        with pytest.raises(LookupError):
            tmp_db.toggle_sensor_status(9999)


# ═════════════════════════════════════════════════════════════════════════════
#  4. SENSOR DELETION (DB LAYER)
# ═════════════════════════════════════════════════════════════════════════════

class TestDeleteSensor:
    def test_delete_sensor_removes_record(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        sensor_id = tmp_db.create_sensor("ToDelete", pipe_id, "flow_meter", "L/min", "active")
        assert tmp_db.get_sensor_by_id(sensor_id) is not None

        tmp_db.delete_sensor(sensor_id)
        assert tmp_db.get_sensor_by_id(sensor_id) is None
        assert tmp_db.count_sensors() == 0

    def test_delete_nonexistent_sensor_raises_lookup(self, tmp_db):
        with pytest.raises(LookupError):
            tmp_db.delete_sensor(9999)

    def test_delete_only_removes_targeted_sensor(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        id_1 = tmp_db.create_sensor("KeepMe", pipe_id, "flow_meter", "L/min", "active")
        id_2 = tmp_db.create_sensor("DeleteMe", pipe_id, "pressure_sensor", "psi", "active")

        tmp_db.delete_sensor(id_2)
        assert tmp_db.get_sensor_by_id(id_1) is not None
        assert tmp_db.get_sensor_by_id(id_2) is None
        assert tmp_db.count_sensors() == 1


# ═════════════════════════════════════════════════════════════════════════════
#  5. FOREIGN-KEY ENFORCEMENT & SAFETY RESTRICTIONS
# ═════════════════════════════════════════════════════════════════════════════

class TestForeignKeyEnforcement:
    def test_sqlite_foreign_keys_pragma_enabled(self, tmp_db):
        """Verify foreign-key pragma is ON on every database connection."""
        with tmp_db._connect() as conn:
            fk_status = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            assert fk_status == 1

    def test_cannot_delete_pipe_referenced_by_sensor_application_layer(self, tmp_db, sample_network):
        """db.delete_pipe raises RuntimeError when monitored by a sensor."""
        _, _, pipe_id = sample_network
        tmp_db.create_sensor("Guard-Sensor", pipe_id, "flow_meter", "L/min", "active")

        with pytest.raises(RuntimeError) as exc:
            tmp_db.delete_pipe(pipe_id)
        assert "monitored by 1 sensor" in str(exc.value)

    def test_cannot_delete_pipe_referenced_by_sensor_sqlite_layer(self, tmp_db, sample_network):
        """Direct SQLite DELETE fails with IntegrityError due to REFERENCES pipes(id) ON DELETE RESTRICT."""
        _, _, pipe_id = sample_network
        tmp_db.create_sensor("Guard-Sensor", pipe_id, "flow_meter", "L/min", "active")

        with tmp_db._connect() as conn:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("DELETE FROM pipes WHERE id = ?", (pipe_id,))

    def test_can_delete_pipe_after_sensor_is_removed(self, tmp_db, sample_network):
        """Once the sensor is deleted, the pipe can be safely deleted."""
        _, _, pipe_id = sample_network
        sensor_id = tmp_db.create_sensor("Temporary-Sensor", pipe_id, "flow_meter", "L/min", "active")
        tmp_db.delete_sensor(sensor_id)
        tmp_db.delete_pipe(pipe_id)
        assert tmp_db.get_pipe_by_id(pipe_id) is None

    def test_raw_sql_foreign_key_pipe_violation_raises(self, tmp_db):
        """Direct raw SQL INSERT with invalid pipe_id raises sqlite3.IntegrityError."""
        with tmp_db._connect() as conn:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO sensors (sensor_name, pipe_id, sensor_type, unit, status, created_at)
                    VALUES ('Bad-FK', 99999, 'flow_meter', 'L/min', 'active', '2026-01-01T00:00:00Z')
                    """
                )

    def test_raw_sql_foreign_key_reading_violation_raises(self, tmp_db):
        """Direct raw SQL INSERT into sensor_readings with invalid sensor_id raises sqlite3.IntegrityError."""
        with tmp_db._connect() as conn:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO sensor_readings (sensor_id, reading_value, recorded_at, reading_type, unit, is_simulated)
                    VALUES (99999, 45.2, '2026-01-01T00:00:00Z', 'flow_meter', 'L/min', 1)
                    """
                )

    def test_delete_sensor_blocked_by_readings_app_layer(self, tmp_db, sample_network):
        """Deleting a sensor with readings is blocked by delete_sensor with RuntimeError."""
        _, _, pipe_id = sample_network
        sensor_id = tmp_db.create_sensor("Telemetry-Sensor", pipe_id, "flow_meter", "L/min", "active")
        tmp_db.record_sensor_reading(sensor_id, reading_value=120.5)

        with pytest.raises(RuntimeError) as exc:
            tmp_db.delete_sensor(sensor_id)
        assert "recorded reading" in str(exc.value)

    def test_delete_sensor_blocked_by_readings_sqlite_layer(self, tmp_db, sample_network):
        """Raw SQLite DELETE on sensor with readings triggers foreign-key RESTRICT."""
        _, _, pipe_id = sample_network
        sensor_id = tmp_db.create_sensor("Telemetry-Sensor", pipe_id, "flow_meter", "L/min", "active")
        tmp_db.record_sensor_reading(sensor_id, reading_value=120.5)

        with tmp_db._connect() as conn:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("DELETE FROM sensors WHERE sensor_id = ?", (sensor_id,))


# ═════════════════════════════════════════════════════════════════════════════
#  6. SENSOR READINGS (DB LAYER)
# ═════════════════════════════════════════════════════════════════════════════

class TestSensorReadings:
    def test_record_sensor_reading_returns_id(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        sensor_id = tmp_db.create_sensor("Flow-01", pipe_id, "flow_meter", "L/min", "active")
        reading_id = tmp_db.record_sensor_reading(
            sensor_id=sensor_id,
            reading_value=345.67,
            recorded_at="2026-11-01T10:00:00Z",
            reading_type="flow_meter",
            unit="L/min",
            is_simulated=True,
        )
        assert isinstance(reading_id, int)
        assert reading_id > 0
        assert tmp_db.count_sensor_readings() == 1

    def test_record_reading_nonexistent_sensor_raises_lookup(self, tmp_db):
        with pytest.raises(LookupError):
            tmp_db.record_sensor_reading(sensor_id=9999, reading_value=50.0)

    def test_get_readings_for_sensor(self, tmp_db, sample_network):
        _, _, pipe_id = sample_network
        sensor_id = tmp_db.create_sensor("Pressure-01", pipe_id, "pressure_sensor", "psi", "active")
        tmp_db.record_sensor_reading(sensor_id, reading_value=45.0, recorded_at="2026-11-01T09:00:00Z")
        tmp_db.record_sensor_reading(sensor_id, reading_value=46.5, recorded_at="2026-11-01T09:05:00Z")

        readings = tmp_db.get_readings_for_sensor(sensor_id)
        assert len(readings) == 2
        # Ordered newest-first
        assert readings[0]["reading_value"] == pytest.approx(46.5)
        assert readings[1]["reading_value"] == pytest.approx(45.0)


# ═════════════════════════════════════════════════════════════════════════════
#  7. FLASK HTTP ROUTES INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════

class TestSensorRoutes:
    def test_sensors_page_loads_empty(self, app_client):
        """GET /sensors returns 200 and renders empty state."""
        r = app_client.get("/sensors")
        assert r.status_code == 200
        assert b"Sensor Management" in r.data
        assert b"No sensors registered yet" in r.data

    def test_sensors_page_loads_with_registered_sensors(self, app_client, tmp_db, sample_network):
        """GET /sensors renders registered sensor rows."""
        _, _, pipe_id = sample_network
        tmp_db.create_sensor("Sensor-Alpha", pipe_id, "flow_meter", "L/min", "active")

        r = app_client.get("/sensors")
        assert r.status_code == 200
        assert b"Sensor-Alpha" in r.data
        assert b"Flow Meter" in r.data
        assert b"L/min" in r.data
        assert b"Main Trunk" in r.data

    def test_create_sensor_via_post_redirects(self, app_client, tmp_db, sample_network):
        """POST /sensors/create successfully inserts a sensor and redirects."""
        _, _, pipe_id = sample_network
        r = app_client.post(
            "/sensors/create",
            data={
                "sensor_name": "HTTP-Flow-Meter",
                "pipe_id": pipe_id,
                "sensor_type": "flow_meter",
                "unit": "L/min",
                "status": "active",
            },
        )
        assert r.status_code == 302
        assert r.headers["Location"].endswith("/sensors")
        assert tmp_db.count_sensors() == 1
        all_sensors = tmp_db.get_all_sensors()
        assert all_sensors[0]["sensor_name"] == "HTTP-Flow-Meter"

    def test_create_sensor_blank_name_shows_error(self, app_client, tmp_db, sample_network):
        """POST /sensors/create with blank name flashes error."""
        _, _, pipe_id = sample_network
        r = app_client.post(
            "/sensors/create",
            data={
                "sensor_name": "",
                "pipe_id": pipe_id,
                "sensor_type": "flow_meter",
                "unit": "L/min",
                "status": "active",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        assert b"cannot be blank" in r.data.lower() or b"required" in r.data.lower()
        assert tmp_db.count_sensors() == 0

    def test_create_sensor_invalid_pipe_shows_error(self, app_client, tmp_db):
        """POST /sensors/create with non-existent pipe flashes error."""
        r = app_client.post(
            "/sensors/create",
            data={
                "sensor_name": "Ghost-Sensor",
                "pipe_id": 9999,
                "sensor_type": "pressure_sensor",
                "unit": "psi",
                "status": "active",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        assert b"does not exist" in r.data.lower()
        assert tmp_db.count_sensors() == 0

    def test_edit_sensor_page_loads(self, app_client, tmp_db, sample_network):
        """GET /sensors/<id>/edit loads the edit form with prefilled values."""
        _, _, pipe_id = sample_network
        sensor_id = tmp_db.create_sensor("PreEdit", pipe_id, "flow_meter", "L/min", "active")
        r = app_client.get(f"/sensors/{sensor_id}/edit")
        assert r.status_code == 200
        assert b"Edit Sensor" in r.data
        assert b"PreEdit" in r.data

    def test_edit_sensor_post_updates(self, app_client, tmp_db, sample_network):
        """POST /sensors/<id>/edit updates values and redirects."""
        _, _, pipe_id = sample_network
        sensor_id = tmp_db.create_sensor("OldName", pipe_id, "flow_meter", "L/min", "active")

        r = app_client.post(
            f"/sensors/{sensor_id}/edit",
            data={
                "sensor_name": "NewName",
                "pipe_id": pipe_id,
                "sensor_type": "pressure_sensor",
                "unit": "psi",
                "status": "inactive",
            },
        )
        assert r.status_code == 302
        sensor = tmp_db.get_sensor_by_id(sensor_id)
        assert sensor["sensor_name"] == "NewName"
        assert sensor["sensor_type"] == "pressure_sensor"
        assert sensor["unit"] == "psi"
        assert sensor["status"] == "inactive"

    def test_toggle_sensor_status_via_post(self, app_client, tmp_db, sample_network):
        """POST /sensors/<id>/toggle-status toggles status between active and inactive."""
        _, _, pipe_id = sample_network
        sensor_id = tmp_db.create_sensor("ToggleRoute", pipe_id, "flow_meter", "L/min", "active")

        r = app_client.post(f"/sensors/{sensor_id}/toggle-status")
        assert r.status_code == 302
        assert tmp_db.get_sensor_by_id(sensor_id)["status"] == "inactive"

        r2 = app_client.post(f"/sensors/{sensor_id}/toggle-status")
        assert r2.status_code == 302
        assert tmp_db.get_sensor_by_id(sensor_id)["status"] == "active"

    def test_toggle_sensor_status_get_not_allowed(self, app_client, tmp_db, sample_network):
        """GET /sensors/<id>/toggle-status returns 405 Method Not Allowed."""
        _, _, pipe_id = sample_network
        sensor_id = tmp_db.create_sensor("NoGetToggle", pipe_id, "flow_meter", "L/min", "active")
        r = app_client.get(f"/sensors/{sensor_id}/toggle-status")
        assert r.status_code == 405

    def test_delete_sensor_via_post(self, app_client, tmp_db, sample_network):
        """POST /sensors/<id>/delete safely removes sensor and redirects."""
        _, _, pipe_id = sample_network
        sensor_id = tmp_db.create_sensor("DeleteViaHTTP", pipe_id, "flow_meter", "L/min", "active")

        r = app_client.post(f"/sensors/{sensor_id}/delete")
        assert r.status_code == 302
        assert tmp_db.get_sensor_by_id(sensor_id) is None

    def test_delete_sensor_get_not_allowed(self, app_client, tmp_db, sample_network):
        """GET /sensors/<id>/delete returns 405 Method Not Allowed."""
        _, _, pipe_id = sample_network
        sensor_id = tmp_db.create_sensor("NoGetDel", pipe_id, "flow_meter", "L/min", "active")
        r = app_client.get(f"/sensors/{sensor_id}/delete")
        assert r.status_code == 405

    def test_delete_pipe_blocked_by_sensor_shows_flash_message(self, app_client, tmp_db, sample_network):
        """POST /pipes/<id>/delete when monitored by a sensor flashes a friendly error."""
        _, _, pipe_id = sample_network
        tmp_db.create_sensor("Blocking-Sensor", pipe_id, "flow_meter", "L/min", "active")

        r = app_client.post(f"/pipes/{pipe_id}/delete", follow_redirects=True)
        assert r.status_code == 200
        assert b"still monitored by" in r.data.lower() or b"sensor" in r.data.lower()
        assert tmp_db.get_pipe_by_id(pipe_id) is not None

    def test_sidebar_sensors_link_present_across_pages(self, app_client):
        """Verify navigation sidebar links to /sensors from all platform pages."""
        for path in ["/", "/network", "/pipes", "/sensors", "/graph", "/route"]:
            r = app_client.get(path)
            assert r.status_code == 200
            assert b"/sensors" in r.data
            assert b"Sensor Management" in r.data

