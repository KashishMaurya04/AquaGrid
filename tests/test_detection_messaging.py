"""
tests/test_detection_messaging.py – Tests for Leak Detection Result Messaging Logic.

Verifies accurate page-level messaging across all detection outcomes:
1. All Insufficient Data: "Detection completed. Insufficient telemetry is available to determine network health." (warning)
2. All Normal: "Detection completed: All active monitored pipes are operating within expected statistical baseline bounds." (success)
3. Mixed (Normal + Insufficient): Communicates both counts accurately (warning).
4. Suspected Leak: Shows appropriate warning banner (not success).
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import pytest

from detection.engine import (
    PipeDiagnostic,
    SensorAnalysisResult,
    build_detection_result_message,
)
import database as db_module
import app as app_module


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    """Isolate database tests with a fresh SQLite DB in a temporary directory."""
    test_db = str(tmp_path / "test_messaging_aquagrid.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    monkeypatch.setattr(app_module, "db", db_module)
    db_module.init_db()
    return db_module


@pytest.fixture()
def app_client(tmp_db):
    """Flask test client wired to isolated database."""
    app_module.app.config["TESTING"] = True
    app_module.app.config["WTF_CSRF_ENABLED"] = False
    app_module.app.config["SECRET_KEY"] = "test-messaging-key"

    with app_module.app.test_client() as client:
        yield client


# ==============================================================================
# Unit Tests for build_detection_result_message
# ==============================================================================

class TestBuildDetectionResultMessageUnit:
    """Unit tests for pure message generation logic across outcomes."""

    def test_all_insufficient_data(self):
        """When every monitored pipe has insufficient data, show explicit notice."""
        diags = [
            PipeDiagnostic(pipe_id=1, pipe_name="Trunk 1", from_node_name="A", to_node_name="B", status="insufficient_data"),
            PipeDiagnostic(pipe_id=2, pipe_name="Trunk 2", from_node_name="B", to_node_name="C", status="insufficient_data"),
            PipeDiagnostic(pipe_id=3, pipe_name="Trunk 4", from_node_name="C", to_node_name="D", status="insufficient_data"),
        ]
        msg, category = build_detection_result_message(diags)
        assert msg == "Detection completed. Insufficient telemetry is available to determine network health."
        assert category == "warning"

    def test_all_normal_data(self):
        """When all monitored pipes are normal, return success notice."""
        diags = [
            PipeDiagnostic(pipe_id=1, pipe_name="Trunk 1", from_node_name="A", to_node_name="B", status="normal"),
            PipeDiagnostic(pipe_id=2, pipe_name="Trunk 2", from_node_name="B", to_node_name="C", status="normal"),
        ]
        msg, category = build_detection_result_message(diags)
        assert msg == "Detection completed: All active monitored pipes are operating within expected statistical baseline bounds."
        assert category == "success"

    def test_mixed_normal_and_insufficient_data(self):
        """When some pipes are normal and others have insufficient data, communicate both counts accurately."""
        diags = [
            PipeDiagnostic(pipe_id=1, pipe_name="Trunk 1", from_node_name="A", to_node_name="B", status="normal"),
            PipeDiagnostic(pipe_id=2, pipe_name="Trunk 2", from_node_name="B", to_node_name="C", status="normal"),
            PipeDiagnostic(pipe_id=3, pipe_name="Trunk 4", from_node_name="C", to_node_name="D", status="insufficient_data"),
        ]
        msg, category = build_detection_result_message(diags)
        assert "2 pipes operating within normal baseline bounds" in msg
        assert "1 pipe has insufficient telemetry to determine health" in msg
        assert category == "warning"

    def test_single_normal_and_multiple_insufficient_singular_plural(self):
        """Test singular/plural handling when 1 pipe is normal and 2 are insufficient."""
        diags = [
            PipeDiagnostic(pipe_id=1, pipe_name="Trunk 1", from_node_name="A", to_node_name="B", status="normal"),
            PipeDiagnostic(pipe_id=2, pipe_name="Trunk 2", from_node_name="B", to_node_name="C", status="insufficient_data"),
            PipeDiagnostic(pipe_id=3, pipe_name="Trunk 4", from_node_name="C", to_node_name="D", status="insufficient_data"),
        ]
        msg, category = build_detection_result_message(diags)
        assert "1 pipe operating within normal baseline bounds" in msg
        assert "2 pipes have insufficient telemetry to determine health" in msg
        assert category == "warning"

    def test_suspected_leak_shows_warning(self):
        """When any pipe has a suspected leak, show an appropriate warning rather than a success banner."""
        diags = [
            PipeDiagnostic(pipe_id=1, pipe_name="Trunk 1", from_node_name="A", to_node_name="B", status="suspected_leak", severity="critical"),
            PipeDiagnostic(pipe_id=2, pipe_name="Trunk 2", from_node_name="B", to_node_name="C", status="normal"),
        ]
        msg, category = build_detection_result_message(diags)
        assert "Detection warning: 1 suspected leak anomaly event detected across monitored pipes." in msg
        assert category == "warning"

    def test_suspected_leak_with_insufficient_data(self):
        """When a suspected leak occurs and other pipes have insufficient data, communicate both."""
        diags = [
            PipeDiagnostic(pipe_id=1, pipe_name="Trunk 1", from_node_name="A", to_node_name="B", status="suspected_leak", severity="high"),
            PipeDiagnostic(pipe_id=2, pipe_name="Trunk 2", from_node_name="B", to_node_name="C", status="insufficient_data"),
        ]
        msg, category = build_detection_result_message(diags)
        assert "1 suspected leak anomaly event detected" in msg
        assert "1 pipe pending telemetry" in msg
        assert category == "warning"

    def test_empty_diagnostics_list(self):
        """When no pipes are monitored, return graceful warning."""
        msg, category = build_detection_result_message([])
        assert "No monitored pipes found in the network" in msg
        assert category == "warning"


# ==============================================================================
# Integration / Route Tests via POST /leak-detection/run
# ==============================================================================

class TestLeakDetectionMessagingRoutes:
    """End-to-end integration tests verifying flash banner output in the UI."""

    def test_route_flash_all_insufficient_data(self, tmp_db, app_client):
        """
        Verify that running detection on pipes without sufficient telemetry
        displays the exact 'Detection completed. Insufficient telemetry is available' warning banner.
        """
        n1 = tmp_db.create_node("Node A", "tank", "Zone A", "active")
        n2 = tmp_db.create_node("Node B", "junction", "Zone B", "active")
        p1 = tmp_db.create_pipe("Trunk-Alpha", n1, n2, "200.0", "500.0", "1.0", "active")
        tmp_db.create_sensor("FM-Alpha", p1, "flow_meter", "L/min", "active")
        # No readings inserted

        res = app_client.post("/leak-detection/run", follow_redirects=True)
        assert res.status_code == 200
        html = res.data.decode("utf-8")

        # Must display the required insufficient telemetry message
        assert "Detection completed. Insufficient telemetry is available to determine network health." in html
        # Must NOT falsely claim operating within normal baseline bounds
        assert "operating within expected statistical baseline bounds" not in html
        # Must render with warning styling
        assert "flash flash-warning" in html

    def test_route_flash_all_normal_data(self, tmp_db, app_client):
        """
        Verify that running detection when all pipes have sufficient verified normal telemetry
        displays the success banner.
        """
        n1 = tmp_db.create_node("Node 1", "tank", "Zone 1", "active")
        n2 = tmp_db.create_node("Node 2", "junction", "Zone 2", "active")
        p1 = tmp_db.create_pipe("Main-Pipe-1", n1, n2, "200.0", "500.0", "1.0", "active")
        fm1 = tmp_db.create_sensor("FM-1", p1, "flow_meter", "L/min", "active")

        base_time = datetime(2026, 11, 10, 10, 0, 0, tzinfo=timezone.utc)
        for i in range(10):
            t = (base_time + timedelta(minutes=i)).isoformat()
            tmp_db.record_sensor_reading(
                sensor_id=fm1,
                reading_value=350.0 + (i % 2) * 1.5,
                recorded_at=t,
                scenario="normal",
            )

        res = app_client.post("/leak-detection/run", follow_redirects=True)
        assert res.status_code == 200
        html = res.data.decode("utf-8")

        # Must display the success message
        assert "Detection completed: All active monitored pipes are operating within expected statistical baseline bounds." in html
        assert "flash flash-success" in html

    def test_route_flash_mixed_normal_and_insufficient_data(self, tmp_db, app_client):
        """
        Verify that running detection when some pipes are normal and others have insufficient data
        accurately communicates both counts.
        """
        n1 = tmp_db.create_node("North Reservoir", "tank", "North Zone", "active")
        n2 = tmp_db.create_node("Central Hub", "junction", "Central Zone", "active")
        n3 = tmp_db.create_node("South Valve", "valve", "South Zone", "active")

        # Pipe 1: Has 10 verified normal readings
        p1 = tmp_db.create_pipe("Pipe-Normal", n1, n2, "200.0", "500.0", "1.0", "active")
        fm1 = tmp_db.create_sensor("FM-Normal", p1, "flow_meter", "L/min", "active")
        base_time = datetime(2026, 11, 10, 10, 0, 0, tzinfo=timezone.utc)
        for i in range(10):
            t = (base_time + timedelta(minutes=i)).isoformat()
            tmp_db.record_sensor_reading(
                sensor_id=fm1,
                reading_value=350.0 + (i % 2) * 2.0,
                recorded_at=t,
                scenario="normal",
            )

        # Pipe 2: Has active sensor but 0 readings (insufficient data)
        p2 = tmp_db.create_pipe("Pipe-Pending", n2, n3, "150.0", "300.0", "1.0", "active")
        tmp_db.create_sensor("FM-Pending", p2, "flow_meter", "L/min", "active")

        res = app_client.post("/leak-detection/run", follow_redirects=True)
        assert res.status_code == 200
        html = res.data.decode("utf-8")

        # Must communicate both counts accurately
        assert "1 pipe operating within normal baseline bounds" in html
        assert "1 pipe has insufficient telemetry to determine health" in html
        # Must NOT claim that all pipes are within normal bounds
        assert "All active monitored pipes are operating within expected statistical baseline bounds" not in html
        # Must be styled as warning
        assert "flash flash-warning" in html

    def test_route_flash_suspected_leak_warning(self, tmp_db, app_client):
        """
        Verify that running detection when an anomaly exists shows a warning banner rather than a success banner.
        """
        n1 = tmp_db.create_node("Node X", "tank", "Zone X", "active")
        n2 = tmp_db.create_node("Node Y", "junction", "Zone Y", "active")
        p1 = tmp_db.create_pipe("Pipe-Leak", n1, n2, "200.0", "500.0", "1.0", "active")
        fm1 = tmp_db.create_sensor("FM-Leak", p1, "flow_meter", "L/min", "active")

        base_time = datetime(2026, 11, 10, 10, 0, 0, tzinfo=timezone.utc)
        # 10 baseline
        for i in range(10):
            t = (base_time + timedelta(minutes=i)).isoformat()
            tmp_db.record_sensor_reading(sensor_id=fm1, reading_value=350.0, recorded_at=t, scenario="normal")
        # 5 severe surge
        for i in range(5):
            t = (base_time + timedelta(minutes=10 + i)).isoformat()
            tmp_db.record_sensor_reading(sensor_id=fm1, reading_value=600.0, recorded_at=t, scenario="leak")

        res = app_client.post("/leak-detection/run", follow_redirects=True)
        assert res.status_code == 200
        html = res.data.decode("utf-8")

        # Must show warning banner
        assert "Detection warning: 1 suspected leak anomaly event detected across monitored pipes." in html
        # Must NOT show success banner
        assert "flash flash-success" not in html
        assert "flash flash-warning" in html

