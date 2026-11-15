"""
tests/test_nodes.py – pytest suite for the Water Network feature.

Run with:
    python -m pytest tests/ -v

The tests use a temporary in-memory SQLite database so they never touch the
real instance/aquagrid.db file, and they use Flask's test client for HTTP
round-trips.
"""

import importlib
import os
import sys
import tempfile

import pytest

# ── Ensure the project root is on sys.path ───────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    """
    Point the database module at a fresh temp directory for each test, then
    initialise the schema.  This completely isolates tests from each other.
    """
    import database as db_module

    test_db = str(tmp_path / "test_aquagrid.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


@pytest.fixture()
def app_client(tmp_db, monkeypatch):
    """
    Flask test client wired to the same temp database.
    CSRF validation is disabled in the test config so form tests stay simple.
    """
    import database as db_module
    import app as app_module

    monkeypatch.setattr(app_module, "db", db_module)

    # Disable CSRF for tests
    app_module.app.config["TESTING"] = True
    app_module.app.config["WTF_CSRF_ENABLED"] = False
    app_module.app.config["SECRET_KEY"] = "test-secret"

    with app_module.app.test_client() as client:
        yield client


# ── Database-layer unit tests ─────────────────────────────────────────────────

class TestCreateNode:
    def test_create_returns_id(self, tmp_db):
        node_id = tmp_db.create_node("Tank Alpha", "tank", "Zone 1", "active")
        assert isinstance(node_id, int)
        assert node_id >= 1

    def test_created_node_is_retrievable(self, tmp_db):
        tmp_db.create_node("Junction Beta", "junction", "Zone 2", "inactive")
        nodes = tmp_db.get_all_nodes()
        assert len(nodes) == 1
        assert nodes[0]["name"] == "Junction Beta"
        assert nodes[0]["node_type"] == "junction"
        assert nodes[0]["status"] == "inactive"

    def test_location_is_optional(self, tmp_db):
        node_id = tmp_db.create_node("Valve Gamma", "valve", "", "active")
        node = tmp_db.get_node_by_id(node_id)
        assert node["location"] is None

    def test_count_nodes_increments(self, tmp_db):
        assert tmp_db.count_nodes() == 0
        tmp_db.create_node("T1", "tank", "", "active")
        tmp_db.create_node("J1", "junction", "", "active")
        assert tmp_db.count_nodes() == 2


class TestValidation:
    def test_blank_name_raises(self, tmp_db):
        with pytest.raises(ValueError) as exc_info:
            tmp_db.create_node("", "tank", "", "active")
        msgs = exc_info.value.args[0]
        assert any("Name" in m for m in msgs)

    def test_whitespace_only_name_raises(self, tmp_db):
        with pytest.raises(ValueError):
            tmp_db.create_node("   ", "tank", "", "active")

    def test_invalid_node_type_raises(self, tmp_db):
        with pytest.raises(ValueError) as exc_info:
            tmp_db.create_node("Node X", "pipe", "", "active")
        msgs = exc_info.value.args[0]
        assert any("type" in m.lower() for m in msgs)

    def test_invalid_status_raises(self, tmp_db):
        with pytest.raises(ValueError) as exc_info:
            tmp_db.create_node("Node Y", "tank", "", "broken")
        msgs = exc_info.value.args[0]
        assert any("status" in m.lower() for m in msgs)

    def test_multiple_errors_returned_together(self, tmp_db):
        with pytest.raises(ValueError) as exc_info:
            tmp_db.create_node("", "pipe", "", "broken")
        msgs = exc_info.value.args[0]
        # Should include both name error and type error (and optionally status)
        assert len(msgs) >= 2

    def test_name_too_long_raises(self, tmp_db):
        with pytest.raises(ValueError):
            tmp_db.create_node("x" * 121, "tank", "", "active")

    def test_valid_all_types_accepted(self, tmp_db):
        for t in ("tank", "junction", "valve"):
            node_id = tmp_db.create_node(f"Node {t}", t, "", "active")
            assert node_id >= 1


class TestEditNode:
    def test_update_changes_values(self, tmp_db):
        node_id = tmp_db.create_node("Old Name", "tank", "Zone 1", "active")
        tmp_db.update_node(node_id, "New Name", "valve", "Zone 9", "inactive")
        node = tmp_db.get_node_by_id(node_id)
        assert node["name"] == "New Name"
        assert node["node_type"] == "valve"
        assert node["location"] == "Zone 9"
        assert node["status"] == "inactive"

    def test_update_nonexistent_raises_lookup(self, tmp_db):
        with pytest.raises(LookupError):
            tmp_db.update_node(9999, "X", "tank", "", "active")

    def test_update_with_invalid_type_raises(self, tmp_db):
        node_id = tmp_db.create_node("Node", "tank", "", "active")
        with pytest.raises(ValueError):
            tmp_db.update_node(node_id, "Node", "pipe", "", "active")


class TestDeleteNode:
    def test_delete_removes_node(self, tmp_db):
        node_id = tmp_db.create_node("To Delete", "junction", "", "active")
        tmp_db.delete_node(node_id)
        assert tmp_db.get_node_by_id(node_id) is None
        assert tmp_db.count_nodes() == 0

    def test_delete_nonexistent_raises_lookup(self, tmp_db):
        with pytest.raises(LookupError):
            tmp_db.delete_node(9999)

    def test_delete_only_removes_target(self, tmp_db):
        id1 = tmp_db.create_node("Keep", "tank", "", "active")
        id2 = tmp_db.create_node("Remove", "valve", "", "active")
        tmp_db.delete_node(id2)
        assert tmp_db.get_node_by_id(id1) is not None
        assert tmp_db.count_nodes() == 1


# ── HTTP integration tests ────────────────────────────────────────────────────

class TestNetworkRoutes:
    def test_network_page_loads(self, app_client):
        r = app_client.get("/network")
        assert r.status_code == 200
        assert b"Water Network" in r.data

    def test_create_via_post_redirects(self, app_client):
        r = app_client.post(
            "/network/create",
            data={"name": "HTTP Tank", "node_type": "tank", "location": "", "status": "active"},
        )
        assert r.status_code == 302
        assert "/network" in r.headers["Location"]

    def test_create_then_listed(self, app_client):
        app_client.post(
            "/network/create",
            data={"name": "Listed Node", "node_type": "junction", "location": "Z1", "status": "active"},
        )
        r = app_client.get("/network")
        assert b"Listed Node" in r.data

    def test_create_blank_name_stays_on_page(self, app_client):
        r = app_client.post(
            "/network/create",
            data={"name": "", "node_type": "tank", "location": "", "status": "active"},
            follow_redirects=True,
        )
        assert r.status_code == 200
        # Flash error message should be present
        assert b"required" in r.data.lower() or b"Name" in r.data

    def test_delete_via_post(self, app_client, tmp_db):
        node_id = tmp_db.create_node("Del Via HTTP", "valve", "", "active")
        r = app_client.post(f"/network/{node_id}/delete")
        assert r.status_code == 302
        assert tmp_db.get_node_by_id(node_id) is None

    def test_delete_get_not_allowed(self, app_client, tmp_db):
        node_id = tmp_db.create_node("GET Del Test", "tank", "", "active")
        r = app_client.get(f"/network/{node_id}/delete")
        assert r.status_code == 405

    def test_edit_page_loads(self, app_client, tmp_db):
        node_id = tmp_db.create_node("Edit Me", "tank", "Zone 4", "active")
        r = app_client.get(f"/network/{node_id}/edit")
        assert r.status_code == 200
        assert b"Edit Me" in r.data

    def test_edit_post_updates(self, app_client, tmp_db):
        node_id = tmp_db.create_node("Before", "tank", "", "active")
        r = app_client.post(
            f"/network/{node_id}/edit",
            data={"name": "After", "node_type": "valve", "location": "Z5", "status": "inactive"},
        )
        assert r.status_code == 302
        node = tmp_db.get_node_by_id(node_id)
        assert node["name"] == "After"

    def test_dashboard_shows_live_count(self, app_client, tmp_db):
        tmp_db.create_node("Count Node", "tank", "", "active")
        r = app_client.get("/")
        assert b"1" in r.data  # node count reflected on dashboard
