"""
tests/test_pipes.py – pytest suite for Pipe Management.

Run with:
    python -m pytest tests/ -v

Covers:
  - Valid pipe creation (DB layer)
  - Invalid endpoints (non-existent node IDs)
  - Same-node self-loop rejection
  - Edit (update) a pipe
  - Delete a pipe
  - FK enforcement: cannot delete a node that is referenced by a pipe
  - HTTP integration: create, list, edit, delete via the Flask test client
  - GET /pipes/delete is 405 (POST-only)
  - Routing-cost field: zero is allowed, negative is rejected
  - get_all_pipes_for_graph() structure is correct for adjacency-list use
"""

import os
import sys

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
    initialise the schema.  Completely isolates tests from each other and from
    the real instance/aquagrid.db file.
    """
    import database as db_module

    test_db = str(tmp_path / "test_aquagrid.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


@pytest.fixture()
def two_nodes(tmp_db):
    """Return (node_a_id, node_b_id) after creating two nodes."""
    id_a = tmp_db.create_node("Tank Alpha", "tank", "Zone 1", "active")
    id_b = tmp_db.create_node("Junction Beta", "junction", "Zone 2", "active")
    return id_a, id_b


@pytest.fixture()
def app_client(tmp_db, monkeypatch):
    """
    Flask test client wired to the same temp database.
    CSRF is disabled in the test config so form tests stay simple.
    """
    import database as db_module
    import app as app_module

    monkeypatch.setattr(app_module, "db", db_module)

    app_module.app.config["TESTING"] = True
    app_module.app.config["WTF_CSRF_ENABLED"] = False
    app_module.app.config["SECRET_KEY"] = "test-secret"

    with app_module.app.test_client() as client:
        yield client


# ── DB-layer unit tests: creation ─────────────────────────────────────────────

class TestCreatePipe:
    def test_valid_creation_returns_id(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        pipe_id = tmp_db.create_pipe(
            "Trunk-01", id_a, id_b, "250", "500", "1", "active"
        )
        assert isinstance(pipe_id, int)
        assert pipe_id >= 1

    def test_created_pipe_is_retrievable(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        tmp_db.create_pipe("Trunk-01", id_a, id_b, "100", "200", "1.5", "active")
        pipes = tmp_db.get_all_pipes()
        assert len(pipes) == 1
        assert pipes[0]["name"] == "Trunk-01"
        assert pipes[0]["from_node_id"] == id_a
        assert pipes[0]["to_node_id"] == id_b
        assert pipes[0]["length_m"] == pytest.approx(100.0)
        assert pipes[0]["capacity_lpm"] == pytest.approx(200.0)
        assert pipes[0]["cost"] == pytest.approx(1.5)
        assert pipes[0]["status"] == "active"

    def test_created_pipe_includes_node_names(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        pid = tmp_db.create_pipe("P1", id_a, id_b, "50", "100", "1", "active")
        pipe = tmp_db.get_pipe_by_id(pid)
        assert pipe["from_node_name"] == "Tank Alpha"
        assert pipe["to_node_name"] == "Junction Beta"

    def test_zero_cost_is_allowed(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        pipe_id = tmp_db.create_pipe("ZeroCost", id_a, id_b, "10", "10", "0", "active")
        pipe = tmp_db.get_pipe_by_id(pipe_id)
        assert pipe["cost"] == pytest.approx(0.0)

    def test_inactive_status_accepted(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        pipe_id = tmp_db.create_pipe("Closed", id_a, id_b, "10", "10", "1", "inactive")
        pipe = tmp_db.get_pipe_by_id(pipe_id)
        assert pipe["status"] == "inactive"

    def test_count_pipes_increments(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        assert tmp_db.count_pipes() == 0
        tmp_db.create_pipe("P1", id_a, id_b, "10", "10", "1", "active")
        assert tmp_db.count_pipes() == 1
        tmp_db.create_pipe("P2", id_b, id_a, "20", "20", "2", "active")
        assert tmp_db.count_pipes() == 2


# ── DB-layer unit tests: validation ──────────────────────────────────────────

class TestPipeValidation:
    def test_blank_name_raises(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        with pytest.raises(ValueError) as exc_info:
            tmp_db.create_pipe("", id_a, id_b, "10", "10", "1", "active")
        msgs = exc_info.value.args[0]
        assert any("name" in m.lower() for m in msgs)

    def test_name_too_long_raises(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        with pytest.raises(ValueError):
            tmp_db.create_pipe("x" * 121, id_a, id_b, "10", "10", "1", "active")

    def test_nonexistent_from_node_raises(self, tmp_db, two_nodes):
        """from_node_id that doesn't exist in the DB must be rejected."""
        _, id_b = two_nodes
        with pytest.raises(ValueError) as exc_info:
            tmp_db.create_pipe("BadFrom", 9999, id_b, "10", "10", "1", "active")
        msgs = exc_info.value.args[0]
        assert any("from" in m.lower() for m in msgs)

    def test_nonexistent_to_node_raises(self, tmp_db, two_nodes):
        """to_node_id that doesn't exist in the DB must be rejected."""
        id_a, _ = two_nodes
        with pytest.raises(ValueError) as exc_info:
            tmp_db.create_pipe("BadTo", id_a, 9999, "10", "10", "1", "active")
        msgs = exc_info.value.args[0]
        assert any("to" in m.lower() for m in msgs)

    def test_none_from_node_raises(self, tmp_db, two_nodes):
        _, id_b = two_nodes
        with pytest.raises(ValueError) as exc_info:
            tmp_db.create_pipe("NoFrom", None, id_b, "10", "10", "1", "active")
        msgs = exc_info.value.args[0]
        assert any("from" in m.lower() for m in msgs)

    def test_none_to_node_raises(self, tmp_db, two_nodes):
        id_a, _ = two_nodes
        with pytest.raises(ValueError) as exc_info:
            tmp_db.create_pipe("NoTo", id_a, None, "10", "10", "1", "active")
        msgs = exc_info.value.args[0]
        assert any("to" in m.lower() for m in msgs)

    def test_same_node_both_ends_raises(self, tmp_db, two_nodes):
        """A pipe whose from_node_id == to_node_id must be rejected."""
        id_a, _ = two_nodes
        with pytest.raises(ValueError) as exc_info:
            tmp_db.create_pipe("Loop", id_a, id_a, "10", "10", "1", "active")
        msgs = exc_info.value.args[0]
        assert any("itself" in m.lower() or "same" in m.lower() for m in msgs)

    def test_negative_length_raises(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        with pytest.raises(ValueError) as exc_info:
            tmp_db.create_pipe("NegLen", id_a, id_b, "-5", "10", "1", "active")
        msgs = exc_info.value.args[0]
        assert any("length" in m.lower() for m in msgs)

    def test_zero_length_raises(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        with pytest.raises(ValueError):
            tmp_db.create_pipe("ZeroLen", id_a, id_b, "0", "10", "1", "active")

    def test_non_numeric_length_raises(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        with pytest.raises(ValueError):
            tmp_db.create_pipe("BadLen", id_a, id_b, "abc", "10", "1", "active")

    def test_negative_capacity_raises(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        with pytest.raises(ValueError) as exc_info:
            tmp_db.create_pipe("NegCap", id_a, id_b, "10", "-1", "1", "active")
        msgs = exc_info.value.args[0]
        assert any("capacity" in m.lower() for m in msgs)

    def test_negative_cost_raises(self, tmp_db, two_nodes):
        """Routing cost must be ≥ 0."""
        id_a, id_b = two_nodes
        with pytest.raises(ValueError) as exc_info:
            tmp_db.create_pipe("NegCost", id_a, id_b, "10", "10", "-1", "active")
        msgs = exc_info.value.args[0]
        assert any("cost" in m.lower() for m in msgs)

    def test_invalid_status_raises(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        with pytest.raises(ValueError) as exc_info:
            tmp_db.create_pipe("BadStatus", id_a, id_b, "10", "10", "1", "broken")
        msgs = exc_info.value.args[0]
        assert any("status" in m.lower() for m in msgs)

    def test_default_routing_cost_calculation(self, tmp_db, two_nodes):
        """When cost is blank, auto-compute default routing cost from length (1 unit per 100m)."""
        id_a, id_b = two_nodes
        pipe_id = tmp_db.create_pipe("DefaultCostPipe", id_a, id_b, "350.0", "500", "", "active")
        pipe = tmp_db.get_pipe_by_id(pipe_id)
        assert pipe["cost"] == pytest.approx(3.5)

    def test_require_positive_cost_enforcement(self, tmp_db, two_nodes):
        """When require_positive_cost=True, zero or negative cost is rejected."""
        id_a, id_b = two_nodes
        with pytest.raises(ValueError) as exc_info:
            tmp_db.create_pipe("ZeroCostPipe", id_a, id_b, "100", "500", "0", "active", require_positive_cost=True)
        msgs = exc_info.value.args[0]
        assert any("greater than zero" in m.lower() for m in msgs)

    def test_multiple_errors_returned_together(self, tmp_db, two_nodes):
        """All validation errors should be collected and raised together."""
        id_a, _ = two_nodes
        with pytest.raises(ValueError) as exc_info:
            tmp_db.create_pipe("", id_a, id_a, "-5", "-1", "-2", "broken")
        msgs = exc_info.value.args[0]
        # At minimum: blank name + self-loop + bad length + bad capacity + bad cost + bad status
        assert len(msgs) >= 3


# ── DB-layer unit tests: edit ─────────────────────────────────────────────────

class TestEditPipe:
    def test_update_changes_values(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        pipe_id = tmp_db.create_pipe("Old", id_a, id_b, "10", "10", "1", "active")
        tmp_db.update_pipe(pipe_id, "New", id_b, id_a, "200", "400", "3.5", "inactive")
        pipe = tmp_db.get_pipe_by_id(pipe_id)
        assert pipe["name"] == "New"
        assert pipe["from_node_id"] == id_b
        assert pipe["to_node_id"] == id_a
        assert pipe["length_m"] == pytest.approx(200.0)
        assert pipe["capacity_lpm"] == pytest.approx(400.0)
        assert pipe["cost"] == pytest.approx(3.5)
        assert pipe["status"] == "inactive"

    def test_update_nonexistent_raises_lookup(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        with pytest.raises(LookupError):
            tmp_db.update_pipe(9999, "X", id_a, id_b, "10", "10", "1", "active")

    def test_update_with_invalid_nodes_raises_value_error(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        pipe_id = tmp_db.create_pipe("P", id_a, id_b, "10", "10", "1", "active")
        with pytest.raises(ValueError):
            tmp_db.update_pipe(pipe_id, "P", 9999, id_b, "10", "10", "1", "active")

    def test_update_to_self_loop_raises_value_error(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        pipe_id = tmp_db.create_pipe("P", id_a, id_b, "10", "10", "1", "active")
        with pytest.raises(ValueError):
            tmp_db.update_pipe(pipe_id, "P", id_a, id_a, "10", "10", "1", "active")


# ── DB-layer unit tests: delete ───────────────────────────────────────────────

class TestDeletePipe:
    def test_delete_removes_pipe(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        pipe_id = tmp_db.create_pipe("Del", id_a, id_b, "10", "10", "1", "active")
        tmp_db.delete_pipe(pipe_id)
        assert tmp_db.get_pipe_by_id(pipe_id) is None
        assert tmp_db.count_pipes() == 0

    def test_delete_nonexistent_raises_lookup(self, tmp_db):
        with pytest.raises(LookupError):
            tmp_db.delete_pipe(9999)

    def test_delete_only_removes_target(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        pid1 = tmp_db.create_pipe("Keep", id_a, id_b, "10", "10", "1", "active")
        pid2 = tmp_db.create_pipe("Remove", id_b, id_a, "20", "20", "2", "active")
        tmp_db.delete_pipe(pid2)
        assert tmp_db.get_pipe_by_id(pid1) is not None
        assert tmp_db.count_pipes() == 1


# ── FK enforcement: cannot delete a node referenced by a pipe ─────────────────

class TestForeignKeyEnforcement:
    def test_cannot_delete_node_used_by_pipe(self, tmp_db, two_nodes):
        """
        delete_node must raise RuntimeError (not crash the DB) when the node
        is still referenced as from_node_id or to_node_id in the pipes table.
        """
        id_a, id_b = two_nodes
        tmp_db.create_pipe("Blocking", id_a, id_b, "10", "10", "1", "active")
        with pytest.raises(RuntimeError) as exc_info:
            tmp_db.delete_node(id_a)
        assert "pipe" in str(exc_info.value).lower()

    def test_can_delete_node_after_pipe_is_removed(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        pipe_id = tmp_db.create_pipe("Temp", id_a, id_b, "10", "10", "1", "active")
        tmp_db.delete_pipe(pipe_id)
        tmp_db.delete_node(id_a)  # should not raise
        assert tmp_db.get_node_by_id(id_a) is None

    def test_cannot_delete_to_node_used_by_pipe(self, tmp_db, two_nodes):
        id_a, id_b = two_nodes
        tmp_db.create_pipe("Blocking", id_a, id_b, "10", "10", "1", "active")
        with pytest.raises(RuntimeError):
            tmp_db.delete_node(id_b)


# ── Graph-readiness test ──────────────────────────────────────────────────────

class TestGraphReadiness:
    def test_get_all_pipes_for_graph_structure(self, tmp_db, two_nodes):
        """
        get_all_pipes_for_graph() must return plain dicts with the keys needed
        to build an adjacency list without any schema changes.
        """
        id_a, id_b = two_nodes
        tmp_db.create_pipe("G1", id_a, id_b, "100", "200", "2.5", "active")
        tmp_db.create_pipe("G2", id_b, id_a, "50", "100", "1.0", "inactive")

        edges = tmp_db.get_all_pipes_for_graph()
        # Only active pipes are included
        assert len(edges) == 1
        edge = edges[0]
        assert set(edge.keys()) >= {"pipe_id", "from_node_id", "to_node_id", "cost", "length_m", "capacity_lpm"}
        assert edge["from_node_id"] == id_a
        assert edge["to_node_id"] == id_b
        assert edge["cost"] == pytest.approx(2.5)

    def test_adjacency_list_can_be_built(self, tmp_db, two_nodes):
        """Demonstrate that the returned data plugs straight into an adjacency list."""
        id_a, id_b = two_nodes
        tmp_db.create_pipe("G1", id_a, id_b, "100", "200", "2.5", "active")

        graph: dict = {}
        for edge in tmp_db.get_all_pipes_for_graph():
            graph.setdefault(edge["from_node_id"], []).append(
                (edge["to_node_id"], edge["cost"], edge["pipe_id"])
            )

        assert id_a in graph
        neighbours = graph[id_a]
        assert len(neighbours) == 1
        to_id, cost, _ = neighbours[0]
        assert to_id == id_b
        assert cost == pytest.approx(2.5)


# ── HTTP integration tests ────────────────────────────────────────────────────

class TestPipeRoutes:
    def _create_two_nodes(self, db):
        id_a = db.create_node("Tank A", "tank", "", "active")
        id_b = db.create_node("Junction B", "junction", "", "active")
        return id_a, id_b

    def test_pipes_page_loads(self, app_client):
        r = app_client.get("/pipes")
        assert r.status_code == 200
        assert b"Pipe Management" in r.data

    def test_create_pipe_redirects(self, app_client, tmp_db):
        id_a, id_b = self._create_two_nodes(tmp_db)
        r = app_client.post(
            "/pipes/create",
            data={
                "name": "HTTP-Pipe-01",
                "from_node_id": id_a,
                "to_node_id": id_b,
                "length_m": "100",
                "capacity_lpm": "200",
                "cost": "1",
                "status": "active",
            },
        )
        assert r.status_code == 302
        assert "/pipes" in r.headers["Location"]

    def test_create_pipe_then_listed(self, app_client, tmp_db):
        id_a, id_b = self._create_two_nodes(tmp_db)
        app_client.post(
            "/pipes/create",
            data={
                "name": "Listed-Pipe",
                "from_node_id": id_a,
                "to_node_id": id_b,
                "length_m": "50",
                "capacity_lpm": "100",
                "cost": "1",
                "status": "active",
            },
        )
        r = app_client.get("/pipes")
        assert b"Listed-Pipe" in r.data

    def test_create_blank_name_shows_error(self, app_client, tmp_db):
        id_a, id_b = self._create_two_nodes(tmp_db)
        r = app_client.post(
            "/pipes/create",
            data={
                "name": "",
                "from_node_id": id_a,
                "to_node_id": id_b,
                "length_m": "10",
                "capacity_lpm": "10",
                "cost": "1",
                "status": "active",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        assert b"required" in r.data.lower() or b"name" in r.data.lower()

    def test_create_same_node_both_ends_shows_error(self, app_client, tmp_db):
        id_a, _ = self._create_two_nodes(tmp_db)
        r = app_client.post(
            "/pipes/create",
            data={
                "name": "SelfLoop",
                "from_node_id": id_a,
                "to_node_id": id_a,
                "length_m": "10",
                "capacity_lpm": "10",
                "cost": "1",
                "status": "active",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        assert b"itself" in r.data.lower() or b"same" in r.data.lower()

    def test_create_nonexistent_node_shows_error(self, app_client, tmp_db):
        id_a, _ = self._create_two_nodes(tmp_db)
        r = app_client.post(
            "/pipes/create",
            data={
                "name": "BadNode",
                "from_node_id": id_a,
                "to_node_id": 9999,
                "length_m": "10",
                "capacity_lpm": "10",
                "cost": "1",
                "status": "active",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        assert b"not exist" in r.data.lower() or b"does not" in r.data.lower()

    def test_web_pipe_create_rejects_zero_cost(self, app_client, tmp_db):
        """Form submission prevents accidental zero routing cost for normal physical pipes."""
        id_a, id_b = self._create_two_nodes(tmp_db)
        r = app_client.post(
            "/pipes/create",
            data={
                "name": "ZeroCostWeb",
                "from_node_id": id_a,
                "to_node_id": id_b,
                "length_m": "150",
                "capacity_lpm": "400",
                "cost": "0",
                "status": "active",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        assert b"greater than zero" in r.data.lower()

    def test_web_pipe_edit_allows_admin_custom_cost(self, app_client, tmp_db):
        """Administrators can edit routing costs and changes are persisted."""
        id_a, id_b = self._create_two_nodes(tmp_db)
        pipe_id = tmp_db.create_pipe("EditableCost", id_a, id_b, "200", "300", "2.0", "active")
        r = app_client.post(
            f"/pipes/{pipe_id}/edit",
            data={
                "name": "EditableCost",
                "from_node_id": id_a,
                "to_node_id": id_b,
                "length_m": "200",
                "capacity_lpm": "300",
                "cost": "5.45",
                "status": "active",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        pipe = tmp_db.get_pipe_by_id(pipe_id)
        assert pipe["cost"] == pytest.approx(5.45)

    def test_delete_pipe_via_post(self, app_client, tmp_db):
        id_a, id_b = self._create_two_nodes(tmp_db)
        pipe_id = tmp_db.create_pipe("ToDelete", id_a, id_b, "10", "10", "1", "active")
        r = app_client.post(f"/pipes/{pipe_id}/delete")
        assert r.status_code == 302
        assert tmp_db.get_pipe_by_id(pipe_id) is None

    def test_delete_pipe_get_not_allowed(self, app_client, tmp_db):
        id_a, id_b = self._create_two_nodes(tmp_db)
        pipe_id = tmp_db.create_pipe("GETDel", id_a, id_b, "10", "10", "1", "active")
        r = app_client.get(f"/pipes/{pipe_id}/delete")
        assert r.status_code == 405

    def test_edit_pipe_page_loads(self, app_client, tmp_db):
        id_a, id_b = self._create_two_nodes(tmp_db)
        pipe_id = tmp_db.create_pipe("EditMe", id_a, id_b, "10", "10", "1", "active")
        r = app_client.get(f"/pipes/{pipe_id}/edit")
        assert r.status_code == 200
        assert b"EditMe" in r.data

    def test_edit_pipe_post_updates(self, app_client, tmp_db):
        id_a, id_b = self._create_two_nodes(tmp_db)
        pipe_id = tmp_db.create_pipe("Before", id_a, id_b, "10", "10", "1", "active")
        r = app_client.post(
            f"/pipes/{pipe_id}/edit",
            data={
                "name": "After",
                "from_node_id": id_b,
                "to_node_id": id_a,
                "length_m": "999",
                "capacity_lpm": "888",
                "cost": "7.7",
                "status": "inactive",
            },
        )
        assert r.status_code == 302
        pipe = tmp_db.get_pipe_by_id(pipe_id)
        assert pipe["name"] == "After"
        assert pipe["length_m"] == pytest.approx(999.0)
        assert pipe["status"] == "inactive"

    def test_delete_node_blocked_by_pipe_shows_message(self, app_client, tmp_db):
        """
        Deleting a node that is referenced by a pipe must show a friendly
        flash message, not a server error.
        """
        id_a, id_b = self._create_two_nodes(tmp_db)
        tmp_db.create_pipe("Blocking", id_a, id_b, "10", "10", "1", "active")
        r = app_client.post(f"/network/{id_a}/delete", follow_redirects=True)
        assert r.status_code == 200
        # The flash error about pipes must appear on the page
        assert b"pipe" in r.data.lower()
