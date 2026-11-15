"""
database.py – AquaGrid database access layer.

All SQLite operations for the network_nodes and pipes tables live here.
Route handlers in app.py should only call these functions; they must not
build SQL strings themselves.

Graph-readiness note
--------------------
get_all_pipes_for_graph() returns every pipe as a dict suitable for
building a Python adjacency list:
    { from_node_id: [(to_node_id, cost, pipe_id), ...], ... }
That function is ready for use by a future Dijkstra / routing module
without any schema changes.
"""

import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any


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

# ── Path resolution ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(_HERE, "instance")
DB_PATH = os.path.join(INSTANCE_DIR, "aquagrid.db")

# ── Allowed domain values ────────────────────────────────────────────────────
ALLOWED_NODE_TYPES   = {"tank", "junction", "valve"}
ALLOWED_STATUSES     = {"active", "inactive"}
ALLOWED_SENSOR_TYPES = {"flow_meter", "pressure_sensor"}
DEFAULT_SENSOR_UNITS = {
    "flow_meter": "L/min",
    "pressure_sensor": "psi",
}
ALLOWED_SENSOR_UNITS = {"L/min", "m³/h", "L/s", "psi", "bar", "kPa"}
ALLOWED_SIMULATION_SCENARIOS = {"normal", "leak"}
ALLOWED_REPAIR_STATUSES      = {"Pending", "In Progress", "Completed"}
ALLOWED_REPAIR_CATEGORIES    = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
ALLOWED_ZONE_CATEGORIES      = {"critical", "standard", "commercial", "agricultural"}
ALLOWED_ZONE_STATUSES        = {"active", "inactive"}
ALLOWED_ROUTING_STRATEGIES   = {"SHORTEST_PATH", "WIDEST_PATH"}
ALLOWED_FEASIBILITY_STATUSES = {"FEASIBLE", "CAPACITY_CONSTRAINED", "UNREACHABLE", "UNMAPPED", "ZERO_ALLOCATION"}


# ── Connection helpers ───────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    """Return a connection with row_factory, busy timeout, and foreign-key enforcement on."""
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    # PRAGMA foreign_keys must be set on every new connection – SQLite resets
    # it to OFF each time.
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ── Schema initialisation ────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they do not already exist. Safe to call on start."""
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        # ── network_nodes (unchanged) ────────────────────────────────────────
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS network_nodes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                node_type   TEXT    NOT NULL CHECK(node_type IN ('tank','junction','valve')),
                location    TEXT,
                status      TEXT    NOT NULL DEFAULT 'active'
                                    CHECK(status IN ('active','inactive')),
                created_at  TEXT    NOT NULL
            )
            """
        )

        # ── pipes ────────────────────────────────────────────────────────────
        # cost is a configurable routing weight (e.g. distance, priority, flow
        # resistance).  It is NOT a financial cost and NOT live sensor data.
        # A future Dijkstra module will read this column as edge weight.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipes (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                from_node_id  INTEGER NOT NULL
                              REFERENCES network_nodes(id) ON DELETE RESTRICT,
                to_node_id    INTEGER NOT NULL
                              REFERENCES network_nodes(id) ON DELETE RESTRICT,
                length_m      REAL    NOT NULL CHECK(length_m > 0),
                capacity_lpm  REAL    NOT NULL CHECK(capacity_lpm > 0),
                cost          REAL    NOT NULL DEFAULT 1.0 CHECK(cost >= 0),
                status        TEXT    NOT NULL DEFAULT 'active'
                                      CHECK(status IN ('active','inactive')),
                created_at    TEXT    NOT NULL,
                CHECK(from_node_id != to_node_id)
            )
            """
        )

        # ── sensors ──────────────────────────────────────────────────────────
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sensors (
                sensor_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                sensor_name  TEXT    NOT NULL,
                pipe_id      INTEGER NOT NULL
                             REFERENCES pipes(id) ON DELETE RESTRICT,
                sensor_type  TEXT    NOT NULL
                             CHECK(sensor_type IN ('flow_meter', 'pressure_sensor')),
                unit         TEXT    NOT NULL,
                status       TEXT    NOT NULL DEFAULT 'active'
                             CHECK(status IN ('active', 'inactive')),
                created_at   TEXT    NOT NULL
            )
            """
        )

        # ── sensor_readings ──────────────────────────────────────────────────
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sensor_readings (
                reading_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                sensor_id      INTEGER NOT NULL
                               REFERENCES sensors(sensor_id) ON DELETE RESTRICT,
                reading_value  REAL    NOT NULL,
                recorded_at    TEXT    NOT NULL,
                reading_type   TEXT,
                unit           TEXT,
                is_simulated   INTEGER NOT NULL DEFAULT 1
                               CHECK(is_simulated IN (0, 1)),
                scenario       TEXT    CHECK(scenario IN ('normal', 'leak') OR scenario IS NULL)
            )
            """
        )

        # Additive migration: ensure scenario provenance column exists in sensor_readings
        sr_cols = [c["name"] for c in conn.execute("PRAGMA table_info(sensor_readings)").fetchall()]
        if "scenario" not in sr_cols:
            conn.execute("ALTER TABLE sensor_readings ADD COLUMN scenario TEXT")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_sensors_pipe_id ON sensors(pipe_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sensor_readings_sensor_id ON sensor_readings(sensor_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sensor_readings_scenario ON sensor_readings(sensor_id, scenario)")

        # ── leak_events ──────────────────────────────────────────────────────
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS leak_events (
                event_id               INTEGER PRIMARY KEY AUTOINCREMENT,
                pipe_id                INTEGER NOT NULL
                                       REFERENCES pipes(id) ON DELETE RESTRICT,
                detected_at            TEXT    NOT NULL,
                severity               TEXT    NOT NULL
                                       CHECK(severity IN ('low', 'medium', 'high', 'critical')),
                status                 TEXT    NOT NULL DEFAULT 'open'
                                       CHECK(status IN ('open', 'investigating', 'resolved')),
                flow_anomaly_score     REAL,
                pressure_anomaly_score REAL,
                baseline_values        TEXT,
                observed_values        TEXT,
                evidence_description   TEXT    NOT NULL,
                is_simulated           INTEGER NOT NULL DEFAULT 1
                                       CHECK(is_simulated IN (0, 1)),
                resolved_at            TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leak_events_pipe_id ON leak_events(pipe_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leak_events_status ON leak_events(status)")

        # ── repair_tasks ─────────────────────────────────────────────────────
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS repair_tasks (
                task_id            INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id           INTEGER NOT NULL UNIQUE
                                   REFERENCES leak_events(event_id) ON DELETE RESTRICT,
                pipe_id            INTEGER NOT NULL
                                   REFERENCES pipes(id) ON DELETE RESTRICT,
                priority_score     REAL    NOT NULL,
                priority_category  TEXT    NOT NULL
                                   CHECK(priority_category IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')),
                status             TEXT    NOT NULL DEFAULT 'Pending'
                                   CHECK(status IN ('Pending', 'In Progress', 'Completed')),
                score_breakdown    TEXT,
                assigned_to        TEXT,
                created_at         TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at         TEXT    NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_repair_tasks_status ON repair_tasks(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_repair_tasks_pipe_id ON repair_tasks(pipe_id)")

        # ── lifecycle_audit_log ──────────────────────────────────────────────
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lifecycle_audit_log (
                audit_id            INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type         TEXT    NOT NULL
                                    CHECK(entity_type IN ('repair_task', 'leak_event')),
                entity_id           INTEGER NOT NULL,
                repair_task_id      INTEGER,
                leak_event_id       INTEGER,
                pipe_id             INTEGER,
                previous_status     TEXT,
                new_status          TEXT    NOT NULL,
                action_description  TEXT    NOT NULL,
                changed_at          TEXT    NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lifecycle_audit_entity ON lifecycle_audit_log(entity_type, entity_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lifecycle_audit_task ON lifecycle_audit_log(repair_task_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lifecycle_audit_event ON lifecycle_audit_log(leak_event_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lifecycle_audit_changed ON lifecycle_audit_log(changed_at)")

        # ── allocation_zones ─────────────────────────────────────────────────
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS allocation_zones (
                zone_id            INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_name          TEXT    NOT NULL UNIQUE,
                node_id            INTEGER REFERENCES network_nodes(id) ON DELETE SET NULL,
                priority_category  TEXT    NOT NULL DEFAULT 'standard'
                                   CHECK(priority_category IN ('critical', 'standard', 'commercial', 'agricultural')),
                requested_demand_l REAL    NOT NULL CHECK(requested_demand_l >= 0),
                essential_min_l    REAL    NOT NULL DEFAULT 0.0
                                   CHECK(essential_min_l >= 0 AND essential_min_l <= requested_demand_l),
                fairness_weight    REAL    NOT NULL DEFAULT 1.0 CHECK(fairness_weight > 0),
                status             TEXT    NOT NULL DEFAULT 'active'
                                   CHECK(status IN ('active', 'inactive')),
                description        TEXT,
                created_at         TEXT    NOT NULL,
                updated_at         TEXT    NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alloc_zones_status ON allocation_zones(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alloc_zones_node_id ON allocation_zones(node_id)")

        # ── allocation_runs ──────────────────────────────────────────────────
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS allocation_runs (
                run_id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp            TEXT    NOT NULL,
                total_available_supply_l REAL    NOT NULL CHECK(total_available_supply_l >= 0),
                total_requested_demand_l REAL    NOT NULL CHECK(total_requested_demand_l >= 0),
                total_allocated_l        REAL    NOT NULL CHECK(total_allocated_l >= 0),
                total_unmet_l            REAL    NOT NULL DEFAULT 0.0 CHECK(total_unmet_l >= 0),
                satisfaction_rate_pct    REAL    NOT NULL CHECK(satisfaction_rate_pct >= 0 AND satisfaction_rate_pct <= 100),
                allocation_policy        TEXT    NOT NULL,
                active_zone_count        INTEGER NOT NULL CHECK(active_zone_count >= 0),
                period_start             TEXT,
                period_end               TEXT,
                notes                    TEXT
            )
            """
        )
        # Additive migration: ensure total_unmet_l column exists in allocation_runs
        ar_cols = [c["name"] for c in conn.execute("PRAGMA table_info(allocation_runs)").fetchall()]
        if "total_unmet_l" not in ar_cols:
            conn.execute("ALTER TABLE allocation_runs ADD COLUMN total_unmet_l REAL NOT NULL DEFAULT 0.0")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_alloc_runs_timestamp ON allocation_runs(run_timestamp)")

        # ── water_allocations ────────────────────────────────────────────────
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS water_allocations (
                allocation_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                INTEGER NOT NULL
                                      REFERENCES allocation_runs(run_id) ON DELETE RESTRICT,
                zone_id               INTEGER NOT NULL
                                      REFERENCES allocation_zones(zone_id) ON DELETE RESTRICT,
                zone_name_snapshot    TEXT    NOT NULL,
                requested_demand_l    REAL    NOT NULL CHECK(requested_demand_l >= 0),
                essential_min_l       REAL    NOT NULL CHECK(essential_min_l >= 0),
                fairness_weight       REAL    NOT NULL CHECK(fairness_weight > 0),
                allocated_amount_l    REAL    NOT NULL CHECK(allocated_amount_l >= 0),
                unmet_demand_l        REAL    NOT NULL CHECK(unmet_demand_l >= 0),
                satisfaction_rate_pct REAL    NOT NULL CHECK(satisfaction_rate_pct >= 0 AND satisfaction_rate_pct <= 100),
                is_bottleneck         INTEGER NOT NULL DEFAULT 0 CHECK(is_bottleneck IN (0, 1)),
                tier_reached          INTEGER NOT NULL,
                recorded_at           TEXT    NOT NULL,
                destination_node_id   INTEGER
                                      REFERENCES network_nodes(id) ON DELETE SET NULL
            )
            """
        )
        # Additive migration: ensure destination_node_id column exists on water_allocations
        wa_cols = [c["name"] for c in conn.execute("PRAGMA table_info(water_allocations)").fetchall()]
        if "destination_node_id" not in wa_cols:
            conn.execute(
                "ALTER TABLE water_allocations ADD COLUMN destination_node_id INTEGER REFERENCES network_nodes(id) ON DELETE SET NULL"
            )

        conn.execute("CREATE INDEX IF NOT EXISTS idx_water_alloc_run_id ON water_allocations(run_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_water_alloc_zone_id ON water_allocations(zone_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_water_alloc_dest_node ON water_allocations(destination_node_id)")

        # ── delivery_assessments ──────────────────────────────────────────────
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS delivery_assessments (
                assessment_id           INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                  INTEGER NOT NULL
                                        REFERENCES allocation_runs(run_id) ON DELETE RESTRICT,
                source_node_id          INTEGER
                                        REFERENCES network_nodes(id) ON DELETE SET NULL,
                source_name_snapshot    TEXT    NOT NULL,
                duration_minutes        REAL    NOT NULL CHECK(duration_minutes > 0),
                routing_strategy        TEXT    NOT NULL
                                        CHECK(routing_strategy IN ('SHORTEST_PATH', 'WIDEST_PATH')),
                total_allocated_l       REAL    NOT NULL CHECK(total_allocated_l >= 0),
                total_deliverable_l     REAL    NOT NULL CHECK(total_deliverable_l >= 0),
                total_undelivered_l     REAL    NOT NULL CHECK(total_undelivered_l >= 0),
                delivery_efficiency_pct REAL    NOT NULL CHECK(delivery_efficiency_pct >= 0 AND delivery_efficiency_pct <= 100),
                has_contention          INTEGER NOT NULL DEFAULT 0 CHECK(has_contention IN (0, 1)),
                contention_warnings_json TEXT,
                notes                   TEXT,
                assessed_at             TEXT    NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_delivery_assess_run_id ON delivery_assessments(run_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_delivery_assess_source_node ON delivery_assessments(source_node_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_delivery_assess_assessed_at ON delivery_assessments(assessed_at)")

        # ── delivery_zone_assessments ─────────────────────────────────────────
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS delivery_zone_assessments (
                zone_assessment_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                assessment_id               INTEGER NOT NULL
                                            REFERENCES delivery_assessments(assessment_id) ON DELETE CASCADE,
                zone_id                     INTEGER
                                            REFERENCES allocation_zones(zone_id) ON DELETE SET NULL,
                zone_name_snapshot          TEXT    NOT NULL,
                destination_node_id         INTEGER
                                            REFERENCES network_nodes(id) ON DELETE SET NULL,
                destination_name_snapshot   TEXT,
                allocated_amount_l          REAL    NOT NULL CHECK(allocated_amount_l >= 0),
                deliverable_amount_l        REAL    NOT NULL CHECK(deliverable_amount_l >= 0),
                undelivered_amount_l        REAL    NOT NULL CHECK(undelivered_amount_l >= 0),
                required_rate_lpm           REAL    NOT NULL CHECK(required_rate_lpm >= 0),
                bottleneck_capacity_lpm     REAL    CHECK(bottleneck_capacity_lpm IS NULL OR bottleneck_capacity_lpm > 0),
                bottleneck_pipe_id          INTEGER
                                            REFERENCES pipes(id) ON DELETE SET NULL,
                bottleneck_pipe_name_snapshot TEXT,
                route_cost                  REAL    CHECK(route_cost IS NULL OR route_cost >= 0),
                route_node_ids              TEXT,
                route_pipe_ids              TEXT,
                delivery_efficiency_pct     REAL    NOT NULL CHECK(delivery_efficiency_pct >= 0 AND delivery_efficiency_pct <= 100),
                feasibility_status          TEXT    NOT NULL
                                            CHECK(feasibility_status IN ('FEASIBLE', 'CAPACITY_CONSTRAINED', 'UNREACHABLE', 'UNMAPPED', 'ZERO_ALLOCATION')),
                explanation                 TEXT    NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_deliv_zone_assess_id ON delivery_zone_assessments(assessment_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_deliv_zone_zone_id ON delivery_zone_assessments(zone_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_deliv_zone_status ON delivery_zone_assessments(feasibility_status)")

        conn.commit()


# ═══════════════════════════════════════════════════════════════════════════
#  NODE CRUD (unchanged from milestone 1)
# ═══════════════════════════════════════════════════════════════════════════

def validate_node(name: str, node_type: str, status: str) -> list[str]:
    """Return a list of human-readable error strings (empty = valid)."""
    errors: list[str] = []
    if not name or not name.strip():
        errors.append("Name is required and cannot be blank.")
    elif len(name.strip()) > 120:
        errors.append("Name must be 120 characters or fewer.")
    if node_type not in ALLOWED_NODE_TYPES:
        errors.append(
            f"Node type '{node_type}' is not allowed. "
            f"Choose one of: {', '.join(sorted(ALLOWED_NODE_TYPES))}."
        )
    if status not in ALLOWED_STATUSES:
        errors.append(
            f"Status '{status}' is not allowed. "
            f"Choose one of: {', '.join(sorted(ALLOWED_STATUSES))}."
        )
    return errors


def create_node(name: str, node_type: str, location: str, status: str) -> int:
    """Insert a new node and return its new id. Raises ValueError on bad input."""
    errors = validate_node(name, node_type, status)
    if errors:
        raise ValueError(errors)

    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO network_nodes (name, node_type, location, status, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name.strip(), node_type, location.strip() or None, status, now),
        )
        conn.commit()
        return cur.lastrowid


def get_all_nodes() -> list[sqlite3.Row]:
    """Return all nodes ordered newest-first."""
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM network_nodes ORDER BY created_at DESC"
        ).fetchall()


def get_node_by_id(node_id: int) -> sqlite3.Row | None:
    """Return a single node or None."""
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM network_nodes WHERE id = ?", (node_id,)
        ).fetchone()


def update_node(
    node_id: int, name: str, node_type: str, location: str, status: str
) -> None:
    """Update a node. Raises ValueError on bad input, LookupError if missing."""
    errors = validate_node(name, node_type, status)
    if errors:
        raise ValueError(errors)

    with _connect() as conn:
        result = conn.execute(
            """
            UPDATE network_nodes
               SET name = ?, node_type = ?, location = ?, status = ?
             WHERE id = ?
            """,
            (name.strip(), node_type, location.strip() or None, status, node_id),
        )
        conn.commit()
        if result.rowcount == 0:
            raise LookupError(f"Node {node_id} not found.")


def delete_node(node_id: int) -> None:
    """
    Delete a node by id.

    Raises LookupError  – node does not exist.
    Raises RuntimeError – node is still referenced by at least one pipe
                          (FK violation).  The caller must show a friendly
                          message rather than letting the DB error surface.
    """
    with _connect() as conn:
        # Check for referencing pipes before attempting the DELETE so the
        # error message is explicit.
        pipe_count = conn.execute(
            """
            SELECT COUNT(*) FROM pipes
             WHERE from_node_id = ? OR to_node_id = ?
            """,
            (node_id, node_id),
        ).fetchone()[0]

        if pipe_count > 0:
            raise RuntimeError(
                f"Cannot delete node {node_id}: it is still used by "
                f"{pipe_count} pipe{'s' if pipe_count != 1 else ''}. "
                f"Delete or reroute those pipes first."
            )

        result = conn.execute(
            "DELETE FROM network_nodes WHERE id = ?", (node_id,)
        )
        conn.commit()
        if result.rowcount == 0:
            raise LookupError(f"Node {node_id} not found.")


def count_nodes() -> int:
    """Return the total number of nodes."""
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM network_nodes").fetchone()[0]


def get_node_name_map() -> dict[int, str]:
    """Return {node_id: node_name} for all nodes – used to populate dropdowns."""
    with _connect() as conn:
        rows = conn.execute("SELECT id, name FROM network_nodes ORDER BY name").fetchall()
        return {row["id"]: row["name"] for row in rows}


# ═══════════════════════════════════════════════════════════════════════════
#  PIPE CRUD
# ═══════════════════════════════════════════════════════════════════════════

def _parse_positive_float(raw: str, field_label: str) -> tuple[float | None, str | None]:
    """
    Parse a string as a positive float.
    Returns (value, None) on success or (None, error_message) on failure.
    """
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None, f"{field_label} must be a number."
    if v <= 0:
        return None, f"{field_label} must be greater than zero."
    return v, None


def _parse_nonneg_float(raw: str, field_label: str) -> tuple[float | None, str | None]:
    """Parse as a non-negative float (≥ 0)."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None, f"{field_label} must be a number."
    if v < 0:
        return None, f"{field_label} must be zero or greater."
    return v, None


def calculate_default_routing_cost(length_m: float) -> float:
    """
    Configurable routing cost model for water network pipes.

    Cost Model:
        Default routing cost = round(length_m / 100.0, 2) (1.0 cost unit per 100 meters).
        Minimum routing cost is 0.01 units.

    This represents physical transit resistance proportional to pipe length while
    allowing administrators to manually configure/override the value.
    """
    if length_m <= 0:
        return 1.0
    return max(0.01, round(length_m / 100.0, 2))


def validate_pipe(
    name: str,
    from_node_id: int | None,
    to_node_id: int | None,
    length_m_raw: str,
    capacity_lpm_raw: str,
    cost_raw: str,
    status: str,
    existing_node_ids: set[int],
    require_positive_cost: bool = False,
) -> tuple[list[str], float | None, float | None, float | None]:
    """
    Validate all pipe fields.

    Returns (errors, length_m, capacity_lpm, cost).
    Numeric fields are None when they failed to parse.
    """
    errors: list[str] = []

    if not name or not name.strip():
        errors.append("Pipe name is required and cannot be blank.")
    elif len(name.strip()) > 120:
        errors.append("Pipe name must be 120 characters or fewer.")

    if from_node_id is None:
        errors.append("From-node is required.")
    elif from_node_id not in existing_node_ids:
        errors.append("From-node does not exist.")

    if to_node_id is None:
        errors.append("To-node is required.")
    elif to_node_id not in existing_node_ids:
        errors.append("To-node does not exist.")

    if (
        from_node_id is not None
        and to_node_id is not None
        and from_node_id == to_node_id
    ):
        errors.append("A pipe cannot connect a node to itself.")

    length_m, err = _parse_positive_float(length_m_raw, "Length (m)")
    if err:
        errors.append(err)

    capacity_lpm, err = _parse_positive_float(capacity_lpm_raw, "Capacity (L/min)")
    if err:
        errors.append(err)

    if (cost_raw is None or str(cost_raw).strip() == "") and length_m is not None:
        cost = calculate_default_routing_cost(length_m)
    else:
        cost, err = _parse_nonneg_float(cost_raw, "Routing cost")
        if err:
            errors.append(err)
        elif require_positive_cost and cost is not None and cost <= 0:
            errors.append("Routing cost must be greater than zero for physical pipes.")

    if status not in ALLOWED_STATUSES:
        errors.append(
            f"Status '{status}' is not allowed. "
            f"Choose one of: {', '.join(sorted(ALLOWED_STATUSES))}."
        )

    return errors, length_m, capacity_lpm, cost


def _get_existing_node_ids(conn: sqlite3.Connection) -> set[int]:
    rows = conn.execute("SELECT id FROM network_nodes").fetchall()
    return {r["id"] for r in rows}


def create_pipe(
    name: str,
    from_node_id: int | None,
    to_node_id: int | None,
    length_m_raw: str,
    capacity_lpm_raw: str,
    cost_raw: str,
    status: str,
    require_positive_cost: bool = False,
) -> int:
    """
    Insert a new pipe. Returns new id.
    Raises ValueError([messages]) on validation failure.
    """
    with _connect() as conn:
        existing = _get_existing_node_ids(conn)
        errors, length_m, capacity_lpm, cost = validate_pipe(
            name, from_node_id, to_node_id,
            length_m_raw, capacity_lpm_raw, cost_raw,
            status, existing,
            require_positive_cost=require_positive_cost,
        )
        if errors:
            raise ValueError(errors)

        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            """
            INSERT INTO pipes
                   (name, from_node_id, to_node_id, length_m,
                    capacity_lpm, cost, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (name.strip(), from_node_id, to_node_id,
             length_m, capacity_lpm, cost, status, now),
        )
        conn.commit()
        return cur.lastrowid


def get_all_pipes() -> list[sqlite3.Row]:
    """
    Return all pipes joined with node names, ordered newest-first.
    Each row includes: all pipes columns + from_node_name, to_node_name.
    """
    with _connect() as conn:
        return conn.execute(
            """
            SELECT p.*,
                   fn.name AS from_node_name,
                   tn.name AS to_node_name
              FROM pipes p
              JOIN network_nodes fn ON fn.id = p.from_node_id
              JOIN network_nodes tn ON tn.id = p.to_node_id
             ORDER BY p.created_at DESC
            """
        ).fetchall()


def get_pipe_by_id(pipe_id: int) -> sqlite3.Row | None:
    """Return a single pipe (with node names) or None."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT p.*,
                   fn.name AS from_node_name,
                   tn.name AS to_node_name
              FROM pipes p
              JOIN network_nodes fn ON fn.id = p.from_node_id
              JOIN network_nodes tn ON tn.id = p.to_node_id
             WHERE p.id = ?
            """,
            (pipe_id,),
        ).fetchone()


def update_pipe(
    pipe_id: int,
    name: str,
    from_node_id: int | None,
    to_node_id: int | None,
    length_m_raw: str,
    capacity_lpm_raw: str,
    cost_raw: str,
    status: str,
    require_positive_cost: bool = False,
) -> None:
    """
    Update an existing pipe.
    Raises ValueError([messages]) on validation failure.
    Raises LookupError if pipe does not exist.
    """
    with _connect() as conn:
        existing = _get_existing_node_ids(conn)
        errors, length_m, capacity_lpm, cost = validate_pipe(
            name, from_node_id, to_node_id,
            length_m_raw, capacity_lpm_raw, cost_raw,
            status, existing,
            require_positive_cost=require_positive_cost,
        )
        if errors:
            raise ValueError(errors)

        result = conn.execute(
            """
            UPDATE pipes
               SET name = ?, from_node_id = ?, to_node_id = ?,
                   length_m = ?, capacity_lpm = ?, cost = ?, status = ?
             WHERE id = ?
            """,
            (name.strip(), from_node_id, to_node_id,
             length_m, capacity_lpm, cost, status, pipe_id),
        )
        conn.commit()
        if result.rowcount == 0:
            raise LookupError(f"Pipe {pipe_id} not found.")


def delete_pipe(pipe_id: int) -> None:
    """
    Delete a pipe.
    Raises LookupError if the pipe does not exist.
    Raises RuntimeError if the pipe is referenced by sensors (FK violation prevention).
    """
    with _connect() as conn:
        sensor_count = conn.execute(
            "SELECT COUNT(*) FROM sensors WHERE pipe_id = ?", (pipe_id,)
        ).fetchone()[0]
        if sensor_count > 0:
            raise RuntimeError(
                f"Cannot delete pipe {pipe_id}: it is still monitored by "
                f"{sensor_count} sensor{'s' if sensor_count != 1 else ''}. "
                f"Delete or reassign those sensors first."
            )

        event_count = conn.execute(
            "SELECT COUNT(*) FROM leak_events WHERE pipe_id = ?", (pipe_id,)
        ).fetchone()[0]
        if event_count > 0:
            raise RuntimeError(
                f"Cannot delete pipe {pipe_id}: it has {event_count} recorded "
                f"leak event{'s' if event_count != 1 else ''}. Resolve or archive those events first."
            )

        result = conn.execute("DELETE FROM pipes WHERE id = ?", (pipe_id,))
        conn.commit()
        if result.rowcount == 0:
            raise LookupError(f"Pipe {pipe_id} not found.")


def count_pipes() -> int:
    """Return the total number of pipes."""
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM pipes").fetchone()[0]


def get_all_pipes_for_graph() -> list[dict]:
    """
    Return all ACTIVE pipes as plain dicts ready for building an adjacency list.

    Each dict has keys: pipe_id, from_node_id, to_node_id, cost, length_m, capacity_lpm.

    Usage (future Dijkstra module)::

        graph: dict[int, list[tuple[int, float, int]]] = {}
        for edge in db.get_all_pipes_for_graph():
            graph.setdefault(edge['from_node_id'], []).append(
                (edge['to_node_id'], edge['cost'], edge['pipe_id'])
            )
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id AS pipe_id, from_node_id, to_node_id,
                   cost, length_m, capacity_lpm
              FROM pipes
             WHERE status = 'active'
             ORDER BY from_node_id, to_node_id
            """
        ).fetchall()
        return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════
#  SENSOR & SENSOR READINGS CRUD
# ═══════════════════════════════════════════════════════════════════════════

def get_pipe_name_map() -> dict[int, str]:
    """Return {pipe_id: display_label} for populating sensor pipe dropdowns."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.name,
                   fn.name AS from_node_name,
                   tn.name AS to_node_name
              FROM pipes p
              JOIN network_nodes fn ON fn.id = p.from_node_id
              JOIN network_nodes tn ON tn.id = p.to_node_id
             ORDER BY p.name
            """
        ).fetchall()
        return {
            row["id"]: f"{row['name']} (#{row['id']} · {row['from_node_name']} → {row['to_node_name']})"
            for row in rows
        }


def _get_existing_pipe_ids(conn: sqlite3.Connection) -> set[int]:
    rows = conn.execute("SELECT id FROM pipes").fetchall()
    return {r["id"] for r in rows}


def validate_sensor(
    sensor_name: str,
    pipe_id: int | None,
    sensor_type: str,
    unit: str,
    status: str,
    existing_pipe_ids: set[int],
) -> list[str]:
    """Validate sensor fields. Returns list of error messages (empty = valid)."""
    errors: list[str] = []

    if not sensor_name or not sensor_name.strip():
        errors.append("Sensor name is required and cannot be blank.")
    elif len(sensor_name.strip()) > 120:
        errors.append("Sensor name must be 120 characters or fewer.")

    if pipe_id is None:
        errors.append("Associated pipe is required.")
    elif pipe_id not in existing_pipe_ids:
        errors.append("Selected pipe does not exist.")

    if sensor_type not in ALLOWED_SENSOR_TYPES:
        errors.append(
            f"Sensor type '{sensor_type}' is not allowed. "
            f"Choose one of: {', '.join(sorted(ALLOWED_SENSOR_TYPES))}."
        )

    if not unit or not unit.strip():
        errors.append("Measurement unit is required.")
    elif len(unit.strip()) > 30:
        errors.append("Measurement unit must be 30 characters or fewer.")

    if status not in ALLOWED_STATUSES:
        errors.append(
            f"Status '{status}' is not allowed. "
            f"Choose one of: {', '.join(sorted(ALLOWED_STATUSES))}."
        )

    return errors


def create_sensor(
    sensor_name: str,
    pipe_id: int | None,
    sensor_type: str,
    unit: str,
    status: str = "active",
) -> int:
    """Insert a new sensor. Returns new sensor_id. Raises ValueError on bad input."""
    with _connect() as conn:
        existing_pipe_ids = _get_existing_pipe_ids(conn)
        errors = validate_sensor(
            sensor_name, pipe_id, sensor_type, unit, status, existing_pipe_ids
        )
        if errors:
            raise ValueError(errors)

        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            """
            INSERT INTO sensors (sensor_name, pipe_id, sensor_type, unit, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (sensor_name.strip(), pipe_id, sensor_type, unit.strip(), status, now),
        )
        conn.commit()
        return cur.lastrowid


def get_all_sensors() -> list[sqlite3.Row]:
    """
    Return all sensors joined with pipe and node names, ordered newest-first.
    Includes: sensor_id, sensor_name, pipe_id, sensor_type, unit, status, created_at,
              pipe_name, from_node_name, to_node_name.
    """
    with _connect() as conn:
        return conn.execute(
            """
            SELECT s.*,
                   p.name AS pipe_name,
                   p.from_node_id,
                   p.to_node_id,
                   fn.name AS from_node_name,
                   tn.name AS to_node_name
              FROM sensors s
              JOIN pipes p ON p.id = s.pipe_id
              JOIN network_nodes fn ON fn.id = p.from_node_id
              JOIN network_nodes tn ON tn.id = p.to_node_id
             ORDER BY s.created_at DESC
            """
        ).fetchall()


def get_sensor_by_id(sensor_id: int) -> sqlite3.Row | None:
    """Return a single sensor (with pipe and node metadata) or None."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT s.*,
                   p.name AS pipe_name,
                   p.from_node_id,
                   p.to_node_id,
                   fn.name AS from_node_name,
                   tn.name AS to_node_name
              FROM sensors s
              JOIN pipes p ON p.id = s.pipe_id
              JOIN network_nodes fn ON fn.id = p.from_node_id
              JOIN network_nodes tn ON tn.id = p.to_node_id
             WHERE s.sensor_id = ?
            """,
            (sensor_id,),
        ).fetchone()


def update_sensor(
    sensor_id: int,
    sensor_name: str,
    pipe_id: int | None,
    sensor_type: str,
    unit: str,
    status: str,
) -> None:
    """Update sensor details. Raises ValueError on bad input, LookupError if not found."""
    with _connect() as conn:
        existing_pipe_ids = _get_existing_pipe_ids(conn)
        errors = validate_sensor(
            sensor_name, pipe_id, sensor_type, unit, status, existing_pipe_ids
        )
        if errors:
            raise ValueError(errors)

        result = conn.execute(
            """
            UPDATE sensors
               SET sensor_name = ?, pipe_id = ?, sensor_type = ?, unit = ?, status = ?
             WHERE sensor_id = ?
            """,
            (sensor_name.strip(), pipe_id, sensor_type, unit.strip(), status, sensor_id),
        )
        conn.commit()
        if result.rowcount == 0:
            raise LookupError(f"Sensor {sensor_id} not found.")


def toggle_sensor_status(sensor_id: int) -> str:
    """Toggle sensor status between 'active' and 'inactive'. Returns the new status."""
    with _connect() as conn:
        sensor = conn.execute(
            "SELECT status FROM sensors WHERE sensor_id = ?", (sensor_id,)
        ).fetchone()
        if sensor is None:
            raise LookupError(f"Sensor {sensor_id} not found.")

        new_status = "inactive" if sensor["status"] == "active" else "active"
        conn.execute(
            "UPDATE sensors SET status = ? WHERE sensor_id = ?",
            (new_status, sensor_id),
        )
        conn.commit()
        return new_status


def delete_sensor(sensor_id: int) -> None:
    """
    Delete a sensor by sensor_id.
    Raises LookupError if the sensor does not exist.
    Raises RuntimeError if the sensor has recorded readings (FK violation prevention).
    """
    with _connect() as conn:
        reading_count = conn.execute(
            "SELECT COUNT(*) FROM sensor_readings WHERE sensor_id = ?", (sensor_id,)
        ).fetchone()[0]
        if reading_count > 0:
            raise RuntimeError(
                f"Cannot delete sensor {sensor_id}: it has {reading_count} recorded "
                f"reading{'s' if reading_count != 1 else ''}. Clear readings first."
            )

        result = conn.execute("DELETE FROM sensors WHERE sensor_id = ?", (sensor_id,))
        conn.commit()
        if result.rowcount == 0:
            raise LookupError(f"Sensor {sensor_id} not found.")


def count_sensors() -> int:
    """Return total number of sensors."""
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM sensors").fetchone()[0]


def count_sensor_readings() -> int:
    """Return total number of sensor readings."""
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]


def record_sensor_reading(
    sensor_id: int,
    reading_value: float,
    recorded_at: str | None = None,
    reading_type: str | None = None,
    unit: str | None = None,
    is_simulated: bool = True,
    scenario: str | None = None,
) -> int:
    """
    Insert a reading into sensor_readings. Returns reading_id.
    Raises LookupError if sensor_id does not exist.
    """
    if scenario is not None and scenario not in ALLOWED_SIMULATION_SCENARIOS:
        raise ValueError(
            f"Invalid simulation scenario '{scenario}'. Allowed scenarios: {', '.join(sorted(ALLOWED_SIMULATION_SCENARIOS))}."
        )

    with _connect() as conn:
        sensor = conn.execute(
            "SELECT sensor_id, sensor_type, unit FROM sensors WHERE sensor_id = ?",
            (sensor_id,),
        ).fetchone()
        if sensor is None:
            raise LookupError(f"Sensor {sensor_id} not found.")

        ts = recorded_at or datetime.now(timezone.utc).isoformat()
        r_type = reading_type or sensor["sensor_type"]
        r_unit = unit or sensor["unit"]
        sim_int = 1 if is_simulated else 0

        cur = conn.execute(
            """
            INSERT INTO sensor_readings (
                sensor_id, reading_value, recorded_at, reading_type, unit, is_simulated, scenario
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (sensor_id, float(reading_value), ts, r_type, r_unit, sim_int, scenario),
        )
        conn.commit()
        return cur.lastrowid


def record_sensor_readings_batch(
    readings: Sequence[Any],
    fail_on_index: int | None = None,
) -> int:
    """
    Atomically insert multiple sensor readings within a single transaction.
    Rolls back completely if any reading is invalid, sensor is missing, or an error occurs.
    fail_on_index is an intentional test hook for verifying atomicity and rollback.

    Returns:
        Count of successfully inserted readings.
    """
    if not readings:
        return 0

    with _connect() as conn:
        sensor_cache: dict[int, sqlite3.Row] = {}

        for idx, r in enumerate(readings):
            if fail_on_index is not None and idx == fail_on_index:
                raise RuntimeError(f"Simulated failure at batch index {idx}")

            if isinstance(r, dict):
                s_id = int(r["sensor_id"])
                val = float(r["reading_value"])
                scen = r.get("scenario")
                ts = r.get("recorded_at") or datetime.now(timezone.utc).isoformat()
                r_type = r.get("reading_type")
                r_unit = r.get("unit")
                is_sim = 1 if r.get("is_simulated", True) else 0
            else:
                s_id = int(r.sensor_id)
                val = float(r.reading_value)
                scen = getattr(r, "simulation_mode", getattr(r, "scenario", None))
                ts = getattr(r, "recorded_at", None) or datetime.now(timezone.utc).isoformat()
                r_type = getattr(r, "sensor_type", getattr(r, "reading_type", None))
                r_unit = getattr(r, "unit", None)
                is_sim = 1 if getattr(r, "is_simulated", True) else 0

            if scen is not None and scen not in ALLOWED_SIMULATION_SCENARIOS:
                raise ValueError(
                    f"Invalid simulation scenario '{scen}'. Allowed scenarios: {', '.join(sorted(ALLOWED_SIMULATION_SCENARIOS))}."
                )

            if s_id not in sensor_cache:
                sensor = conn.execute(
                    "SELECT sensor_id, sensor_type, unit FROM sensors WHERE sensor_id = ?",
                    (s_id,),
                ).fetchone()
                if sensor is None:
                    raise LookupError(f"Sensor {s_id} not found.")
                sensor_cache[s_id] = sensor
            else:
                sensor = sensor_cache[s_id]

            final_type = r_type or sensor["sensor_type"]
            final_unit = r_unit or sensor["unit"]

            conn.execute(
                """
                INSERT INTO sensor_readings (
                    sensor_id, reading_value, recorded_at, reading_type, unit, is_simulated, scenario
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (s_id, val, ts, final_type, final_unit, is_sim, scen),
            )

        conn.commit()
        return len(readings)


def get_readings_for_sensor(sensor_id: int, limit: int = 50) -> list[sqlite3.Row]:
    """Return recent readings for a specific sensor."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT * FROM sensor_readings
             WHERE sensor_id = ?
             ORDER BY recorded_at DESC
             LIMIT ?
            """,
            (sensor_id, limit),
        ).fetchall()


def get_active_sensors(pipe_id: int | None = None) -> list[sqlite3.Row]:
    """
    Return all active sensors joined with pipe and node names.
    If pipe_id is provided, filters to that pipe.
    """
    with _connect() as conn:
        if pipe_id is not None:
            return conn.execute(
                """
                SELECT s.*,
                       p.name AS pipe_name,
                       p.from_node_id,
                       p.to_node_id,
                       fn.name AS from_node_name,
                       tn.name AS to_node_name
                  FROM sensors s
                  JOIN pipes p ON p.id = s.pipe_id
                  JOIN network_nodes fn ON fn.id = p.from_node_id
                  JOIN network_nodes tn ON tn.id = p.to_node_id
                 WHERE s.status = 'active' AND s.pipe_id = ?
                 ORDER BY s.sensor_id ASC
                """,
                (pipe_id,),
            ).fetchall()
        return conn.execute(
            """
            SELECT s.*,
                   p.name AS pipe_name,
                   p.from_node_id,
                   p.to_node_id,
                   fn.name AS from_node_name,
                   tn.name AS to_node_name
              FROM sensors s
              JOIN pipes p ON p.id = s.pipe_id
              JOIN network_nodes fn ON fn.id = p.from_node_id
              JOIN network_nodes tn ON tn.id = p.to_node_id
             WHERE s.status = 'active'
             ORDER BY s.sensor_id ASC
            """
        ).fetchall()


def count_active_sensors() -> int:
    """Return total number of active sensors."""
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM sensors WHERE status = 'active'").fetchone()[0]


def get_latest_sensor_readings(limit: int = 50) -> list[sqlite3.Row]:
    """
    Return recent sensor readings joined with sensor, pipe, and node details,
    ordered newest-first.
    """
    with _connect() as conn:
        return conn.execute(
            """
            SELECT sr.*,
                   s.sensor_name,
                   s.sensor_type,
                   s.pipe_id,
                   p.name AS pipe_name,
                   fn.name AS from_node_name,
                   tn.name AS to_node_name
              FROM sensor_readings sr
              JOIN sensors s ON s.sensor_id = sr.sensor_id
              JOIN pipes p ON p.id = s.pipe_id
              JOIN network_nodes fn ON fn.id = p.from_node_id
              JOIN network_nodes tn ON tn.id = p.to_node_id
             ORDER BY sr.recorded_at DESC, sr.reading_id DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()


def get_latest_reading_by_type(sensor_type: str) -> sqlite3.Row | None:
    """Return the most recently recorded reading for a specific sensor type."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT sr.*, s.sensor_name, s.unit
              FROM sensor_readings sr
              JOIN sensors s ON s.sensor_id = sr.sensor_id
             WHERE s.sensor_type = ?
             ORDER BY sr.recorded_at DESC, sr.reading_id DESC
             LIMIT 1
            """,
            (sensor_type,),
        ).fetchone()


def clear_sensor_readings() -> int:
    """Delete all sensor readings. Returns number of rows deleted."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM sensor_readings")
        conn.commit()
        return cur.rowcount


# ═══════════════════════════════════════════════════════════════════════════
#  LEAK EVENTS CRUD & HELPERS
# ═══════════════════════════════════════════════════════════════════════════

ALLOWED_EVENT_STATUSES = {"open", "investigating", "resolved"}
ALLOWED_EVENT_SEVERITIES = {"low", "medium", "high", "critical"}


def record_leak_event(
    pipe_id: int,
    severity: str,
    evidence_description: str,
    flow_anomaly_score: float | None = None,
    pressure_anomaly_score: float | None = None,
    baseline_values: str | None = None,
    observed_values: str | None = None,
    detected_at: str | None = None,
    is_simulated: bool = True,
) -> tuple[int, bool]:
    """
    Record a detected leak anomaly.
    Prevents duplicate open events for the same pipe by checking for existing
    records with status IN ('open', 'investigating'). If an active event already exists,
    it updates the scores, observed values, and latest detected timestamp.

    Returns:
        (event_id, is_newly_created: bool)
    """
    if severity not in ALLOWED_EVENT_SEVERITIES:
        raise ValueError(f"Invalid severity '{severity}'. Must be one of {ALLOWED_EVENT_SEVERITIES}.")

    now = detected_at or datetime.now(timezone.utc).isoformat()
    sim_int = 1 if is_simulated else 0

    with _connect() as conn:
        # Check pipe existence (FK check)
        pipe = conn.execute("SELECT id FROM pipes WHERE id = ?", (pipe_id,)).fetchone()
        if pipe is None:
            raise LookupError(f"Pipe #{pipe_id} not found.")

        # Check for existing open or investigating event on this pipe
        existing = conn.execute(
            """
            SELECT event_id, severity, status
              FROM leak_events
             WHERE pipe_id = ? AND status IN ('open', 'investigating')
             ORDER BY detected_at DESC
             LIMIT 1
            """,
            (pipe_id,),
        ).fetchone()

        if existing:
            # Update the existing event without creating a duplicate
            event_id = existing["event_id"]
            # Escalate severity if new severity is higher
            cur_sev = existing["severity"]
            rank = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            final_sev = severity if rank.get(severity, 0) > rank.get(cur_sev, 0) else cur_sev

            conn.execute(
                """
                UPDATE leak_events
                   SET severity = ?,
                       flow_anomaly_score = COALESCE(?, flow_anomaly_score),
                       pressure_anomaly_score = COALESCE(?, pressure_anomaly_score),
                       baseline_values = COALESCE(?, baseline_values),
                       observed_values = COALESCE(?, observed_values),
                       evidence_description = ?,
                       detected_at = ?
                 WHERE event_id = ?
                """,
                (
                    final_sev,
                    flow_anomaly_score,
                    pressure_anomaly_score,
                    baseline_values,
                    observed_values,
                    evidence_description,
                    now,
                    event_id,
                ),
            )
            conn.commit()
            return (event_id, False)

        # Protect against stale telemetry re-triggering an event on a resolved pipe
        latest_resolved = conn.execute(
            """
            SELECT event_id, resolved_at, detected_at
              FROM leak_events
             WHERE pipe_id = ? AND status = 'resolved'
             ORDER BY COALESCE(resolved_at, detected_at) DESC, event_id DESC
             LIMIT 1
            """,
            (pipe_id,),
        ).fetchone()

        if latest_resolved:
            res_at = latest_resolved["resolved_at"] or latest_resolved["detected_at"]
            if res_at:
                # If detected_at is before or equal to resolved_at, do not create a duplicate
                if not _is_strictly_after(now, res_at):
                    return (latest_resolved["event_id"], False)

                # If sensor readings exist on this pipe, ensure at least one reading occurred strictly after resolved_at
                latest_reading = conn.execute(
                    """
                    SELECT MAX(sr.recorded_at) AS max_rec
                      FROM sensor_readings sr
                      JOIN sensors s ON s.sensor_id = sr.sensor_id
                     WHERE s.pipe_id = ?
                    """,
                    (pipe_id,),
                ).fetchone()
                if latest_reading and latest_reading["max_rec"]:
                    if not _is_strictly_after(latest_reading["max_rec"], res_at):
                        return (latest_resolved["event_id"], False)

        # Insert new event
        cur = conn.execute(
            """
            INSERT INTO leak_events (
                pipe_id, detected_at, severity, status,
                flow_anomaly_score, pressure_anomaly_score,
                baseline_values, observed_values, evidence_description,
                is_simulated
            )
            VALUES (?, ?, ?, 'open', ?, ?, ?, ?, ?, ?)
            """,
            (
                pipe_id,
                now,
                severity,
                flow_anomaly_score,
                pressure_anomaly_score,
                baseline_values,
                observed_values,
                evidence_description,
                sim_int,
            ),
        )
        event_id = cur.lastrowid
        record_lifecycle_audit(
            conn=conn,
            entity_type="leak_event",
            entity_id=event_id,
            previous_status=None,
            new_status="open",
            action_description=f"Leak anomaly detected on pipe #{pipe_id} ({severity.upper()})",
            leak_event_id=event_id,
            pipe_id=pipe_id,
            changed_at=now,
        )
        conn.commit()
        return (event_id, True)


def get_all_leak_events(limit: int = 100) -> list[sqlite3.Row]:
    """Return all leak events joined with pipe and node names, ordered newest-first."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT le.*,
                   p.name AS pipe_name,
                   p.from_node_id,
                   p.to_node_id,
                   fn.name AS from_node_name,
                   tn.name AS to_node_name
              FROM leak_events le
              JOIN pipes p ON p.id = le.pipe_id
              JOIN network_nodes fn ON fn.id = p.from_node_id
              JOIN network_nodes tn ON tn.id = p.to_node_id
             ORDER BY le.detected_at DESC, le.event_id DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()


def get_open_leak_events() -> list[sqlite3.Row]:
    """Return open or investigating leak events ordered by severity and date."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT le.*,
                   p.name AS pipe_name,
                   p.from_node_id,
                   p.to_node_id,
                   fn.name AS from_node_name,
                   tn.name AS to_node_name
              FROM leak_events le
              JOIN pipes p ON p.id = le.pipe_id
              JOIN network_nodes fn ON fn.id = p.from_node_id
              JOIN network_nodes tn ON tn.id = p.to_node_id
             WHERE le.status IN ('open', 'investigating')
             ORDER BY
                 CASE le.severity
                     WHEN 'critical' THEN 1
                     WHEN 'high' THEN 2
                     WHEN 'medium' THEN 3
                     WHEN 'low' THEN 4
                     ELSE 5
                 END,
                 le.detected_at DESC
            """
        ).fetchall()


def get_leak_event_by_id(event_id: int) -> sqlite3.Row | None:
    """Return a single leak event or None."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT le.*,
                   p.name AS pipe_name,
                   p.from_node_id,
                   p.to_node_id,
                   fn.name AS from_node_name,
                   tn.name AS to_node_name
              FROM leak_events le
              JOIN pipes p ON p.id = le.pipe_id
              JOIN network_nodes fn ON fn.id = p.from_node_id
              JOIN network_nodes tn ON tn.id = p.to_node_id
             WHERE le.event_id = ?
            """,
            (event_id,),
        ).fetchone()


def get_latest_resolved_leak_event(pipe_id: int) -> sqlite3.Row | None:
    """Return the most recently resolved leak event for the given pipe, or None."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT *
              FROM leak_events
             WHERE pipe_id = ? AND status = 'resolved'
             ORDER BY COALESCE(resolved_at, detected_at) DESC, event_id DESC
             LIMIT 1
            """,
            (pipe_id,),
        ).fetchone()


def update_leak_event_status(
    event_id: int,
    new_status: str,
    action_description: str | None = None,
) -> bool:
    """
    Update event status to 'open', 'investigating', or 'resolved'.
    If 'resolved', sets resolved_at timestamp.
    Records a corresponding lifecycle audit entry in the same transaction.
    """
    if new_status not in ALLOWED_EVENT_STATUSES:
        raise ValueError(
            f"Invalid event status '{new_status}'. Choose one of: {', '.join(sorted(ALLOWED_EVENT_STATUSES))}."
        )

    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        event = conn.execute(
            "SELECT event_id, pipe_id, status FROM leak_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if event is None:
            raise LookupError(f"Leak event #{event_id} not found.")

        prev_status = event["status"]
        pipe_id = event["pipe_id"]

        if prev_status == new_status:
            return True

        resolved_at = now if new_status == "resolved" else None
        conn.execute(
            """
            UPDATE leak_events
               SET status = ?,
                   resolved_at = CASE WHEN ? = 'resolved' THEN ? ELSE resolved_at END
             WHERE event_id = ?
            """,
            (new_status, new_status, resolved_at, event_id),
        )

        desc = action_description or f"Leak event #{event_id} status updated from '{prev_status}' to '{new_status}'"
        record_lifecycle_audit(
            conn=conn,
            entity_type="leak_event",
            entity_id=event_id,
            previous_status=prev_status,
            new_status=new_status,
            action_description=desc,
            leak_event_id=event_id,
            pipe_id=pipe_id,
            changed_at=now,
        )
        conn.commit()
        return True


def count_open_leak_events() -> int:
    """Return total count of open or investigating leak events."""
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM leak_events WHERE status IN ('open', 'investigating')"
        ).fetchone()[0]


def count_pipes_under_investigation() -> int:
    """Return count of distinct pipes currently flagged with open or investigating events."""
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(DISTINCT pipe_id) FROM leak_events WHERE status IN ('open', 'investigating')"
        ).fetchone()[0]


def count_total_leak_events() -> int:
    """Return total count of all leak events ever recorded."""
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM leak_events").fetchone()[0]


# ═══════════════════════════════════════════════════════════════════════════
#  REPAIR TASKS CRUD & HELPERS (SMART REPAIR PRIORITY QUEUE)
# ═══════════════════════════════════════════════════════════════════════════

def create_or_update_repair_task(
    event_id: int,
    pipe_id: int,
    priority_score: float,
    priority_category: str,
    score_breakdown: str | None = None,
    assigned_to: str | None = None,
) -> tuple[int, bool]:
    """
    Create a new repair task for a leak event, or update its priority score if already exists.
    Maintains the existing task status (e.g. In Progress) if the task already exists.
    Returns (task_id, is_new: bool).
    """
    if priority_category not in ALLOWED_REPAIR_CATEGORIES:
        raise ValueError(
            f"Invalid priority category '{priority_category}'. Must be one of: {', '.join(sorted(ALLOWED_REPAIR_CATEGORIES))}."
        )

    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        # Check event existence
        ev = conn.execute("SELECT event_id, pipe_id FROM leak_events WHERE event_id = ?", (event_id,)).fetchone()
        if ev is None:
            raise LookupError(f"Leak event #{event_id} not found.")

        existing = conn.execute("SELECT task_id, status FROM repair_tasks WHERE event_id = ?", (event_id,)).fetchone()
        if existing:
            task_id = existing["task_id"]
            conn.execute(
                """
                UPDATE repair_tasks
                   SET priority_score = ?,
                       priority_category = ?,
                       score_breakdown = COALESCE(?, score_breakdown),
                       updated_at = ?
                 WHERE task_id = ?
                """,
                (float(priority_score), priority_category, score_breakdown, now, task_id),
            )
            conn.commit()
            return (task_id, False)

        try:
            cur = conn.execute(
                """
                INSERT INTO repair_tasks (
                    event_id, pipe_id, priority_score, priority_category,
                    status, score_breakdown, assigned_to, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, 'Pending', ?, ?, ?, ?)
                """,
                (
                    event_id,
                    pipe_id,
                    float(priority_score),
                    priority_category,
                    score_breakdown,
                    assigned_to,
                    now,
                    now,
                ),
            )
            task_id = cur.lastrowid
            record_lifecycle_audit(
                conn=conn,
                entity_type="repair_task",
                entity_id=task_id,
                previous_status=None,
                new_status="Pending",
                action_description=f"Repair task #{task_id} initialized from Leak Event #{event_id}",
                repair_task_id=task_id,
                leak_event_id=event_id,
                pipe_id=pipe_id,
                changed_at=now,
            )
            conn.commit()
            return (task_id, True)
        except sqlite3.IntegrityError as exc:
            # Race condition: another thread/worker inserted the task concurrently
            if "UNIQUE" in str(exc):
                conn.rollback()
                existing = conn.execute(
                    "SELECT task_id FROM repair_tasks WHERE event_id = ?",
                    (event_id,),
                ).fetchone()
                if existing:
                    return (existing["task_id"], False)
            raise


def update_repair_task_status(
    task_id: int,
    new_status: str,
    action_description: str | None = None,
) -> bool:
    """
    Update the lifecycle status of a repair task ('Pending', 'In Progress', 'Completed')
    and record a corresponding lifecycle audit entry in the same transaction.
    """
    if new_status not in ALLOWED_REPAIR_STATUSES:
        raise ValueError(
            f"Invalid repair status '{new_status}'. Choose one of: {', '.join(sorted(ALLOWED_REPAIR_STATUSES))}."
        )

    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        task = conn.execute(
            """
            SELECT task_id, event_id, pipe_id, status
              FROM repair_tasks
             WHERE task_id = ?
            """,
            (task_id,),
        ).fetchone()
        if task is None:
            raise LookupError(f"Repair task #{task_id} not found.")

        prev_status = task["status"]
        pipe_id = task["pipe_id"]
        event_id = task["event_id"]

        if prev_status == new_status:
            return True

        conn.execute(
            """
            UPDATE repair_tasks
               SET status = ?,
                   updated_at = ?
             WHERE task_id = ?
            """,
            (new_status, now, task_id),
        )

        desc = action_description or f"Repair task #{task_id} status updated from '{prev_status}' to '{new_status}'"
        record_lifecycle_audit(
            conn=conn,
            entity_type="repair_task",
            entity_id=task_id,
            previous_status=prev_status,
            new_status=new_status,
            action_description=desc,
            repair_task_id=task_id,
            leak_event_id=event_id,
            pipe_id=pipe_id,
            changed_at=now,
        )
        conn.commit()
        return True


def get_all_repair_tasks() -> list[sqlite3.Row]:
    """Return all repair tasks joined with leak events, pipes, and node names."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT rt.*,
                   le.severity,
                   le.detected_at,
                   le.flow_anomaly_score,
                   le.pressure_anomaly_score,
                   le.observed_values,
                   le.baseline_values,
                   le.evidence_description,
                   le.is_simulated,
                   p.name AS pipe_name,
                   p.from_node_id,
                   p.to_node_id,
                   p.capacity_lpm,
                   fn.name AS from_node_name,
                   tn.name AS to_node_name
              FROM repair_tasks rt
              JOIN leak_events le ON le.event_id = rt.event_id
              JOIN pipes p ON p.id = rt.pipe_id
              JOIN network_nodes fn ON fn.id = p.from_node_id
              JOIN network_nodes tn ON tn.id = p.to_node_id
             ORDER BY rt.priority_score DESC, rt.created_at ASC, rt.task_id ASC
            """
        ).fetchall()


def get_active_repair_tasks() -> list[sqlite3.Row]:
    """Return active repair tasks (Pending or In Progress)."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT rt.*,
                   le.severity,
                   le.detected_at,
                   le.flow_anomaly_score,
                   le.pressure_anomaly_score,
                   le.observed_values,
                   le.baseline_values,
                   le.evidence_description,
                   le.is_simulated,
                   p.name AS pipe_name,
                   p.from_node_id,
                   p.to_node_id,
                   p.capacity_lpm,
                   fn.name AS from_node_name,
                   tn.name AS to_node_name
              FROM repair_tasks rt
              JOIN leak_events le ON le.event_id = rt.event_id
              JOIN pipes p ON p.id = rt.pipe_id
              JOIN network_nodes fn ON fn.id = p.from_node_id
              JOIN network_nodes tn ON tn.id = p.to_node_id
             WHERE rt.status IN ('Pending', 'In Progress')
             ORDER BY rt.priority_score DESC, rt.created_at ASC, rt.task_id ASC
            """
        ).fetchall()


def get_repair_task_by_id(task_id: int) -> sqlite3.Row | None:
    """Return a single repair task or None."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT rt.*,
                   le.severity,
                   le.detected_at,
                   le.flow_anomaly_score,
                   le.pressure_anomaly_score,
                   le.observed_values,
                   le.baseline_values,
                   le.evidence_description,
                   le.is_simulated,
                   p.name AS pipe_name,
                   p.from_node_id,
                   p.to_node_id,
                   p.capacity_lpm,
                   fn.name AS from_node_name,
                   tn.name AS to_node_name
              FROM repair_tasks rt
              JOIN leak_events le ON le.event_id = rt.event_id
              JOIN pipes p ON p.id = rt.pipe_id
              JOIN network_nodes fn ON fn.id = p.from_node_id
              JOIN network_nodes tn ON tn.id = p.to_node_id
             WHERE rt.task_id = ?
            """,
            (task_id,),
        ).fetchone()


def get_repair_task_by_event_id(event_id: int) -> sqlite3.Row | None:
    """Return the repair task associated with a specific leak event, or None."""
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM repair_tasks WHERE event_id = ?",
            (event_id,),
        ).fetchone()


def count_pending_repair_tasks() -> int:
    """Return count of repair tasks with status 'Pending'."""
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM repair_tasks WHERE status = 'Pending'"
        ).fetchone()[0]


def count_active_repair_tasks() -> int:
    """Return count of active repair tasks (Pending or In Progress)."""
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM repair_tasks WHERE status IN ('Pending', 'In Progress')"
        ).fetchone()[0]


def count_total_repair_tasks() -> int:
    """Return count of all repair tasks recorded."""
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM repair_tasks").fetchone()[0]


def count_repair_tasks_by_category() -> dict[str, int]:
    """Return a dictionary mapping priority category to active task count."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT priority_category, COUNT(*) as cnt
              FROM repair_tasks
             WHERE status IN ('Pending', 'In Progress')
             GROUP BY priority_category
            """
        ).fetchall()
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for r in rows:
            if r["priority_category"] in counts:
                counts[r["priority_category"]] = r["cnt"]
        return counts


# ═══════════════════════════════════════════════════════════════════════════
#  LIFECYCLE AUDIT LOG HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def record_lifecycle_audit(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_id: int,
    previous_status: str | None,
    new_status: str,
    action_description: str,
    repair_task_id: int | None = None,
    leak_event_id: int | None = None,
    pipe_id: int | None = None,
    changed_at: str | None = None,
) -> int | None:
    """
    Record a lifecycle status transition inside an existing connection/transaction.
    Guarantees:
      - previous_status == new_status creates NO audit entry (skip redundant).
      - Returns audit_id of inserted row, or None if skipped.
    """
    if previous_status == new_status:
        return None

    now = changed_at or datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """
        INSERT INTO lifecycle_audit_log (
            entity_type, entity_id, repair_task_id, leak_event_id, pipe_id,
            previous_status, new_status, action_description, changed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entity_type,
            entity_id,
            repair_task_id,
            leak_event_id,
            pipe_id,
            previous_status,
            new_status,
            action_description,
            now,
        ),
    )
    return cur.lastrowid


def get_lifecycle_audit_logs(
    limit: int = 100,
    entity_type: str | None = None,
    entity_id: int | None = None,
) -> list[sqlite3.Row]:
    """
    Return lifecycle transition audit logs ordered newest-first (audit_id DESC).
    Gracefully handles deleted parent tasks or events by preserving pipe_id.
    """
    with _connect() as conn:
        query = """
            SELECT l.*,
                   COALESCE(p.name, p1.name, p2.name, 'N/A') AS pipe_name
              FROM lifecycle_audit_log l
              LEFT JOIN pipes p ON p.id = l.pipe_id
              LEFT JOIN repair_tasks rt ON rt.task_id = l.repair_task_id
              LEFT JOIN pipes p1 ON p1.id = rt.pipe_id
              LEFT JOIN leak_events le ON le.event_id = l.leak_event_id
              LEFT JOIN pipes p2 ON p2.id = le.pipe_id
        """
        params: list[Any] = []
        clauses: list[str] = []
        if entity_type:
            clauses.append("l.entity_type = ?")
            params.append(entity_type)
        if entity_id:
            clauses.append("l.entity_id = ?")
            params.append(entity_id)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        query += " ORDER BY l.audit_id DESC LIMIT ?"
        params.append(limit)

        return conn.execute(query, tuple(params)).fetchall()


def count_lifecycle_audit_logs() -> int:
    """Return total count of recorded lifecycle audit entries."""
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM lifecycle_audit_log").fetchone()[0]


# ═══════════════════════════════════════════════════════════════════════════
#  FAIR WATER ALLOCATION CRUD & PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════

def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Helper to access fields from either dataclass/object attributes or dict keys."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def validate_allocation_zone(
    zone_name: str,
    requested_demand_l: float | int,
    essential_min_l: float | int = 0.0,
    fairness_weight: float | int = 1.0,
    priority_category: str = "standard",
    status: str = "active",
    node_id: int | None = None,
) -> list[str]:
    """Validate zone parameters. Return empty list if valid, or list of error messages."""
    errors: list[str] = []
    if not zone_name or not str(zone_name).strip():
        errors.append("Zone name is required and cannot be blank.")
    elif len(str(zone_name).strip()) > 120:
        errors.append("Zone name must be 120 characters or fewer.")

    # Validate requested demand
    req_d: float | None = None
    try:
        req_d = float(requested_demand_l)
        if req_d < 0:
            errors.append("Requested demand must be zero or greater.")
    except (TypeError, ValueError):
        errors.append("Requested demand must be a valid number.")

    # Validate essential minimum
    try:
        ess_m = float(essential_min_l)
        if ess_m < 0:
            errors.append("Essential minimum must be zero or greater.")
        elif req_d is not None and ess_m > req_d:
            errors.append("Essential minimum cannot exceed requested demand.")
    except (TypeError, ValueError):
        errors.append("Essential minimum must be a valid number.")

    # Validate fairness weight
    try:
        fw = float(fairness_weight)
        if fw <= 0:
            errors.append("Fairness weight must be strictly greater than zero.")
    except (TypeError, ValueError):
        errors.append("Fairness weight must be a valid number.")

    if priority_category not in ALLOWED_ZONE_CATEGORIES:
        errors.append(
            f"Priority category '{priority_category}' is invalid. "
            f"Must be one of: {', '.join(sorted(ALLOWED_ZONE_CATEGORIES))}."
        )

    if status not in ALLOWED_ZONE_STATUSES:
        errors.append(
            f"Status '{status}' is invalid. "
            f"Must be one of: {', '.join(sorted(ALLOWED_ZONE_STATUSES))}."
        )

    if node_id is not None:
        try:
            node = get_node_by_id(int(node_id))
            if node is None:
                errors.append(f"Referenced network node {node_id} does not exist.")
        except Exception:
            errors.append(f"Invalid network node reference {node_id}.")

    return errors


def create_allocation_zone(
    zone_name: str,
    requested_demand_l: float | int,
    essential_min_l: float | int = 0.0,
    fairness_weight: float | int = 1.0,
    priority_category: str = "standard",
    status: str = "active",
    node_id: int | None = None,
    description: str | None = None,
) -> int:
    """Insert a new water allocation zone. Returns zone_id."""
    errors = validate_allocation_zone(
        zone_name=zone_name,
        requested_demand_l=requested_demand_l,
        essential_min_l=essential_min_l,
        fairness_weight=fairness_weight,
        priority_category=priority_category,
        status=status,
        node_id=node_id,
    )
    if errors:
        raise ValueError("; ".join(errors))

    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO allocation_zones (
                zone_name, node_id, priority_category, requested_demand_l,
                essential_min_l, fairness_weight, status, description,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(zone_name).strip(),
                int(node_id) if node_id is not None else None,
                priority_category,
                float(requested_demand_l),
                float(essential_min_l),
                float(fairness_weight),
                status,
                description.strip() if description else None,
                now,
                now,
            ),
        )
        conn.commit()
        return cur.lastrowid


def update_allocation_zone(
    zone_id: int,
    zone_name: str,
    requested_demand_l: float | int,
    essential_min_l: float | int = 0.0,
    fairness_weight: float | int = 1.0,
    priority_category: str = "standard",
    status: str = "active",
    node_id: int | None = None,
    description: str | None = None,
) -> None:
    """Update an existing allocation zone."""
    errors = validate_allocation_zone(
        zone_name=zone_name,
        requested_demand_l=requested_demand_l,
        essential_min_l=essential_min_l,
        fairness_weight=fairness_weight,
        priority_category=priority_category,
        status=status,
        node_id=node_id,
    )
    if errors:
        raise ValueError("; ".join(errors))

    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        existing = conn.execute(
            "SELECT zone_id FROM allocation_zones WHERE zone_id = ?", (zone_id,)
        ).fetchone()
        if not existing:
            raise KeyError(f"Allocation zone {zone_id} does not exist.")

        conn.execute(
            """
            UPDATE allocation_zones
               SET zone_name = ?,
                   node_id = ?,
                   priority_category = ?,
                   requested_demand_l = ?,
                   essential_min_l = ?,
                   fairness_weight = ?,
                   status = ?,
                   description = ?,
                   updated_at = ?
             WHERE zone_id = ?
            """,
            (
                str(zone_name).strip(),
                int(node_id) if node_id is not None else None,
                priority_category,
                float(requested_demand_l),
                float(essential_min_l),
                float(fairness_weight),
                status,
                description.strip() if description else None,
                now,
                zone_id,
            ),
        )
        conn.commit()


def delete_allocation_zone(zone_id: int) -> None:
    """
    Delete an allocation zone.
    Raises RuntimeError if the zone is referenced by historical allocations.
    """
    with _connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM water_allocations WHERE zone_id = ?", (zone_id,)
        ).fetchone()[0]
        if count > 0:
            raise RuntimeError(
                f"Cannot delete allocation zone {zone_id}: it is referenced in {count} historical allocation records."
            )
        cur = conn.execute("DELETE FROM allocation_zones WHERE zone_id = ?", (zone_id,))
        if cur.rowcount == 0:
            raise KeyError(f"Allocation zone {zone_id} does not exist.")
        conn.commit()


def get_all_allocation_zones(status: str | None = None) -> list[sqlite3.Row]:
    """Return all allocation zones with attached node name if available."""
    with _connect() as conn:
        query = """
            SELECT az.*, nn.name AS node_name
              FROM allocation_zones az
              LEFT JOIN network_nodes nn ON nn.id = az.node_id
        """
        params: list[Any] = []
        if status:
            query += " WHERE az.status = ?"
            params.append(status)
        query += " ORDER BY az.zone_name ASC, az.zone_id ASC"
        return conn.execute(query, tuple(params)).fetchall()


def get_allocation_zone_by_id(zone_id: int) -> sqlite3.Row | None:
    """Return an allocation zone by its ID, or None if not found."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT az.*, nn.name AS node_name
              FROM allocation_zones az
              LEFT JOIN network_nodes nn ON nn.id = az.node_id
             WHERE az.zone_id = ?
            """,
            (zone_id,),
        ).fetchone()


def get_active_allocation_zones() -> list[sqlite3.Row]:
    """Return all active allocation zones."""
    return get_all_allocation_zones(status="active")


def count_allocation_zones(status: str | None = None) -> int:
    """Return count of allocation zones, optionally filtered by status."""
    with _connect() as conn:
        if status:
            return conn.execute(
                "SELECT COUNT(*) FROM allocation_zones WHERE status = ?", (status,)
            ).fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM allocation_zones").fetchone()[0]


def record_allocation_run(
    run_result: Any,
    period_start: str | None = None,
    period_end: str | None = None,
    notes: str | None = None,
    fail_on_zone_id: int | None = None,
) -> int:
    """
    Atomically record an allocation run and all its per-zone allocations.
    Rolls back completely if an error occurs during execution.
    fail_on_zone_id provides an intentional hook for testing atomicity and rollback.
    """
    now = datetime.now(timezone.utc).isoformat()
    allocations = _get_field(run_result, "allocations", [])

    with _connect() as conn:
        # 1. Insert allocation run summary
        sup = _get_field(run_result, "total_supply_l", _get_field(run_result, "total_supply", 0.0))
        dem = _get_field(run_result, "total_demand_l", _get_field(run_result, "total_demand", 0.0))
        alloc_tot = _get_field(run_result, "total_allocated_l", _get_field(run_result, "total_allocated", 0.0))
        unmet_tot = _get_field(run_result, "total_unmet_l", _get_field(run_result, "total_unmet", 0.0))
        sat = _get_field(run_result, "satisfaction_rate_pct", _get_field(run_result, "satisfaction_pct", 0.0))
        pol = _get_field(run_result, "policy_applied", "UNKNOWN")

        cur = conn.execute(
            """
            INSERT INTO allocation_runs (
                run_timestamp, total_available_supply_l, total_requested_demand_l,
                total_allocated_l, total_unmet_l, satisfaction_rate_pct, allocation_policy,
                active_zone_count, period_start, period_end, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                float(sup),
                float(dem),
                float(alloc_tot),
                float(unmet_tot),
                float(sat),
                str(pol),
                len(allocations),
                period_start,
                period_end,
                notes,
            ),
        )
        run_id = cur.lastrowid

        # 2. Insert per-zone allocations
        for alloc in allocations:
            zone_id = _get_field(alloc, "zone_id")
            if fail_on_zone_id is not None and int(zone_id) == int(fail_on_zone_id):
                raise RuntimeError(
                    f"Simulated transaction failure triggered for zone_id={fail_on_zone_id}"
                )

            z_name = _get_field(alloc, "zone_name_snapshot", _get_field(alloc, "zone_name", f"Zone-{zone_id}"))
            req_d = _get_field(alloc, "requested_demand_l", _get_field(alloc, "requested_demand", 0.0))
            ess_m = _get_field(alloc, "essential_min_l", _get_field(alloc, "essential_min", 0.0))
            fw = _get_field(alloc, "fairness_weight", 1.0)
            a_amt = _get_field(alloc, "allocated_amount_l", _get_field(alloc, "allocated_amount", 0.0))
            unm = _get_field(alloc, "unmet_demand_l", _get_field(alloc, "unmet_demand", 0.0))
            sat_p = _get_field(alloc, "satisfaction_rate_pct", _get_field(alloc, "satisfaction_pct", 0.0))
            bot = 1 if _get_field(alloc, "is_bottleneck", False) else 0
            tier = int(_get_field(alloc, "tier_reached", 1))

            # Destination node snapshot at allocation time
            dest_nid = _get_field(alloc, "destination_node_id", _get_field(alloc, "node_id", None))
            if dest_nid is None and zone_id is not None:
                z_row = conn.execute(
                    "SELECT node_id FROM allocation_zones WHERE zone_id = ?",
                    (int(zone_id),),
                ).fetchone()
                if z_row and z_row["node_id"] is not None:
                    dest_nid = int(z_row["node_id"])

            conn.execute(
                """
                INSERT INTO water_allocations (
                    run_id, zone_id, zone_name_snapshot, requested_demand_l,
                    essential_min_l, fairness_weight, allocated_amount_l,
                    unmet_demand_l, satisfaction_rate_pct, is_bottleneck,
                    tier_reached, recorded_at, destination_node_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    int(zone_id),
                    str(z_name),
                    float(req_d),
                    float(ess_m),
                    float(fw),
                    float(a_amt),
                    float(unm),
                    float(sat_p),
                    bot,
                    tier,
                    now,
                    int(dest_nid) if dest_nid is not None else None,
                ),
            )

        conn.commit()
        return run_id


def get_all_allocation_runs(limit: int = 50) -> list[sqlite3.Row]:
    """Return recorded allocation runs, most recent first."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT * FROM allocation_runs
             ORDER BY run_timestamp DESC, run_id DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()


def get_allocation_run_by_id(run_id: int) -> sqlite3.Row | None:
    """Return an allocation run by its ID, or None if not found."""
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM allocation_runs WHERE run_id = ?", (run_id,)
        ).fetchone()


def get_latest_allocation_run() -> sqlite3.Row | None:
    """Return the most recent allocation run, or None if no runs exist."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT * FROM allocation_runs
             ORDER BY run_timestamp DESC, run_id DESC
             LIMIT 1
            """
        ).fetchone()


def get_allocations_for_run(run_id: int) -> list[sqlite3.Row]:
    """Return all per-zone allocation records for a specific run."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT wa.*, az.priority_category, az.status AS current_zone_status
              FROM water_allocations wa
              LEFT JOIN allocation_zones az ON az.zone_id = wa.zone_id
             WHERE wa.run_id = ?
             ORDER BY wa.allocation_id ASC
            """,
            (run_id,),
        ).fetchall()


def count_allocation_runs() -> int:
    """Return total count of recorded allocation runs."""
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM allocation_runs").fetchone()[0]


def delete_allocation_run(run_id: int) -> None:
    """
    Delete an allocation run.
    Raises RuntimeError if the run has associated water_allocations records (RESTRICT)
    or delivery_assessments records (RESTRICT).
    """
    with _connect() as conn:
        assess_count = conn.execute(
            "SELECT COUNT(*) FROM delivery_assessments WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        if assess_count > 0:
            raise RuntimeError(
                f"Cannot delete allocation run {run_id}: it has {assess_count} delivery assessment records attached."
            )
        count = conn.execute(
            "SELECT COUNT(*) FROM water_allocations WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        if count > 0:
            raise RuntimeError(
                f"Cannot delete allocation run {run_id}: it has {count} per-zone allocation records attached."
            )
        cur = conn.execute("DELETE FROM allocation_runs WHERE run_id = ?", (run_id,))
        if cur.rowcount == 0:
            raise KeyError(f"Allocation run {run_id} does not exist.")
        conn.commit()


def count_water_allocations() -> int:
    """Return total count of per-zone water allocation records."""
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM water_allocations").fetchone()[0]


# ═══════════════════════════════════════════════════════════════════════════
#  NETWORK-AWARE WATER DISTRIBUTION ASSESSMENT (Day 9)
# ═══════════════════════════════════════════════════════════════════════════

def create_delivery_assessment(
    run_id: int,
    source_node_id: int | None,
    source_name_snapshot: str,
    duration_minutes: float | int,
    routing_strategy: str,
    total_allocated_l: float | int,
    total_deliverable_l: float | int,
    total_undelivered_l: float | int,
    delivery_efficiency_pct: float | int,
    has_contention: int = 0,
    contention_warnings_json: str | None = None,
    notes: str | None = None,
    assessed_at: str | None = None,
) -> int:
    """
    Insert a standalone delivery assessment record.
    Returns assessment_id.
    """
    strat = str(routing_strategy).strip().upper()
    if strat not in ALLOWED_ROUTING_STRATEGIES:
        raise ValueError(
            f"Routing strategy '{routing_strategy}' is invalid. "
            f"Must be one of: {', '.join(sorted(ALLOWED_ROUTING_STRATEGIES))}."
        )

    try:
        dur = float(duration_minutes)
        if dur <= 0 or math.isnan(dur) or math.isinf(dur):
            raise ValueError()
    except Exception:
        raise ValueError("Delivery duration must be strictly positive and finite.")

    tot_alloc = float(total_allocated_l)
    tot_deliv = float(total_deliverable_l)
    tot_undeliv = float(total_undelivered_l)
    eff = float(delivery_efficiency_pct)

    for name, val in [("total_allocated_l", tot_alloc), ("total_deliverable_l", tot_deliv), ("total_undelivered_l", tot_undeliv)]:
        if val < 0 or math.isnan(val) or math.isinf(val):
            raise ValueError(f"{name} must be non-negative and finite (got {val}).")

    if eff < 0 or eff > 100 or math.isnan(eff) or math.isinf(eff):
        raise ValueError(f"Delivery efficiency percentage must be between 0 and 100 (got {eff}).")

    now = assessed_at or datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        run = conn.execute("SELECT run_id FROM allocation_runs WHERE run_id = ?", (int(run_id),)).fetchone()
        if run is None:
            raise KeyError(f"Referenced allocation run #{run_id} does not exist.")

        if source_node_id is not None:
            src = conn.execute("SELECT id, status FROM network_nodes WHERE id = ?", (int(source_node_id),)).fetchone()
            if src is None:
                raise ValueError(f"Referenced source node #{source_node_id} does not exist.")
            if src["status"] != "active":
                raise ValueError(f"Referenced source node #{source_node_id} is inactive.")

        cur = conn.execute(
            """
            INSERT INTO delivery_assessments (
                run_id, source_node_id, source_name_snapshot, duration_minutes,
                routing_strategy, total_allocated_l, total_deliverable_l,
                total_undelivered_l, delivery_efficiency_pct, has_contention,
                contention_warnings_json, notes, assessed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(run_id),
                int(source_node_id) if source_node_id is not None else None,
                str(source_name_snapshot).strip(),
                dur,
                strat,
                tot_alloc,
                tot_deliv,
                tot_undeliv,
                eff,
                1 if has_contention else 0,
                contention_warnings_json,
                notes,
                now,
            ),
        )
        conn.commit()
        return cur.lastrowid


def create_delivery_zone_assessment(
    assessment_id: int,
    zone_id: int | None,
    zone_name_snapshot: str,
    destination_node_id: int | None,
    destination_name_snapshot: str | None,
    allocated_amount_l: float | int,
    deliverable_amount_l: float | int,
    undelivered_amount_l: float | int,
    required_rate_lpm: float | int,
    feasibility_status: str,
    explanation: str,
    bottleneck_capacity_lpm: float | None = None,
    bottleneck_pipe_id: int | None = None,
    bottleneck_pipe_name_snapshot: str | None = None,
    route_cost: float | None = None,
    route_node_ids: list[int] | str | None = None,
    route_pipe_ids: list[int] | str | None = None,
    delivery_efficiency_pct: float | int = 0.0,
) -> int:
    """
    Insert a single zone delivery assessment detail row.
    Returns zone_assessment_id.
    """
    status = str(feasibility_status).strip().upper()
    if status not in ALLOWED_FEASIBILITY_STATUSES:
        raise ValueError(
            f"Feasibility status '{feasibility_status}' is invalid. "
            f"Must be one of: {', '.join(sorted(ALLOWED_FEASIBILITY_STATUSES))}."
        )

    alloc_amt = float(allocated_amount_l)
    deliv_amt = float(deliverable_amount_l)
    undeliv_amt = float(undelivered_amount_l)
    req_rate = float(required_rate_lpm)
    eff = float(delivery_efficiency_pct)

    for name, val in [
        ("allocated_amount_l", alloc_amt),
        ("deliverable_amount_l", deliv_amt),
        ("undelivered_amount_l", undeliv_amt),
        ("required_rate_lpm", req_rate),
    ]:
        if val < 0 or math.isnan(val) or math.isinf(val):
            raise ValueError(f"{name} must be non-negative and finite (got {val}).")

    if eff < 0 or eff > 100 or math.isnan(eff) or math.isinf(eff):
        raise ValueError(f"Delivery efficiency percentage must be between 0 and 100 (got {eff}).")

    nodes_json = route_node_ids if isinstance(route_node_ids, str) else json.dumps(route_node_ids or [])
    pipes_json = route_pipe_ids if isinstance(route_pipe_ids, str) else json.dumps(route_pipe_ids or [])

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO delivery_zone_assessments (
                assessment_id, zone_id, zone_name_snapshot, destination_node_id,
                destination_name_snapshot, allocated_amount_l, deliverable_amount_l,
                undelivered_amount_l, required_rate_lpm, bottleneck_capacity_lpm,
                bottleneck_pipe_id, bottleneck_pipe_name_snapshot, route_cost,
                route_node_ids, route_pipe_ids, delivery_efficiency_pct,
                feasibility_status, explanation
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(assessment_id),
                int(zone_id) if zone_id is not None else None,
                str(zone_name_snapshot).strip(),
                int(destination_node_id) if destination_node_id is not None else None,
                str(destination_name_snapshot).strip() if destination_name_snapshot else None,
                alloc_amt,
                deliv_amt,
                undeliv_amt,
                req_rate,
                float(bottleneck_capacity_lpm) if bottleneck_capacity_lpm is not None else None,
                int(bottleneck_pipe_id) if bottleneck_pipe_id is not None else None,
                str(bottleneck_pipe_name_snapshot).strip() if bottleneck_pipe_name_snapshot else None,
                float(route_cost) if route_cost is not None else None,
                nodes_json,
                pipes_json,
                eff,
                status,
                str(explanation).strip(),
            ),
        )
        conn.commit()
        return cur.lastrowid


def record_distribution_assessment(
    plan_result: Any,
    run_id: int,
    notes: str | None = None,
    fail_on_zone_id: int | None = None,
) -> int:
    """
    Atomically persist a complete DistributionPlanResult along with all its per-zone assessment rows.
    Rolls back the entire assessment if an error occurs.
    fail_on_zone_id provides an intentional hook for testing atomicity and rollback.
    """
    now = datetime.now(timezone.utc).isoformat()
    zone_results = _get_field(plan_result, "zone_results", [])

    src_id = _get_field(plan_result, "source_node_id")
    dur = float(_get_field(plan_result, "duration_minutes", 60.0))
    strat = str(_get_field(plan_result, "routing_strategy", "SHORTEST_PATH")).strip().upper()

    if strat not in ALLOWED_ROUTING_STRATEGIES:
        raise ValueError(
            f"Routing strategy '{strat}' is invalid. "
            f"Must be one of: {', '.join(sorted(ALLOWED_ROUTING_STRATEGIES))}."
        )

    if dur <= 0 or math.isnan(dur) or math.isinf(dur):
        raise ValueError(f"Delivery duration must be strictly positive and finite (got {dur}).")

    tot_alloc = float(_get_field(plan_result, "total_allocated_l", 0.0))
    tot_deliv = float(_get_field(plan_result, "total_deliverable_l", 0.0))
    tot_undeliv = float(_get_field(plan_result, "total_undelivered_l", 0.0))
    eff = float(
        _get_field(
            plan_result,
            "overall_delivery_efficiency_pct",
            _get_field(plan_result, "delivery_efficiency_pct", 0.0),
        )
    )

    for name, val in [("total_allocated_l", tot_alloc), ("total_deliverable_l", tot_deliv), ("total_undelivered_l", tot_undeliv)]:
        if val < 0 or math.isnan(val) or math.isinf(val):
            raise ValueError(f"{name} must be non-negative and finite (got {val}).")

    if eff < 0 or eff > 100 or math.isnan(eff) or math.isinf(eff):
        raise ValueError(f"Overall delivery efficiency must be between 0 and 100 (got {eff}).")

    # Reconciliation check between plan totals and sum of individual zone results
    if zone_results:
        sum_alloc = round(sum(float(_get_field(z, "allocated_volume_l", _get_field(z, "allocated_amount_l", 0.0))) for z in zone_results), 2)
        sum_deliv = round(sum(float(_get_field(z, "deliverable_volume_l", _get_field(z, "deliverable_amount_l", 0.0))) for z in zone_results), 2)
        sum_undeliv = round(sum(float(_get_field(z, "undelivered_volume_l", _get_field(z, "undelivered_amount_l", 0.0))) for z in zone_results), 2)

        if not math.isclose(sum_alloc, tot_alloc, abs_tol=0.05):
            raise ValueError(
                f"Totals reconciliation error: sum of zone allocations ({sum_alloc} L) "
                f"does not match plan total_allocated_l ({tot_alloc} L)."
            )
        if not math.isclose(sum_deliv, tot_deliv, abs_tol=0.05):
            raise ValueError(
                f"Totals reconciliation error: sum of zone deliverable ({sum_deliv} L) "
                f"does not match plan total_deliverable_l ({tot_deliv} L)."
            )
        if not math.isclose(sum_undeliv, tot_undeliv, abs_tol=0.05):
            raise ValueError(
                f"Totals reconciliation error: sum of zone undelivered ({sum_undeliv} L) "
                f"does not match plan total_undelivered_l ({tot_undeliv} L)."
            )

    shared_warnings = _get_field(plan_result, "shared_pipe_warnings", [])
    has_contention = 1 if shared_warnings else 0
    warnings_json = json.dumps(shared_warnings) if shared_warnings else None
    assessment_notes = notes if notes is not None else _get_field(plan_result, "notes", None)

    with _connect() as conn:
        # Validate allocation run exists
        run = conn.execute("SELECT run_id FROM allocation_runs WHERE run_id = ?", (int(run_id),)).fetchone()
        if run is None:
            raise KeyError(f"Referenced allocation run #{run_id} does not exist.")

        # Validate and snapshot source node
        src_row = conn.execute("SELECT id, name, status FROM network_nodes WHERE id = ?", (int(src_id),)).fetchone()
        if src_row is None:
            raise ValueError(f"Source node #{src_id} does not exist in network_nodes.")
        if src_row["status"] != "active":
            raise ValueError(f"Source node #{src_id} ('{src_row['name']}') is inactive.")
        source_name_snapshot = str(src_row["name"])

        # 1. Insert assessment summary
        cur = conn.execute(
            """
            INSERT INTO delivery_assessments (
                run_id, source_node_id, source_name_snapshot, duration_minutes,
                routing_strategy, total_allocated_l, total_deliverable_l,
                total_undelivered_l, delivery_efficiency_pct, has_contention,
                contention_warnings_json, notes, assessed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(run_id),
                int(src_id),
                source_name_snapshot,
                dur,
                strat,
                tot_alloc,
                tot_deliv,
                tot_undeliv,
                eff,
                has_contention,
                warnings_json,
                assessment_notes,
                now,
            ),
        )
        assessment_id = cur.lastrowid

        # 2. Insert per-zone assessment rows
        for z in zone_results:
            zid = _get_field(z, "zone_id")
            if fail_on_zone_id is not None and zid is not None and int(zid) == int(fail_on_zone_id):
                raise RuntimeError(
                    f"Simulated transaction failure triggered for zone_id={fail_on_zone_id}"
                )

            zname = str(_get_field(z, "zone_name", f"Zone-{zid}"))
            dest_id = _get_field(z, "destination_node_id")

            dest_name = None
            if dest_id is not None:
                d_row = conn.execute("SELECT name FROM network_nodes WHERE id = ?", (int(dest_id),)).fetchone()
                dest_name = d_row["name"] if d_row else f"Node-{dest_id}"

            z_alloc = float(_get_field(z, "allocated_volume_l", _get_field(z, "allocated_amount_l", 0.0)))
            z_deliv = float(_get_field(z, "deliverable_volume_l", _get_field(z, "deliverable_amount_l", 0.0)))
            z_undeliv = float(_get_field(z, "undelivered_volume_l", _get_field(z, "undelivered_amount_l", 0.0)))
            z_rate = float(_get_field(z, "required_rate_lpm", 0.0))
            z_eff = float(_get_field(z, "delivery_efficiency_pct", 0.0))

            bot_cap = _get_field(z, "bottleneck_capacity_lpm")
            bot_pid = _get_field(z, "bottleneck_pipe_id")
            bot_pname = _get_field(z, "bottleneck_pipe_name", _get_field(z, "bottleneck_pipe_name_snapshot"))
            r_cost = _get_field(z, "route_cost")

            route_nodes = _get_field(z, "route_node_ids", [])
            route_pipes = _get_field(z, "route_pipe_ids", [])
            nodes_json = json.dumps(route_nodes) if not isinstance(route_nodes, str) else route_nodes
            pipes_json = json.dumps(route_pipes) if not isinstance(route_pipes, str) else route_pipes

            status = str(_get_field(z, "feasibility_status", "UNMAPPED")).strip().upper()
            if status not in ALLOWED_FEASIBILITY_STATUSES:
                raise ValueError(f"Invalid feasibility status '{status}' for zone '{zname}'.")

            explanation = str(_get_field(z, "explanation", ""))

            conn.execute(
                """
                INSERT INTO delivery_zone_assessments (
                    assessment_id, zone_id, zone_name_snapshot, destination_node_id,
                    destination_name_snapshot, allocated_amount_l, deliverable_amount_l,
                    undelivered_amount_l, required_rate_lpm, bottleneck_capacity_lpm,
                    bottleneck_pipe_id, bottleneck_pipe_name_snapshot, route_cost,
                    route_node_ids, route_pipe_ids, delivery_efficiency_pct,
                    feasibility_status, explanation
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    assessment_id,
                    int(zid) if zid is not None else None,
                    zname,
                    int(dest_id) if dest_id is not None else None,
                    dest_name,
                    z_alloc,
                    z_deliv,
                    z_undeliv,
                    z_rate,
                    float(bot_cap) if bot_cap is not None else None,
                    int(bot_pid) if bot_pid is not None else None,
                    str(bot_pname) if bot_pname is not None else None,
                    float(r_cost) if r_cost is not None else None,
                    nodes_json,
                    pipes_json,
                    z_eff,
                    status,
                    explanation,
                ),
            )

        conn.commit()
        return assessment_id


def get_delivery_assessment_by_id(assessment_id: int) -> sqlite3.Row | None:
    """
    Return a delivery assessment by its ID, or None if not found.
    Aliases source_name_snapshot AS source_node_name so callers/UI receive historical
    snapshot data and never depend on current network_nodes.name.
    """
    with _connect() as conn:
        return conn.execute(
            """
            SELECT da.*,
                   da.source_name_snapshot AS source_node_name
              FROM delivery_assessments da
             WHERE da.assessment_id = ?
            """,
            (int(assessment_id),),
        ).fetchone()


def get_delivery_assessments_for_run(run_id: int, limit: int = 50) -> list[sqlite3.Row]:
    """Return all delivery assessments recorded for an allocation run, newest first."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT da.*,
                   da.source_name_snapshot AS source_node_name
              FROM delivery_assessments da
             WHERE da.run_id = ?
             ORDER BY da.assessed_at DESC, da.assessment_id DESC
             LIMIT ?
            """,
            (int(run_id), limit),
        ).fetchall()


def get_all_delivery_assessments(limit: int = 50) -> list[sqlite3.Row]:
    """Return all delivery assessments recorded across runs, newest first."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT da.*,
                   da.source_name_snapshot AS source_node_name
              FROM delivery_assessments da
             ORDER BY da.assessed_at DESC, da.assessment_id DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()


def get_delivery_zone_assessments(assessment_id: int) -> list[sqlite3.Row]:
    """
    Return all per-zone assessment rows for a given assessment, ordered by ID.
    Aliases zone_name_snapshot AS zone_name and destination_name_snapshot AS destination_node_name
    so callers/UI receive historical snapshot data without depending on live entity tables.
    """
    with _connect() as conn:
        return conn.execute(
            """
            SELECT dza.*,
                   dza.zone_name_snapshot AS zone_name,
                   dza.destination_name_snapshot AS destination_node_name
              FROM delivery_zone_assessments dza
             WHERE dza.assessment_id = ?
             ORDER BY dza.zone_assessment_id ASC
            """,
            (int(assessment_id),),
        ).fetchall()


def get_latest_delivery_assessment(run_id: int | None = None) -> sqlite3.Row | None:
    """Return the most recent delivery assessment, optionally filtered by run_id."""
    with _connect() as conn:
        if run_id is not None:
            return conn.execute(
                """
                SELECT da.*,
                       da.source_name_snapshot AS source_node_name
                  FROM delivery_assessments da
                 WHERE da.run_id = ?
                 ORDER BY da.assessed_at DESC, da.assessment_id DESC
                 LIMIT 1
                """,
                (int(run_id),),
            ).fetchone()
        return conn.execute(
            """
            SELECT da.*,
                   da.source_name_snapshot AS source_node_name
              FROM delivery_assessments da
             ORDER BY da.assessed_at DESC, da.assessment_id DESC
             LIMIT 1
            """
        ).fetchone()


def delete_delivery_assessment(assessment_id: int) -> None:
    """
    Delete a delivery assessment.
    Cascades to delete attached delivery_zone_assessments rows.
    """
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM delivery_assessments WHERE assessment_id = ?",
            (int(assessment_id),),
        )
        if cur.rowcount == 0:
            raise KeyError(f"Delivery assessment #{assessment_id} does not exist.")
        conn.commit()


def count_delivery_assessments(run_id: int | None = None) -> int:
    """Return the total number of recorded delivery assessments."""
    with _connect() as conn:
        if run_id is not None:
            return conn.execute(
                "SELECT COUNT(*) FROM delivery_assessments WHERE run_id = ?",
                (int(run_id),),
            ).fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM delivery_assessments").fetchone()[0]


def count_delivery_zone_assessments(assessment_id: int | None = None) -> int:
    """Return the total number of recorded delivery zone detail rows."""
    with _connect() as conn:
        if assessment_id is not None:
            return conn.execute(
                "SELECT COUNT(*) FROM delivery_zone_assessments WHERE assessment_id = ?",
                (int(assessment_id),),
            ).fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM delivery_zone_assessments").fetchone()[0]

