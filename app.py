import os
import secrets
from datetime import datetime, timedelta, timezone
import json
import math
import sqlite3

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_wtf.csrf import CSRFProtect

import database as db
from dsa.graph import load_graph_from_db
from dsa.dijkstra import dijkstra
from simulation import SensorSimulator
from detection import LeakDetector, build_detection_result_message
from repair_queue import PriorityQueueManager, MaxHeap, RepairTask
from allocation.engine import ZoneDemandInput, allocate_water
from distribution import evaluate_and_record_run_distribution
from distribution.engine import STRATEGY_SHORTEST_PATH, STRATEGY_WIDEST_PATH

# ── Application setup ────────────────────────────────────────────────────────

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "AQUAGRID_SECRET_KEY", 'aquagrid-prototype-secret-key-2026'
)

csrf = CSRFProtect(app)

try:
    import zoneinfo
    IST = zoneinfo.ZoneInfo("Asia/Kolkata")
except Exception:
    IST = timezone(timedelta(hours=5, minutes=30), name="IST")


def format_ist_timestamp(ts_str: str | None) -> str:
    """
    Convert a UTC ISO timestamp string into an Asia/Kolkata (IST) display string.
    Expected format: '15 Nov 2026, 01:04:30 PM IST'.
    Gracefully handles naive strings, 'Z' suffixes, None, and malformed inputs.
    """
    if not ts_str:
        return "N/A"
    try:
        clean_ts = str(ts_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_ist = dt.astimezone(IST)
        return dt_ist.strftime("%d %b %Y, %I:%M:%S %p IST")
    except Exception:
        return str(ts_str)


@app.template_filter("format_ist")
def format_ist_filter(ts_str: str | None) -> str:
    return format_ist_timestamp(ts_str)


# Operational simulation horizon for continuous consecutive batches (2 hours)
MAX_SIMULATION_LEAD = timedelta(hours=2)


def calculate_simulator_base_time(
    now_utc: datetime,
    latest_recorded_at: str | datetime | None,
    max_simulation_lead: timedelta = MAX_SIMULATION_LEAD,
) -> datetime:
    """
    Calculate the starting base timestamp for a new simulator batch.

    Rules:
    1. Base time defaults to now_utc (current system wall-clock).
    2. If latest_recorded_at is in the past (now_utc > last_dt), e.g. historical data
       from previous days or earlier today, snap directly to now_utc.
    3. If latest_recorded_at is in the future compared to now_utc due to recent
       consecutive simulation batches within the allowed max_simulation_lead (up to 2 hours),
       advance by 1 second past last_dt (last_dt + 1s) to preserve strict chronological
       monotonicity and avoid duplicate timestamps across rapid batches.
    4. If latest_recorded_at is an arbitrary future-dated legacy reading that exceeds
       max_simulation_lead ahead of now_utc (e.g., year 2099 or days in the future),
       do not allow indefinite drift; clamp and reset to now_utc.
    """
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    if not latest_recorded_at:
        return now_utc

    try:
        if isinstance(latest_recorded_at, datetime):
            last_dt = latest_recorded_at
        else:
            clean_ts = str(latest_recorded_at).replace("Z", "+00:00")
            last_dt = datetime.fromisoformat(clean_ts)

        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)

        if now_utc > last_dt:
            return now_utc

        lead = last_dt - now_utc
        if lead <= max_simulation_lead:
            return last_dt + timedelta(seconds=1)
        else:
            return now_utc
    except Exception:
        return now_utc


with app.app_context():
    db.init_db()


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    node_count = db.count_nodes()
    open_leaks_count = db.count_open_leak_events()
    repair_manager = PriorityQueueManager()
    top_repair_tasks = repair_manager.get_ordered_tasks(db)[:5]
    pending_repairs_count = db.count_pending_repair_tasks()
    active_repairs_count = db.count_active_repair_tasks()
    return render_template(
        "index.html",
        node_count=node_count,
        open_leaks_count=open_leaks_count,
        pending_repairs_count=pending_repairs_count,
        active_repairs_count=active_repairs_count,
        top_repair_tasks=top_repair_tasks,
    )


# ── Water Network – Nodes ──────────────────────────────────────────────────────

@app.route("/network")
def network():
    nodes = db.get_all_nodes()
    return render_template(
        "network.html",
        nodes=nodes,
        allowed_types=sorted(db.ALLOWED_NODE_TYPES),
        allowed_statuses=sorted(db.ALLOWED_STATUSES),
    )


@app.route("/network/create", methods=["POST"])
def network_create():
    name      = request.form.get("name", "").strip()
    node_type = request.form.get("node_type", "")
    location  = request.form.get("location", "").strip()
    status    = request.form.get("status", "active")

    try:
        db.create_node(name, node_type, location, status)
        flash("Node created successfully.", "success")
        return redirect(url_for("network"))
    except ValueError as exc:
        for msg in exc.args[0]:
            flash(msg, "error")
        return redirect(url_for("network"))


@app.route("/network/<int:node_id>/edit", methods=["GET", "POST"])
def network_edit(node_id):
    node = db.get_node_by_id(node_id)
    if node is None:
        flash("Node not found.", "error")
        return redirect(url_for("network"))

    if request.method == "POST":
        name      = request.form.get("name", "").strip()
        node_type = request.form.get("node_type", "")
        location  = request.form.get("location", "").strip()
        status    = request.form.get("status", "active")

        try:
            db.update_node(node_id, name, node_type, location, status)
            flash("Node updated successfully.", "success")
            return redirect(url_for("network"))
        except ValueError as exc:
            for msg in exc.args[0]:
                flash(msg, "error")
            bad_node = {
                "id": node_id, "name": name, "node_type": node_type,
                "location": location, "status": status,
                "created_at": node["created_at"],
            }
            return render_template(
                "network_edit.html",
                node=bad_node,
                allowed_types=sorted(db.ALLOWED_NODE_TYPES),
                allowed_statuses=sorted(db.ALLOWED_STATUSES),
            )

    return render_template(
        "network_edit.html",
        node=node,
        allowed_types=sorted(db.ALLOWED_NODE_TYPES),
        allowed_statuses=sorted(db.ALLOWED_STATUSES),
    )


@app.route("/network/<int:node_id>/delete", methods=["POST"])
def network_delete(node_id):
    try:
        db.delete_node(node_id)
        flash("Node deleted.", "success")
    except RuntimeError as exc:
        # Node is still referenced by pipes – surface friendly message.
        flash(str(exc), "error")
    except LookupError:
        flash("Node not found.", "error")
    return redirect(url_for("network"))


# ── Water Network – Pipes ──────────────────────────────────────────────────────

@app.route("/pipes")
def pipes():
    all_pipes = db.get_all_pipes()
    node_map  = db.get_node_name_map()   # {id: name} for dropdowns
    return render_template(
        "pipes.html",
        pipes=all_pipes,
        node_map=node_map,
        allowed_statuses=sorted(db.ALLOWED_STATUSES),
    )


@app.route("/pipes/create", methods=["POST"])
def pipes_create():
    name             = request.form.get("name", "").strip()
    from_node_id_raw = request.form.get("from_node_id", "")
    to_node_id_raw   = request.form.get("to_node_id", "")
    length_m_raw     = request.form.get("length_m", "")
    capacity_lpm_raw = request.form.get("capacity_lpm", "")
    cost_raw         = request.form.get("cost", "1")
    status           = request.form.get("status", "active")

    from_node_id = _parse_int_or_none(from_node_id_raw)
    to_node_id   = _parse_int_or_none(to_node_id_raw)

    try:
        db.create_pipe(
            name, from_node_id, to_node_id,
            length_m_raw, capacity_lpm_raw, cost_raw, status,
            require_positive_cost=True,
        )
        flash("Pipe created successfully.", "success")
        return redirect(url_for("pipes"))
    except ValueError as exc:
        for msg in exc.args[0]:
            flash(msg, "error")
        return redirect(url_for("pipes"))


@app.route("/pipes/<int:pipe_id>/edit", methods=["GET", "POST"])
def pipes_edit(pipe_id):
    pipe = db.get_pipe_by_id(pipe_id)
    if pipe is None:
        flash("Pipe not found.", "error")
        return redirect(url_for("pipes"))

    node_map = db.get_node_name_map()

    if request.method == "POST":
        name             = request.form.get("name", "").strip()
        from_node_id_raw = request.form.get("from_node_id", "")
        to_node_id_raw   = request.form.get("to_node_id", "")
        length_m_raw     = request.form.get("length_m", "")
        capacity_lpm_raw = request.form.get("capacity_lpm", "")
        cost_raw         = request.form.get("cost", "1")
        status           = request.form.get("status", "active")

        from_node_id = _parse_int_or_none(from_node_id_raw)
        to_node_id   = _parse_int_or_none(to_node_id_raw)

        try:
            db.update_pipe(
                pipe_id, name, from_node_id, to_node_id,
                length_m_raw, capacity_lpm_raw, cost_raw, status,
                require_positive_cost=True,
            )
            flash("Pipe updated successfully.", "success")
            return redirect(url_for("pipes"))
        except ValueError as exc:
            for msg in exc.args[0]:
                flash(msg, "error")
            # Reconstruct a dict so the template can pre-fill the form.
            bad_pipe = {
                "id": pipe_id, "name": name,
                "from_node_id": from_node_id, "to_node_id": to_node_id,
                "length_m": length_m_raw, "capacity_lpm": capacity_lpm_raw,
                "cost": cost_raw, "status": status,
                "created_at": pipe["created_at"],
            }
            return render_template(
                "pipes_edit.html",
                pipe=bad_pipe,
                node_map=node_map,
                allowed_statuses=sorted(db.ALLOWED_STATUSES),
            )

    return render_template(
        "pipes_edit.html",
        pipe=pipe,
        node_map=node_map,
        allowed_statuses=sorted(db.ALLOWED_STATUSES),
    )


@app.route("/pipes/<int:pipe_id>/delete", methods=["POST"])
def pipes_delete(pipe_id):
    try:
        db.delete_pipe(pipe_id)
        flash("Pipe deleted.", "success")
    except RuntimeError as exc:
        flash(str(exc), "error")
    except LookupError:
        flash("Pipe not found.", "error")
    return redirect(url_for("pipes"))


# ── Sensor Management ─────────────────────────────────────────────────────────

@app.route("/sensors")
def sensors():
    all_sensors = db.get_all_sensors()
    pipe_map = db.get_pipe_name_map()
    return render_template(
        "sensors.html",
        sensors=all_sensors,
        pipe_map=pipe_map,
        allowed_types=sorted(db.ALLOWED_SENSOR_TYPES),
        allowed_statuses=sorted(db.ALLOWED_STATUSES),
        default_units=db.DEFAULT_SENSOR_UNITS,
    )


@app.route("/sensors/create", methods=["POST"])
def sensors_create():
    sensor_name = request.form.get("sensor_name", "").strip()
    pipe_id_raw = request.form.get("pipe_id", "")
    sensor_type = request.form.get("sensor_type", "").strip()
    unit = request.form.get("unit", "").strip()
    status = request.form.get("status", "active").strip()

    pipe_id = _parse_int_or_none(pipe_id_raw)
    if not unit and sensor_type in db.DEFAULT_SENSOR_UNITS:
        unit = db.DEFAULT_SENSOR_UNITS[sensor_type]

    try:
        db.create_sensor(sensor_name, pipe_id, sensor_type, unit, status)
        flash("Sensor registered successfully.", "success")
        return redirect(url_for("sensors"))
    except ValueError as exc:
        for msg in exc.args[0]:
            flash(msg, "error")
        return redirect(url_for("sensors"))


@app.route("/sensors/<int:sensor_id>/edit", methods=["GET", "POST"])
def sensors_edit(sensor_id):
    sensor = db.get_sensor_by_id(sensor_id)
    if sensor is None:
        flash("Sensor not found.", "error")
        return redirect(url_for("sensors"))

    pipe_map = db.get_pipe_name_map()

    if request.method == "POST":
        sensor_name = request.form.get("sensor_name", "").strip()
        pipe_id_raw = request.form.get("pipe_id", "")
        sensor_type = request.form.get("sensor_type", "").strip()
        unit = request.form.get("unit", "").strip()
        status = request.form.get("status", "active").strip()

        pipe_id = _parse_int_or_none(pipe_id_raw)
        if not unit and sensor_type in db.DEFAULT_SENSOR_UNITS:
            unit = db.DEFAULT_SENSOR_UNITS[sensor_type]

        try:
            db.update_sensor(sensor_id, sensor_name, pipe_id, sensor_type, unit, status)
            flash("Sensor updated successfully.", "success")
            return redirect(url_for("sensors"))
        except ValueError as exc:
            for msg in exc.args[0]:
                flash(msg, "error")
            bad_sensor = {
                "sensor_id": sensor_id,
                "sensor_name": sensor_name,
                "pipe_id": pipe_id,
                "sensor_type": sensor_type,
                "unit": unit,
                "status": status,
                "created_at": sensor["created_at"],
                "pipe_name": sensor["pipe_name"],
                "from_node_name": sensor["from_node_name"],
                "to_node_name": sensor["to_node_name"],
            }
            return render_template(
                "sensors_edit.html",
                sensor=bad_sensor,
                pipe_map=pipe_map,
                allowed_types=sorted(db.ALLOWED_SENSOR_TYPES),
                allowed_statuses=sorted(db.ALLOWED_STATUSES),
                default_units=db.DEFAULT_SENSOR_UNITS,
            )

    return render_template(
        "sensors_edit.html",
        sensor=sensor,
        pipe_map=pipe_map,
        allowed_types=sorted(db.ALLOWED_SENSOR_TYPES),
        allowed_statuses=sorted(db.ALLOWED_STATUSES),
        default_units=db.DEFAULT_SENSOR_UNITS,
    )


@app.route("/sensors/<int:sensor_id>/toggle-status", methods=["POST"])
def sensors_toggle_status(sensor_id):
    try:
        new_status = db.toggle_sensor_status(sensor_id)
        flash(f"Sensor #{sensor_id} status changed to {new_status}.", "success")
    except LookupError:
        flash("Sensor not found.", "error")
    return redirect(url_for("sensors"))


@app.route("/sensors/<int:sensor_id>/delete", methods=["POST"])
def sensors_delete(sensor_id):
    try:
        db.delete_sensor(sensor_id)
        flash("Sensor deleted.", "success")
    except RuntimeError as exc:
        flash(str(exc), "error")
    except LookupError:
        flash("Sensor not found.", "error")
    return redirect(url_for("sensors"))


# ── Sensor Simulator ──────────────────────────────────────────────────────────

@app.route("/sensor-simulator", methods=["GET"])
def sensor_simulator():
    all_sensors = db.get_all_sensors()
    active_sensors = db.get_active_sensors()
    pipe_map = db.get_pipe_name_map()
    total_readings = db.count_sensor_readings()
    active_sensor_count = len(active_sensors)
    latest_flow = db.get_latest_reading_by_type("flow_meter")
    latest_pressure = db.get_latest_reading_by_type("pressure_sensor")
    latest_readings = db.get_latest_sensor_readings(limit=50)

    mode = request.args.get("mode", "normal")
    selected_pipe_id = _parse_int_or_none(request.args.get("pipe_id"))

    return render_template(
        "sensor_simulator.html",
        all_sensors=all_sensors,
        active_sensors=active_sensors,
        active_sensor_count=active_sensor_count,
        pipe_map=pipe_map,
        total_readings=total_readings,
        latest_flow=latest_flow,
        latest_pressure=latest_pressure,
        latest_readings=latest_readings,
        mode=mode,
        selected_pipe_id=selected_pipe_id,
    )


@app.route("/sensor-simulator/generate", methods=["POST"])
def sensor_simulator_generate():
    mode = request.form.get("mode", "normal").strip().lower()
    pipe_id_raw = request.form.get("pipe_id", "")
    pipe_id = _parse_int_or_none(pipe_id_raw)
    count_raw = request.form.get("count", "1")
    seed_raw = request.form.get("seed", "")
    seed = _parse_int_or_none(seed_raw)

    count = 10 if count_raw == "10" else 1

    active_sensors = db.get_active_sensors()
    if not active_sensors:
        flash("No active sensors found in the network. Register or activate sensors first.", "error")
        return redirect(url_for("sensor_simulator", mode=mode))

    if mode == "leak":
        if pipe_id is None:
            flash("Please select a monitored pipe for the leak scenario.", "error")
            return redirect(url_for("sensor_simulator", mode="leak"))

        pipe = db.get_pipe_by_id(pipe_id)
        if pipe is None:
            flash(f"Selected pipe #{pipe_id} does not exist.", "error")
            return redirect(url_for("sensor_simulator", mode="leak"))

        pipe_active_sensors = [s for s in active_sensors if s["pipe_id"] == pipe_id]
        if not pipe_active_sensors:
            flash(
                f"Selected pipe '{pipe['name']}' (#{pipe_id}) has no active sensors installed. "
                f"Choose a pipe with active sensors or register sensors on this pipe to simulate an anomaly.",
                "error",
            )
            return redirect(url_for("sensor_simulator", mode="leak", pipe_id=pipe_id))
    else:
        mode = "normal"
        pipe = None

    clock_fn = app.config.get("SIMULATOR_CLOCK", None)
    now_utc = clock_fn() if callable(clock_fn) else datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    latest_readings = db.get_latest_sensor_readings(limit=1)
    latest_ts = latest_readings[0]["recorded_at"] if latest_readings else None
    base_time = calculate_simulator_base_time(now_utc, latest_ts)

    simulator = SensorSimulator(seed=seed)
    readings = simulator.generate_batch(
        active_sensors=active_sensors,
        mode=mode,
        leak_pipe_id=pipe_id if mode == "leak" else None,
        iterations=count,
        base_time=base_time,
    )

    db.record_sensor_readings_batch(readings)

    if mode == "leak":
        pipe_name = pipe["name"] if pipe else f"#{pipe_id}"
        flash(
            f"Generated {len(readings)} synthetic telemetry reading(s) across {len(active_sensors)} active sensor(s), "
            f"including simulated leak anomaly on pipe '{pipe_name}'.",
            "success",
        )
        return redirect(url_for("sensor_simulator", mode="leak", pipe_id=pipe_id))
    else:
        flash(
            f"Generated {len(readings)} synthetic telemetry reading(s) across {len(active_sensors)} active sensor(s) in Normal mode.",
            "success",
        )
        return redirect(url_for("sensor_simulator", mode="normal"))


@app.route("/sensor-simulator/clear", methods=["POST"])
def sensor_simulator_clear():
    deleted = db.clear_sensor_readings()
    flash(f"Cleared {deleted} simulated sensor reading(s).", "success")
    return redirect(url_for("sensor_simulator"))


# ── Leak Detection Engine ───────────────────────────────────────────────────

@app.route("/leak-detection", methods=["GET"])
def leak_detection():
    detector = LeakDetector()
    diagnostics = detector.run_detection_network(db, auto_record=False)

    open_events = db.get_open_leak_events()
    all_events = db.get_all_leak_events(limit=50)
    open_count = db.count_open_leak_events()
    investigating_pipes_count = db.count_pipes_under_investigation()
    total_events = db.count_total_leak_events()
    total_readings = db.count_sensor_readings()

    anomalous_pipes = [d for d in diagnostics if d.status == "suspected_leak"]
    normal_pipes = [d for d in diagnostics if d.status == "normal"]
    insufficient_pipes = [d for d in diagnostics if d.status == "insufficient_data"]

    return render_template(
        "leak_detection.html",
        diagnostics=diagnostics,
        open_events=open_events,
        all_events=all_events,
        open_count=open_count,
        investigating_pipes_count=investigating_pipes_count,
        total_events=total_events,
        total_readings=total_readings,
        anomalous_pipes_count=len(anomalous_pipes),
        normal_pipes_count=len(normal_pipes),
        insufficient_pipes_count=len(insufficient_pipes),
        allowed_statuses=sorted(db.ALLOWED_EVENT_STATUSES),
    )


@app.route("/leak-detection/run", methods=["POST"])
def leak_detection_run():
    detector = LeakDetector()
    diagnostics = detector.run_detection_network(db, auto_record=True)
    queue_mgr = PriorityQueueManager()
    queue_mgr.sync_and_build_heap(db)

    message, category = build_detection_result_message(diagnostics)
    flash(message, category)
    return redirect(url_for("leak_detection"))


@app.route("/leak-detection/events/<int:event_id>/status", methods=["POST"])
def leak_detection_update_status(event_id):
    new_status = request.form.get("status", "").strip().lower()
    try:
        db.update_leak_event_status(event_id, new_status)
        flash(f"Leak event #{event_id} status updated to '{new_status}'.", "success")
    except (ValueError, LookupError) as exc:
        flash(str(exc), "error")
    return redirect(url_for("leak_detection"))


# ── Smart Repair Priority Queue ───────────────────────────────────────────────

@app.route("/repairs", methods=["GET"])
@app.route("/repair-queue", methods=["GET"])
def repair_queue():
    manager = PriorityQueueManager()
    ordered_tasks = manager.get_ordered_tasks(db)
    all_tasks = manager.get_all_tasks_for_history(db)
    pending_count = db.count_pending_repair_tasks()
    active_count = db.count_active_repair_tasks()
    category_counts = db.count_repair_tasks_by_category()
    lifecycle_logs = db.get_lifecycle_audit_logs(limit=50)

    return render_template(
        "repairs.html",
        ordered_tasks=ordered_tasks,
        all_tasks=all_tasks,
        pending_count=pending_count,
        active_count=active_count,
        category_counts=category_counts,
        allowed_statuses=sorted(db.ALLOWED_REPAIR_STATUSES),
        lifecycle_logs=lifecycle_logs,
    )


@app.route("/repairs/<int:task_id>/status", methods=["POST"])
def repair_task_update_status(task_id: int):
    new_status = request.form.get("status", "").strip()
    try:
        db.update_repair_task_status(task_id, new_status)
        flash(f"Repair task #{task_id} status updated to '{new_status}'.", "success")
    except (ValueError, LookupError) as exc:
        flash(str(exc), "error")
    return redirect(url_for("repair_queue"))


@app.route("/repairs/<int:task_id>/investigate", methods=["POST"])
def repair_task_investigate(task_id: int):
    try:
        task = db.get_repair_task_by_id(task_id)
        if not task:
            raise LookupError(f"Repair task #{task_id} not found.")
        db.update_repair_task_status(
            task_id,
            "In Progress",
            action_description=f"Investigation begun for repair task #{task_id}. Field dispatch crew notified.",
        )
        task_dict = dict(task)
        event_id = task_dict.get("event_id")
        if event_id:
            ev = db.get_leak_event_by_id(event_id)
            if ev and dict(ev).get("status") == "open":
                db.update_leak_event_status(
                    event_id,
                    "investigating",
                    action_description=f"Field investigation initiated via repair task #{task_id}",
                )
        flash(f"Investigation begun for repair task #{task_id}. Status set to 'In Progress'.", "success")
    except (ValueError, LookupError) as exc:
        flash(str(exc), "error")
    return redirect(url_for("repair_queue"))


@app.route("/repairs/sync", methods=["POST"])
def repair_queue_sync():
    manager = PriorityQueueManager()
    manager.sync_and_build_heap(db)
    flash("Repair queue synchronized with latest suspected leak events.", "success")
    return redirect(url_for("repair_queue"))


# ── Graph Explorer ────────────────────────────────────────────────────────────

@app.route("/graph", methods=["GET", "POST"])
def graph_explorer():
    all_nodes = db.get_all_nodes()
    node_map = {
        int(n["id"]): f"{n['name']} (#{n['id']})"
        for n in all_nodes
    }

    # If the network has no nodes, render the empty network view
    if not all_nodes:
        return render_template(
            "graph.html",
            is_empty_network=True,
            all_nodes=[],
            node_map={},
            selected_node_id=None,
            traversal_mode="undirected",
            simulation=None,
            graph=None,
        )

    # Determine selected node and traversal mode
    raw_start = None
    traversal_mode = "undirected"

    if request.method == "POST":
        raw_start = request.form.get("start_node_id")
        traversal_mode = request.form.get("traversal_mode", "undirected")
    else:
        raw_start = request.args.get("start_node_id")
        traversal_mode = request.args.get("traversal_mode", "undirected")

    selected_node_id = _parse_int_or_none(raw_start)
    if selected_node_id is None and all_nodes:
        # Default to the first node in the database
        selected_node_id = int(all_nodes[0]["id"])

    is_directed = (traversal_mode == "directed")
    graph = load_graph_from_db(db_module=db, directed=is_directed)
    simulation = graph.bfs_detailed(selected_node_id) if selected_node_id is not None else None

    return render_template(
        "graph.html",
        is_empty_network=False,
        all_nodes=all_nodes,
        node_map=node_map,
        selected_node_id=selected_node_id,
        traversal_mode=traversal_mode,
        simulation=simulation,
        graph=graph,
    )


# ── Route Finder (Dijkstra) ───────────────────────────────────────────────────

@app.route("/route", methods=["GET", "POST"])
@app.route("/route-finder", methods=["GET", "POST"])
def route_finder():
    all_nodes = db.get_all_nodes()
    node_map = {
        int(n["id"]): f"{n['name']} (#{n['id']})"
        for n in all_nodes
    }

    if not all_nodes:
        return render_template(
            "route.html",
            is_empty_network=True,
            all_nodes=[],
            node_map={},
            source_id=None,
            destination_id=None,
            traversal_mode="undirected",
            route_result=None,
            path_nodes=[],
            pipe_sequence=[],
            graph=None,
        )

    raw_source = None
    raw_dest = None
    traversal_mode = "undirected"

    if request.method == "POST":
        raw_source = request.form.get("source_id")
        raw_dest = request.form.get("destination_id")
        traversal_mode = request.form.get("traversal_mode", "undirected")
    else:
        raw_source = request.args.get("source_id")
        raw_dest = request.args.get("destination_id")
        traversal_mode = request.args.get("traversal_mode", "undirected")

    source_id = _parse_int_or_none(raw_source)
    destination_id = _parse_int_or_none(raw_dest)

    # Defaults if not supplied
    if source_id is None and all_nodes:
        source_id = int(all_nodes[0]["id"])
    if destination_id is None and all_nodes:
        destination_id = int(all_nodes[1]["id"]) if len(all_nodes) > 1 else int(all_nodes[0]["id"])

    is_directed = (traversal_mode == "directed")
    graph = load_graph_from_db(db_module=db, directed=is_directed)

    route_result = None
    validation_error = None
    path_nodes = []
    pipe_sequence = []

    if source_id is not None and destination_id is not None:
        if source_id == destination_id:
            validation_error = "Source and destination must be different."
        else:
            route_result = dijkstra(graph, source_id, destination_id)
            if route_result.found:
                for nid in route_result.node_path:
                    node_data = graph.nodes.get(
                        nid,
                        {"id": nid, "name": f"Node-{nid}", "node_type": "junction", "status": "active"}
                    )
                    path_nodes.append(node_data)
                pipe_sequence = route_result.pipe_path

    return render_template(
        "route.html",
        is_empty_network=False,
        all_nodes=all_nodes,
        node_map=node_map,
        source_id=source_id,
        destination_id=destination_id,
        traversal_mode=traversal_mode,
        route_result=route_result,
        validation_error=validation_error,
        path_nodes=path_nodes,
        pipe_sequence=pipe_sequence,
        graph=graph,
    )


# ── Fair Water Allocation ──────────────────────────────────────────────────────

def _validate_allocation_period(start_raw: str | None, end_raw: str | None) -> tuple[bool, str | None]:
    """Validate optional ISO / datetime-local strings. Return (is_valid, error_msg)."""
    if not start_raw and not end_raw:
        return True, None
    dt_start = None
    dt_end = None
    if start_raw:
        try:
            dt_start = datetime.fromisoformat(start_raw.strip())
        except ValueError:
            return False, "Period start must be a valid date/time format."
    if end_raw:
        try:
            dt_end = datetime.fromisoformat(end_raw.strip())
        except ValueError:
            return False, "Period end must be a valid date/time format."
    if dt_start and dt_end and dt_start > dt_end:
        return False, "Allocation period start cannot be after period end."
    return True, None


@app.route("/allocation")
def allocation():
    """Main allocation dashboard – strictly read-only."""
    all_zones = db.get_all_allocation_zones()
    active_zones = db.get_active_allocation_zones()
    total_active_demand = sum(float(z["requested_demand_l"]) for z in active_zones)

    latest_run = db.get_latest_allocation_run()
    latest_allocations = db.get_allocations_for_run(latest_run["run_id"]) if latest_run else []
    historical_runs = db.get_all_allocation_runs(limit=50)

    all_nodes = db.get_all_nodes()
    allowed_categories = sorted(db.ALLOWED_ZONE_CATEGORIES)
    allowed_statuses = sorted(db.ALLOWED_ZONE_STATUSES)

    return render_template(
        "allocation.html",
        all_zones=all_zones,
        active_zones=active_zones,
        total_active_demand=total_active_demand,
        latest_run=latest_run,
        latest_allocations=latest_allocations,
        historical_runs=historical_runs,
        all_nodes=all_nodes,
        allowed_categories=allowed_categories,
        allowed_statuses=allowed_statuses,
    )


@app.route("/allocation/run", methods=["POST"])
def allocation_run():
    """Execute fairness engine and atomically persist allocation run."""
    supply_raw = request.form.get("available_supply", "").strip()
    period_start = request.form.get("period_start", "").strip() or None
    period_end = request.form.get("period_end", "").strip() or None
    notes = request.form.get("notes", "").strip() or None

    if not supply_raw:
        flash("Available water supply is required.", "error")
        return redirect(url_for("allocation"))

    try:
        supply_val = float(supply_raw)
        if supply_val < 0:
            flash("Available water supply cannot be negative.", "error")
            return redirect(url_for("allocation"))
    except ValueError:
        flash("Available water supply must be a valid number.", "error")
        return redirect(url_for("allocation"))

    is_valid_period, period_err = _validate_allocation_period(period_start, period_end)
    if not is_valid_period:
        flash(period_err, "error")
        return redirect(url_for("allocation"))

    active_zones = db.get_active_allocation_zones()
    if not active_zones:
        flash("No active allocation zones exist to receive water. Please configure and activate at least one zone.", "warning")
        return redirect(url_for("allocation"))

    zone_inputs = [
        ZoneDemandInput(
            zone_id=int(z["zone_id"]),
            zone_name=str(z["zone_name"]),
            requested_demand=float(z["requested_demand_l"]),
            essential_min=float(z["essential_min_l"]),
            fairness_weight=float(z["fairness_weight"]),
            node_id=z["node_id"],
            priority_category=z["priority_category"],
        )
        for z in active_zones
    ]

    try:
        run_result = allocate_water(total_supply=supply_val, zones=zone_inputs)
        run_id = db.record_allocation_run(
            run_result=run_result,
            period_start=period_start,
            period_end=period_end,
            notes=notes,
        )
        flash(f"Fair water allocation run #{run_id} completed successfully using {run_result.policy_applied} policy.", "success")
        return redirect(url_for("allocation_run_details", run_id=run_id))
    except Exception as exc:
        flash(f"Allocation run failed: {exc}", "error")
        return redirect(url_for("allocation"))


@app.route("/allocation/zones/create", methods=["POST"])
def allocation_zone_create():
    """Create a new allocation zone."""
    zone_name = request.form.get("zone_name", "").strip()
    requested_demand_l = request.form.get("requested_demand_l", "").strip()
    essential_min_l = request.form.get("essential_min_l", "0").strip() or "0"
    fairness_weight = request.form.get("fairness_weight", "1.0").strip() or "1.0"
    priority_category = request.form.get("priority_category", "standard").strip()
    status = request.form.get("status", "active").strip()
    node_id_raw = request.form.get("node_id", "").strip()
    node_id = int(node_id_raw) if node_id_raw else None
    description = request.form.get("description", "").strip() or None

    errors = db.validate_allocation_zone(
        zone_name=zone_name,
        requested_demand_l=requested_demand_l,
        essential_min_l=essential_min_l,
        fairness_weight=fairness_weight,
        priority_category=priority_category,
        status=status,
        node_id=node_id,
    )
    if errors:
        for err in errors:
            flash(err, "error")
        return redirect(url_for("allocation"))

    try:
        zid = db.create_allocation_zone(
            zone_name=zone_name,
            requested_demand_l=float(requested_demand_l),
            essential_min_l=float(essential_min_l),
            fairness_weight=float(fairness_weight),
            priority_category=priority_category,
            status=status,
            node_id=node_id,
            description=description,
        )
        flash(f"Allocation zone '{zone_name}' created successfully.", "success")
    except sqlite3.IntegrityError:
        flash(f"A zone with name '{zone_name}' already exists. Please choose a unique name.", "error")
    except ValueError as exc:
        flash(str(exc), "error")

    return redirect(url_for("allocation"))


@app.route("/allocation/zones/<int:zone_id>/edit", methods=["GET", "POST"])
def allocation_zone_edit(zone_id: int):
    """View and update allocation zone configuration."""
    zone = db.get_allocation_zone_by_id(zone_id)
    if zone is None:
        flash("Zone not found.", "error")
        return redirect(url_for("allocation"))

    if request.method == "POST":
        zone_name = request.form.get("zone_name", "").strip()
        requested_demand_l = request.form.get("requested_demand_l", "").strip()
        essential_min_l = request.form.get("essential_min_l", "0").strip() or "0"
        fairness_weight = request.form.get("fairness_weight", "1.0").strip() or "1.0"
        priority_category = request.form.get("priority_category", "standard").strip()
        status = request.form.get("status", "active").strip()
        node_id_raw = request.form.get("node_id", "").strip()
        node_id = int(node_id_raw) if node_id_raw else None
        description = request.form.get("description", "").strip() or None

        errors = db.validate_allocation_zone(
            zone_name=zone_name,
            requested_demand_l=requested_demand_l,
            essential_min_l=essential_min_l,
            fairness_weight=fairness_weight,
            priority_category=priority_category,
            status=status,
            node_id=node_id,
        )
        if errors:
            for err in errors:
                flash(err, "error")
            return redirect(url_for("allocation_zone_edit", zone_id=zone_id))

        try:
            db.update_allocation_zone(
                zone_id=zone_id,
                zone_name=zone_name,
                requested_demand_l=float(requested_demand_l),
                essential_min_l=float(essential_min_l),
                fairness_weight=float(fairness_weight),
                priority_category=priority_category,
                status=status,
                node_id=node_id,
                description=description,
            )
            flash(f"Zone '{zone_name}' updated successfully.", "success")
            return redirect(url_for("allocation"))
        except sqlite3.IntegrityError:
            flash(f"A zone with name '{zone_name}' already exists.", "error")
            return redirect(url_for("allocation_zone_edit", zone_id=zone_id))
        except (KeyError, ValueError) as exc:
            flash(str(exc), "error")
            return redirect(url_for("allocation_zone_edit", zone_id=zone_id))

    all_nodes = db.get_all_nodes()
    allowed_categories = sorted(db.ALLOWED_ZONE_CATEGORIES)
    allowed_statuses = sorted(db.ALLOWED_ZONE_STATUSES)

    return render_template(
        "allocation_zone_edit.html",
        zone=zone,
        all_nodes=all_nodes,
        allowed_categories=allowed_categories,
        allowed_statuses=allowed_statuses,
    )


@app.route("/allocation/zones/<int:zone_id>/delete", methods=["POST"])
def allocation_zone_delete(zone_id: int):
    """Safely delete unreferenced zone or advise deactivation."""
    try:
        db.delete_allocation_zone(zone_id)
        flash("Zone deleted successfully.", "success")
    except RuntimeError:
        flash("Cannot delete zone: historical allocation records exist. Deactivate the zone instead by setting its status to 'inactive'.", "error")
    except KeyError:
        flash("Zone not found.", "error")

    return redirect(url_for("allocation"))


@app.route("/allocation/runs/<int:run_id>")
def allocation_run_details(run_id: int):
    """Display complete historical allocation result using preserved snapshots."""
    run = db.get_allocation_run_by_id(run_id)
    if run is None:
        flash(f"Allocation run #{run_id} not found.", "error")
        return redirect(url_for("allocation"))

    allocations = db.get_allocations_for_run(run_id)
    return render_template(
        "allocation_run_details.html",
        run=run,
        allocations=allocations,
    )


# ── Distribution Routes ───────────────────────────────────────────────────────

@app.route("/distribution")
def distribution():
    """
    Distribution dashboard – strictly read-only.
    Displays available allocation runs, source nodes, duration, routing strategy,
    previously recorded assessments, and summary of distribution feasibility.
    """
    selected_run_id = request.args.get("run_id", type=int)

    historical_runs = db.get_all_allocation_runs(limit=50)
    all_nodes = db.get_all_nodes()
    active_sources = [n for n in all_nodes if n["status"] == "active"]

    selected_run = None
    if selected_run_id:
        selected_run = db.get_allocation_run_by_id(selected_run_id)
    if selected_run is None and historical_runs:
        selected_run = historical_runs[0]
        selected_run_id = selected_run["run_id"]

    latest_assessment = (
        db.get_latest_delivery_assessment(run_id=selected_run_id)
        if selected_run_id
        else db.get_latest_delivery_assessment()
    )

    if latest_assessment is None:
        latest_assessment = db.get_latest_delivery_assessment()

    latest_zones = []
    latest_contention_warnings = []
    if latest_assessment:
        latest_zones = db.get_delivery_zone_assessments(latest_assessment["assessment_id"])
        if latest_assessment["contention_warnings_json"]:
            try:
                latest_contention_warnings = json.loads(latest_assessment["contention_warnings_json"])
            except Exception:
                latest_contention_warnings = []

    recent_assessments = db.get_all_delivery_assessments(limit=30)

    return render_template(
        "distribution.html",
        historical_runs=historical_runs,
        selected_run=selected_run,
        selected_run_id=selected_run_id,
        active_sources=active_sources,
        latest_assessment=latest_assessment,
        latest_zones=latest_zones,
        latest_contention_warnings=latest_contention_warnings,
        recent_assessments=recent_assessments,
        allowed_strategies=sorted(db.ALLOWED_ROUTING_STRATEGIES),
    )


@app.route("/distribution/run", methods=["POST"])
def distribution_run():
    """
    Execute delivery feasibility assessment and atomically persist results.
    Calls evaluate_and_record_run_distribution().
    """
    run_id_raw = request.form.get("allocation_run_id", request.form.get("run_id", "")).strip()
    source_node_id_raw = request.form.get("source_node_id", "").strip()
    duration_raw = request.form.get("duration_minutes", "60").strip()
    strategy_raw = request.form.get("routing_strategy", STRATEGY_SHORTEST_PATH).strip().upper()
    notes = request.form.get("notes", "").strip() or None

    if not run_id_raw:
        flash("Allocation run selection is required.", "error")
        return redirect(url_for("distribution"))

    try:
        run_id = int(run_id_raw)
    except ValueError:
        flash("Invalid allocation run ID.", "error")
        return redirect(url_for("distribution"))

    alloc_run = db.get_allocation_run_by_id(run_id)
    if alloc_run is None:
        flash(f"Allocation run #{run_id} does not exist.", "error")
        return redirect(url_for("distribution"))

    if not source_node_id_raw:
        flash("Water supply source node selection is required.", "error")
        return redirect(url_for("distribution", run_id=run_id))

    try:
        source_node_id = int(source_node_id_raw)
    except ValueError:
        flash("Invalid source node ID.", "error")
        return redirect(url_for("distribution", run_id=run_id))

    source_node = db.get_node_by_id(source_node_id)
    if source_node is None:
        flash(f"Source node #{source_node_id} does not exist.", "error")
        return redirect(url_for("distribution", run_id=run_id))
    if source_node["status"] != "active":
        flash(f"Source node #{source_node_id} ('{source_node['name']}') is inactive.", "error")
        return redirect(url_for("distribution", run_id=run_id))

    try:
        duration_val = float(duration_raw)
        if duration_val <= 0 or math.isnan(duration_val) or math.isinf(duration_val):
            flash("Delivery duration must be a positive number of minutes.", "error")
            return redirect(url_for("distribution", run_id=run_id))
    except ValueError:
        flash("Delivery duration must be a valid number of minutes.", "error")
        return redirect(url_for("distribution", run_id=run_id))

    if strategy_raw not in db.ALLOWED_ROUTING_STRATEGIES:
        flash(f"Invalid routing strategy '{strategy_raw}'. Choose SHORTEST_PATH or WIDEST_PATH.", "error")
        return redirect(url_for("distribution", run_id=run_id))

    allocations = db.get_allocations_for_run(run_id)
    if not allocations:
        flash(f"Allocation run #{run_id} contains no per-zone allocation records.", "warning")
        return redirect(url_for("distribution", run_id=run_id))

    try:
        assessment_id, plan = evaluate_and_record_run_distribution(
            run_id=run_id,
            source_node_id=source_node_id,
            duration_minutes=duration_val,
            routing_strategy=strategy_raw,
            notes=notes,
            db_module=db,
        )
        flash(
            f"Delivery feasibility assessment #{assessment_id} recorded successfully "
            f"({plan.overall_delivery_efficiency_pct:.1f}% efficiency via {plan.routing_strategy}).",
            "success",
        )
        return redirect(url_for("distribution_assessment_details", assessment_id=assessment_id))
    except Exception as exc:
        flash(f"Distribution assessment failed: {exc}", "error")
        return redirect(url_for("distribution", run_id=run_id))


@app.route("/distribution/assessments/<int:assessment_id>")
def distribution_assessment_details(assessment_id: int):
    """
    Display a saved historical delivery assessment.
    CRITICAL: Strictly uses persisted snapshot fields. Never recomputes historical assessment results.
    """
    assessment = db.get_delivery_assessment_by_id(assessment_id)
    if assessment is None:
        flash(f"Delivery assessment #{assessment_id} does not exist.", "error")
        return redirect(url_for("distribution"))

    run = db.get_allocation_run_by_id(assessment["run_id"])
    zone_rows = db.get_delivery_zone_assessments(assessment_id)

    contention_warnings = []
    if assessment["contention_warnings_json"]:
        try:
            contention_warnings = json.loads(assessment["contention_warnings_json"])
        except Exception:
            contention_warnings = []

    parsed_zones = []
    feasible_count = 0
    constrained_count = 0
    unreachable_count = 0
    unmapped_count = 0

    for z in zone_rows:
        z_dict = dict(z)
        try:
            z_dict["route_nodes"] = json.loads(z["route_node_ids"]) if z["route_node_ids"] else []
        except Exception:
            z_dict["route_nodes"] = []
        try:
            z_dict["route_pipes"] = json.loads(z["route_pipe_ids"]) if z["route_pipe_ids"] else []
        except Exception:
            z_dict["route_pipes"] = []

        status = z["feasibility_status"]
        if status == "FEASIBLE":
            feasible_count += 1
        elif status == "CAPACITY_CONSTRAINED":
            constrained_count += 1
        elif status == "UNREACHABLE":
            unreachable_count += 1
        elif status == "UNMAPPED":
            unmapped_count += 1

        parsed_zones.append(z_dict)

    run_assessments = db.get_delivery_assessments_for_run(assessment["run_id"])

    return render_template(
        "distribution_assessment_details.html",
        assessment=assessment,
        run=run,
        zones=parsed_zones,
        contention_warnings=contention_warnings,
        run_assessments=run_assessments,
        feasible_count=feasible_count,
        constrained_count=constrained_count,
        unreachable_count=unreachable_count,
        unmapped_count=unmapped_count,
    )


@app.route("/distribution/runs/<int:run_id>")
def distribution_run_history(run_id: int):
    """
    Display run-specific assessment history and shortcut to assess that run.
    """
    run = db.get_allocation_run_by_id(run_id)
    if run is None:
        flash(f"Allocation run #{run_id} not found.", "error")
        return redirect(url_for("distribution"))

    assessments = db.get_delivery_assessments_for_run(run_id)
    allocations = db.get_allocations_for_run(run_id)
    all_nodes = db.get_all_nodes()
    active_sources = [n for n in all_nodes if n["status"] == "active"]

    return render_template(
        "distribution_run_history.html",
        run=run,
        assessments=assessments,
        allocations=allocations,
        active_sources=active_sources,
        allowed_strategies=sorted(db.ALLOWED_ROUTING_STRATEGIES),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_int_or_none(raw: str) -> int | None:
    """Return int(raw) or None if raw is blank / non-numeric."""
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)
