"""
tests/test_simulator_timestamp_accuracy.py – Sensor Simulator Timestamp Accuracy and Timezone Test Suite.

Validates Phase 2 requirements:
1. Old November 1 readings followed by a new November 15 batch.
2. Existing future-dated historical readings (> 5 min) clamped to current time.
3. Rapid consecutive batch generation without interleaving or timestamp collisions.
4. No duplicate timestamps across consecutive batches.
5. All active sensors sharing one sampling-cycle timestamp within an iteration step.
6. Correct UTC-to-IST conversion ('15 Nov 2026, 01:04:30 PM IST').
7. Malformed and legacy timestamps gracefully handled without crashing.
8. Random seed reproducibility preserved.
9. Leak detection still receiving correctly ordered readings.
10. Historical readings remaining unchanged.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta
import pytest

import database as db
from app import (
    app,
    format_ist_timestamp,
    calculate_simulator_base_time,
    MAX_SIMULATION_LEAD,
)
from detection.engine import LeakDetector
from simulation.engine import SensorSimulator


@pytest.fixture
def tmp_sim_env():
    """Create an isolated temporary environment with fresh SQLite database for timestamp tests."""
    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test_timestamp.db")

    orig_db_path = db.DB_PATH
    orig_instance_dir = db.INSTANCE_DIR

    db.DB_PATH = temp_db_path
    db.INSTANCE_DIR = temp_dir
    db.init_db()

    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False

    with app.test_client() as client:
        yield {
            "client": client,
            "db_path": temp_db_path,
            "temp_dir": temp_dir,
        }

    # Reset any custom clock
    app.config.pop("SIMULATOR_CLOCK", None)
    db.DB_PATH = orig_db_path
    db.INSTANCE_DIR = orig_instance_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================================================
# 1. Old November 1 Readings Followed by a New November 15 Batch
# ==============================================================================

def test_old_readings_followed_by_new_current_batch(tmp_sim_env):
    """Verify that an older historical reading (Nov 1) does NOT anchor a new batch on Nov 15."""
    client = tmp_sim_env["client"]

    # Setup network and sensor
    n1 = db.create_node("Tank A", "tank", "North", "active")
    n2 = db.create_node("Junction B", "junction", "Central", "active")
    p1 = db.create_pipe("Mainline", n1, n2, "100.0", "500.0", "1.0", "active")
    s1 = db.create_sensor("Flow-1", p1, "flow_meter", "L/min", "active")

    # Insert historical reading from November 1, 2026
    old_ts = "2026-11-01T13:27:46.239525+00:00"
    db.record_sensor_readings_batch([{
        "sensor_id": s1,
        "reading_value": 350.0,
        "recorded_at": old_ts,
        "reading_type": "flow_meter",
        "unit": "L/min",
        "is_simulated": 1,
        "scenario": "normal",
    }])

    # Freeze clock to November 15, 2026 at 07:34:30 UTC (01:04:30 PM IST)
    frozen_now = datetime(2026, 11, 15, 7, 34, 30, tzinfo=timezone.utc)
    app.config["SIMULATOR_CLOCK"] = lambda: frozen_now

    # Generate new reading
    res = client.post("/sensor-simulator/generate", data={
        "mode": "normal",
        "count": "1",
    }, follow_redirects=True)
    assert res.status_code == 200

    latest = db.get_latest_sensor_readings(limit=1)[0]
    assert latest["recorded_at"].startswith("2026-11-15T07:34:30")
    assert not latest["recorded_at"].startswith("2026-11-01")


# ==============================================================================
# 2. Existing Future-Dated Historical Readings
# ==============================================================================

def test_future_dated_historical_reading_does_not_advance_indefinitely(tmp_sim_env):
    """Verify that an arbitrary future-dated reading (> 5 min ahead) does not drag new batches into the future."""
    client = tmp_sim_env["client"]

    n1 = db.create_node("Tank A", "tank", "North", "active")
    n2 = db.create_node("Junction B", "junction", "Central", "active")
    p1 = db.create_pipe("Mainline", n1, n2, "100.0", "500.0", "1.0", "active")
    s1 = db.create_sensor("Flow-1", p1, "flow_meter", "L/min", "active")

    # Insert an outlier reading dated in 2099
    future_ts = "2099-01-01T00:00:00.000000+00:00"
    db.record_sensor_readings_batch([{
        "sensor_id": s1,
        "reading_value": 350.0,
        "recorded_at": future_ts,
        "reading_type": "flow_meter",
        "unit": "L/min",
        "is_simulated": 1,
        "scenario": "normal",
    }])

    frozen_now = datetime(2026, 11, 15, 7, 34, 30, tzinfo=timezone.utc)
    app.config["SIMULATOR_CLOCK"] = lambda: frozen_now

    res = client.post("/sensor-simulator/generate", data={
        "mode": "normal",
        "count": "1",
    }, follow_redirects=True)
    assert res.status_code == 200

    # Query newly inserted reading (by ID)
    conn = sqlite3.connect(tmp_sim_env["db_path"])
    cur = conn.cursor()
    cur.execute("SELECT recorded_at FROM sensor_readings WHERE recorded_at != ? ORDER BY reading_id DESC LIMIT 1", (future_ts,))
    new_ts = cur.fetchone()[0]
    conn.close()

    assert new_ts.startswith("2026-11-15T07:34:30")


# ==============================================================================
# 3. Rapid Consecutive Batch Generation
# ==============================================================================

def test_rapid_consecutive_batch_generation_preserves_monotonicity(tmp_sim_env):
    """Verify that rapid consecutive generation without delay produces strictly non-overlapping timestamps."""
    client = tmp_sim_env["client"]

    n1 = db.create_node("Tank A", "tank", "North", "active")
    n2 = db.create_node("Junction B", "junction", "Central", "active")
    p1 = db.create_pipe("Mainline", n1, n2, "100.0", "500.0", "1.0", "active")
    s1 = db.create_sensor("Flow-1", p1, "flow_meter", "L/min", "active")

    frozen_now = datetime(2026, 11, 15, 7, 34, 30, tzinfo=timezone.utc)
    app.config["SIMULATOR_CLOCK"] = lambda: frozen_now

    # Batch 1 (10 readings: 07:34:30 to 07:39:00)
    client.post("/sensor-simulator/generate", data={"mode": "normal", "count": "10"}, follow_redirects=True)
    b1_readings = db.get_latest_sensor_readings(limit=10)
    b1_last_ts = b1_readings[0]["recorded_at"]  # Newest of batch 1

    # Batch 2 generated immediately with clock still at 07:34:30
    client.post("/sensor-simulator/generate", data={"mode": "normal", "count": "10"}, follow_redirects=True)
    all_readings = db.get_latest_sensor_readings(limit=20)
    b2_readings = all_readings[:10]  # Newest 10 readings are batch 2
    b2_first_ts = b2_readings[-1]["recorded_at"]  # Oldest of batch 2

    # Batch 2 must start strictly after Batch 1 ends
    assert b2_first_ts > b1_last_ts


# ==============================================================================
# 4. No Duplicate Timestamps Across Consecutive Batches
# ==============================================================================

def test_no_duplicate_timestamps_across_consecutive_single_readings(tmp_sim_env):
    """Verify that generating single readings in immediate succession produces strictly increasing timestamps."""
    client = tmp_sim_env["client"]

    n1 = db.create_node("Tank A", "tank", "North", "active")
    n2 = db.create_node("Junction B", "junction", "Central", "active")
    p1 = db.create_pipe("Mainline", n1, n2, "100.0", "500.0", "1.0", "active")
    s1 = db.create_sensor("Flow-1", p1, "flow_meter", "L/min", "active")

    frozen_now = datetime(2026, 11, 15, 7, 34, 30, tzinfo=timezone.utc)
    app.config["SIMULATOR_CLOCK"] = lambda: frozen_now

    client.post("/sensor-simulator/generate", data={"mode": "normal", "count": "1"}, follow_redirects=True)
    client.post("/sensor-simulator/generate", data={"mode": "normal", "count": "1"}, follow_redirects=True)

    readings = db.get_latest_sensor_readings(limit=2)
    assert len(readings) == 2
    ts_newer = readings[0]["recorded_at"]
    ts_older = readings[1]["recorded_at"]

    assert ts_newer > ts_older


# ==============================================================================
# 5. All Active Sensors Sharing One Sampling-Cycle Timestamp
# ==============================================================================

def test_all_sensors_in_sampling_cycle_share_same_timestamp(tmp_sim_env):
    """Verify that multiple active sensors in the same iteration step share the exact same timestamp."""
    client = tmp_sim_env["client"]

    n1 = db.create_node("Tank A", "tank", "North", "active")
    n2 = db.create_node("Junction B", "junction", "Central", "active")
    p1 = db.create_pipe("Mainline", n1, n2, "100.0", "500.0", "1.0", "active")
    s1 = db.create_sensor("Flow-1", p1, "flow_meter", "L/min", "active")
    s2 = db.create_sensor("Pressure-1", p1, "pressure_sensor", "bar", "active")
    s3 = db.create_sensor("Flow-2", p1, "flow_meter", "m³/h", "active")

    frozen_now = datetime(2026, 11, 15, 7, 34, 30, tzinfo=timezone.utc)
    app.config["SIMULATOR_CLOCK"] = lambda: frozen_now

    client.post("/sensor-simulator/generate", data={"mode": "normal", "count": "1"}, follow_redirects=True)

    readings = db.get_latest_sensor_readings(limit=3)
    assert len(readings) == 3
    # All 3 active sensors must share identical recorded_at
    timestamps = {r["recorded_at"] for r in readings}
    assert len(timestamps) == 1
    assert list(timestamps)[0].startswith("2026-11-15T07:34:30")


# ==============================================================================
# 6. Correct UTC-to-IST Conversion
# ==============================================================================

def test_utc_to_ist_timezone_conversion():
    """Verify exact formatting '15 Nov 2026, 01:04:30 PM IST' from UTC ISO timestamps."""
    # 07:34:30 UTC + 5:30 = 13:04:30 (01:04:30 PM IST)
    formatted = format_ist_timestamp("2026-11-15T07:34:30+00:00")
    assert formatted == "15 Nov 2026, 01:04:30 PM IST"

    # With Z suffix
    formatted_z = format_ist_timestamp("2026-11-15T07:34:30Z")
    assert formatted_z == "15 Nov 2026, 01:04:30 PM IST"

    # Historical reading
    formatted_hist = format_ist_timestamp("2026-11-01T13:27:46.239525+00:00")
    assert formatted_hist == "01 Nov 2026, 06:57:46 PM IST"


# ==============================================================================
# 7. Malformed Legacy Timestamps
# ==============================================================================

def test_malformed_legacy_timestamps_do_not_crash():
    """Verify that malformed or empty timestamp strings gracefully format without raising unhandled exceptions."""
    assert format_ist_timestamp(None) == "N/A"
    assert format_ist_timestamp("") == "N/A"
    assert format_ist_timestamp("corrupted-timestamp") == "corrupted-timestamp"

    # Naive ISO string (implicitly treated as UTC)
    res_naive = format_ist_timestamp("2026-11-15T07:34:30")
    assert res_naive == "15 Nov 2026, 01:04:30 PM IST"


# ==============================================================================
# 8. Random Seed Reproducibility
# ==============================================================================

def test_random_seed_reproducibility():
    """Verify that specifying seed produces identical values across independent simulator runs."""
    sensors = [
        {"sensor_id": 1, "sensor_name": "F1", "sensor_type": "flow_meter", "pipe_id": 1, "unit": "L/min", "status": "active"},
        {"sensor_id": 2, "sensor_name": "P1", "sensor_type": "pressure_sensor", "pipe_id": 1, "unit": "bar", "status": "active"},
    ]

    sim1 = SensorSimulator(seed=12345)
    r1 = sim1.generate_batch(sensors, mode="leak", leak_pipe_id=1, iterations=5)

    sim2 = SensorSimulator(seed=12345)
    r2 = sim2.generate_batch(sensors, mode="leak", leak_pipe_id=1, iterations=5)

    assert len(r1) == len(r2) == 10
    for a, b in zip(r1, r2):
        assert a.reading_value == b.reading_value
        assert a.is_anomaly == b.is_anomaly
        assert a.simulation_mode == b.simulation_mode


# ==============================================================================
# 9. Leak Detection Still Receiving Correctly Ordered Readings
# ==============================================================================

def test_leak_detection_with_updated_timestamps(tmp_sim_env):
    """Verify that leak detection correctly processes chronologically sequenced telemetry."""
    client = tmp_sim_env["client"]

    n1 = db.create_node("Tank A", "tank", "North", "active")
    n2 = db.create_node("Junction B", "junction", "Central", "active")
    p1 = db.create_pipe("Mainline", n1, n2, "100.0", "500.0", "1.0", "active")
    db.create_sensor("Flow-1", p1, "flow_meter", "L/min", "active")
    db.create_sensor("Pressure-1", p1, "pressure_sensor", "bar", "active")

    # Day 1: Normal readings
    t_nov1 = datetime(2026, 11, 1, 10, 0, 0, tzinfo=timezone.utc)
    app.config["SIMULATOR_CLOCK"] = lambda: t_nov1
    client.post("/sensor-simulator/generate", data={"mode": "normal", "count": "10", "seed": "42"}, follow_redirects=True)

    # Day 15: Leak readings on Pipe 1
    t_nov15 = datetime(2026, 11, 15, 7, 34, 30, tzinfo=timezone.utc)
    app.config["SIMULATOR_CLOCK"] = lambda: t_nov15
    client.post("/sensor-simulator/generate", data={"mode": "leak", "pipe_id": str(p1), "count": "10", "seed": "99"}, follow_redirects=True)

    # Run leak detection
    detector = LeakDetector()
    diagnostics = detector.run_detection_network(db, auto_record=True)
    assert len(diagnostics) >= 1
    diag_p1 = [d for d in diagnostics if d.pipe_id == p1][0]
    assert diag_p1.status == "suspected_leak"


# ==============================================================================
# 10. Historical Readings Remaining Unchanged
# ==============================================================================

def test_historical_readings_remain_unchanged(tmp_sim_env):
    """Verify that generating new batches does not modify, overwrite, or delete historical readings."""
    client = tmp_sim_env["client"]

    n1 = db.create_node("Tank A", "tank", "North", "active")
    n2 = db.create_node("Junction B", "junction", "Central", "active")
    p1 = db.create_pipe("Mainline", n1, n2, "100.0", "500.0", "1.0", "active")
    s1 = db.create_sensor("Flow-1", p1, "flow_meter", "L/min", "active")

    # Insert 3 historical readings
    hist_records = [
        {"sensor_id": s1, "reading_value": 348.5, "recorded_at": "2026-11-01T10:00:00+00:00", "reading_type": "flow_meter", "unit": "L/min", "is_simulated": 1, "scenario": "normal"},
        {"sensor_id": s1, "reading_value": 351.2, "recorded_at": "2026-11-01T10:00:30+00:00", "reading_type": "flow_meter", "unit": "L/min", "is_simulated": 1, "scenario": "normal"},
        {"sensor_id": s1, "reading_value": 349.8, "recorded_at": "2026-11-01T10:01:00+00:00", "reading_type": "flow_meter", "unit": "L/min", "is_simulated": 1, "scenario": "normal"},
    ]
    db.record_sensor_readings_batch(hist_records)

    conn = sqlite3.connect(tmp_sim_env["db_path"])
    cur = conn.cursor()
    cur.execute("SELECT reading_id, recorded_at, reading_value FROM sensor_readings ORDER BY reading_id ASC")
    before_rows = cur.fetchall()
    conn.close()

    # Generate new readings on Nov 15
    t_nov15 = datetime(2026, 11, 15, 7, 34, 30, tzinfo=timezone.utc)
    app.config["SIMULATOR_CLOCK"] = lambda: t_nov15
    client.post("/sensor-simulator/generate", data={"mode": "normal", "count": "1"}, follow_redirects=True)

    conn = sqlite3.connect(tmp_sim_env["db_path"])
    cur = conn.cursor()
    cur.execute("SELECT reading_id, recorded_at, reading_value FROM sensor_readings ORDER BY reading_id ASC LIMIT 3")
    after_hist_rows = cur.fetchall()
    conn.close()

    # Original historical rows must be identical
    assert before_rows == after_hist_rows


# ==============================================================================
# 11. Phase 2.1: Four Consecutive 10-Reading Batches Without Delay
# ==============================================================================

def test_four_consecutive_ten_reading_batches_strict_chronological_order(tmp_sim_env):
    """
    Verify that generating 4 consecutive 10-reading batches without delay produces
    strictly chronological ordering across all batches (batch1 < batch2 < batch3 < batch4)
    with zero duplicate timestamps.
    """
    client = tmp_sim_env["client"]

    n1 = db.create_node("Tank A", "tank", "North", "active")
    n2 = db.create_node("Junction B", "junction", "Central", "active")
    p1 = db.create_pipe("Mainline", n1, n2, "100.0", "500.0", "1.0", "active")
    s1 = db.create_sensor("Flow-1", p1, "flow_meter", "L/min", "active")

    # Freeze system clock at 2026-11-15 07:34:30 UTC
    frozen_now = datetime(2026, 11, 15, 7, 34, 30, tzinfo=timezone.utc)
    app.config["SIMULATOR_CLOCK"] = lambda: frozen_now

    # Generate 4 consecutive batches of 10 readings each with no delay
    batches = []
    for batch_num in range(1, 5):
        res = client.post(
            "/sensor-simulator/generate",
            data={"mode": "normal", "count": "10"},
            follow_redirects=True,
        )
        assert res.status_code == 200

        # Query readings recorded for this batch
        conn = sqlite3.connect(tmp_sim_env["db_path"])
        cur = conn.cursor()
        cur.execute(
            "SELECT reading_id, recorded_at FROM sensor_readings ORDER BY reading_id DESC LIMIT 10"
        )
        rows = cur.fetchall()
        conn.close()

        # rows are ordered DESC, so reverse to chronological ASC
        batch_ts = [r[1] for r in reversed(rows)]
        batches.append(batch_ts)

    batch1, batch2, batch3, batch4 = batches

    # Check each batch has exactly 10 readings
    assert len(batch1) == 10
    assert len(batch2) == 10
    assert len(batch3) == 10
    assert len(batch4) == 10

    # Verify inter-batch chronological ordering: batch1 < batch2 < batch3 < batch4
    assert max(batch1) < min(batch2), f"Batch 1 max ({max(batch1)}) not strictly less than Batch 2 min ({min(batch2)})"
    assert max(batch2) < min(batch3), f"Batch 2 max ({max(batch2)}) not strictly less than Batch 3 min ({min(batch3)})"
    assert max(batch3) < min(batch4), f"Batch 3 max ({max(batch3)}) not strictly less than Batch 4 min ({min(batch4)})"

    # Verify all 40 timestamps are strictly monotonically increasing
    all_readings = batch1 + batch2 + batch3 + batch4
    assert len(all_readings) == 40
    for i in range(len(all_readings) - 1):
        assert all_readings[i] < all_readings[i + 1], f"Timestamps not monotonic at index {i}: {all_readings[i]} vs {all_readings[i+1]}"

    # Verify zero duplicate timestamps across all 4 batches
    unique_ts = set(all_readings)
    assert len(unique_ts) == 40, f"Expected 40 unique timestamps, found {len(unique_ts)}"


# ==============================================================================
# 12. Phase 2.1: Legitimate Simulated Batch Ending > 5 Min Ahead Continues
# ==============================================================================

def test_simulation_stream_extending_beyond_five_minutes_ahead_continues_chronologically(tmp_sim_env):
    """
    Verify that an active simulation stream ending more than 5 minutes ahead of wall-clock time
    (e.g., 15 minutes ahead) continues monotonically without resetting to wall-clock time.
    """
    client = tmp_sim_env["client"]

    n1 = db.create_node("Tank A", "tank", "North", "active")
    n2 = db.create_node("Junction B", "junction", "Central", "active")
    p1 = db.create_pipe("Mainline", n1, n2, "100.0", "500.0", "1.0", "active")
    s1 = db.create_sensor("Flow-1", p1, "flow_meter", "L/min", "active")

    frozen_now = datetime(2026, 11, 15, 7, 34, 30, tzinfo=timezone.utc)
    app.config["SIMULATOR_CLOCK"] = lambda: frozen_now

    # Seed reading at 15 minutes ahead of frozen_now (07:49:30 UTC)
    ahead_15min = (frozen_now + timedelta(minutes=15)).isoformat()
    db.record_sensor_readings_batch([{
        "sensor_id": s1,
        "reading_value": 350.0,
        "recorded_at": ahead_15min,
        "reading_type": "flow_meter",
        "unit": "L/min",
        "is_simulated": 1,
        "scenario": "normal",
    }])

    # Generate a new reading batch (count=10)
    res = client.post(
        "/sensor-simulator/generate",
        data={"mode": "normal", "count": "10"},
        follow_redirects=True,
    )
    assert res.status_code == 200

    conn = sqlite3.connect(tmp_sim_env["db_path"])
    cur = conn.cursor()
    cur.execute("SELECT recorded_at FROM sensor_readings ORDER BY reading_id DESC LIMIT 10")
    new_rows = cur.fetchall()
    conn.close()

    new_ts = [r[0] for r in reversed(new_rows)]
    assert len(new_ts) == 10
    # All new readings must be strictly greater than ahead_15min
    assert min(new_ts) > ahead_15min
    # Ensure they start 1 second after ahead_15min
    assert new_ts[0] == (frozen_now + timedelta(minutes=15, seconds=1)).isoformat()


# ==============================================================================
# 13. Phase 2.1: calculate_simulator_base_time Matrix Validation
# ==============================================================================

def test_calculate_simulator_base_time_matrix():
    """Unit test for calculate_simulator_base_time covering all boundary conditions."""
    now = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)

    # 1. None or empty latest timestamp -> returns now
    assert calculate_simulator_base_time(now, None) == now
    assert calculate_simulator_base_time(now, "") == now

    # 2. Corrupted string -> returns now
    assert calculate_simulator_base_time(now, "invalid-timestamp") == now

    # 3. Genuinely stale historical timestamp (Nov 1) -> resets to now
    stale_ts = "2026-11-01T10:00:00+00:00"
    assert calculate_simulator_base_time(now, stale_ts) == now

    # 4. Immediate past (1 second ago) -> resets to now
    past_1s = (now - timedelta(seconds=1)).isoformat()
    assert calculate_simulator_base_time(now, past_1s) == now

    # 5. Exact match with now -> within horizon (0 <= 0 <= 2h), advances by 1s
    assert calculate_simulator_base_time(now, now.isoformat()) == now + timedelta(seconds=1)

    # 6. Recent simulation ahead by 5 minutes -> continues: last_dt + 1s
    ahead_5m = now + timedelta(minutes=5)
    assert calculate_simulator_base_time(now, ahead_5m.isoformat()) == ahead_5m + timedelta(seconds=1)

    # 7. Recent simulation ahead by 30 minutes -> continues: last_dt + 1s
    ahead_30m = now + timedelta(minutes=30)
    assert calculate_simulator_base_time(now, ahead_30m.isoformat()) == ahead_30m + timedelta(seconds=1)

    # 8. Boundary condition: exactly at MAX_SIMULATION_LEAD (2 hours ahead) -> continues
    ahead_2h = now + MAX_SIMULATION_LEAD
    assert calculate_simulator_base_time(now, ahead_2h.isoformat()) == ahead_2h + timedelta(seconds=1)

    # 9. Beyond MAX_SIMULATION_LEAD (2 hours + 1 second) -> resets to now
    ahead_over_2h = now + MAX_SIMULATION_LEAD + timedelta(seconds=1)
    assert calculate_simulator_base_time(now, ahead_over_2h.isoformat()) == now

    # 10. Arbitrary future legacy date (Year 2099) -> resets to now
    assert calculate_simulator_base_time(now, "2099-01-01T00:00:00+00:00") == now

