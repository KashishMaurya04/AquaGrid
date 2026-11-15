"""
distribution/service.py – Integration Service between Fair Water Allocation and Distribution Engine.

Provides an end-to-end integration pipeline that:
1. Retrieves a recorded Fair Water Allocation run from SQLite.
2. Retrieves its per-zone allocation entitlements.
3. Retrieves active zone destination node mappings from allocation_zones.
4. Loads the network topology using the existing Graph loader.
5. Evaluates delivery feasibility via the pure DSA distribution engine.
6. Persists the resulting DistributionPlanResult and zone details atomically to SQLite.
"""

from __future__ import annotations

from typing import Any

from distribution.engine import (
    DistributionPlanResult,
    ZoneDeliveryInput,
    assess_distribution,
    STRATEGY_SHORTEST_PATH,
)
from dsa.graph import load_graph_from_db


def evaluate_and_record_run_distribution(
    run_id: int,
    source_node_id: int,
    duration_minutes: float | int = 60.0,
    routing_strategy: str = STRATEGY_SHORTEST_PATH,
    notes: str | None = None,
    db_module: Any = None,
) -> tuple[int, DistributionPlanResult]:
    """
    Retrieve an allocation run, evaluate its delivery feasibility across the physical network,
    and persist the assessment result atomically.

    Args:
        run_id: ID of the historical allocation run.
        source_node_id: ID of the water supply source node (e.g. Tank / Reservoir).
        duration_minutes: Available delivery period in minutes (must be strictly positive).
        routing_strategy: 'SHORTEST_PATH' or 'WIDEST_PATH'.
        db_module: Optional database access module (defaults to project's `database.py`).

    Returns:
        tuple of (assessment_id: int, plan_result: DistributionPlanResult).

    Raises:
        KeyError: If the allocation run #{run_id} does not exist.
        ValueError: If source node is invalid or inactive, or duration/strategy is invalid.
    """
    if db_module is None:
        import database as db_module

    # 1. Retrieve allocation run record
    run = db_module.get_allocation_run_by_id(run_id)
    if run is None:
        raise KeyError(f"Allocation run #{run_id} does not exist.")

    # 2. Retrieve source node and validate active status
    source_node = db_module.get_node_by_id(source_node_id)
    if source_node is None:
        raise ValueError(f"Source node #{source_node_id} does not exist in the network.")
    if source_node["status"] != "active":
        raise ValueError(
            f"Source node #{source_node_id} ('{source_node['name']}') is inactive."
        )

    # 3. Retrieve per-zone allocation entitlements for this run
    allocations = db_module.get_allocations_for_run(run_id)

    # 4. Build ZoneDeliveryInput collection preserving historical allocation snapshots.
    # CRITICAL: We strictly use the destination_node_id snapshot preserved in water_allocations
    # at the time the allocation run was executed.
    # If destination_node_id was not preserved (or unmapped at allocation time), it remains None,
    # causing the engine to explicitly classify the zone as UNMAPPED.
    # It NEVER silently substitutes the current operational mapping from allocation_zones.
    zone_inputs: list[ZoneDeliveryInput] = []
    for alloc in allocations:
        raw_zid = alloc["zone_id"]
        zid = int(raw_zid) if raw_zid is not None else None
        zname = str(alloc["zone_name_snapshot"])
        vol = float(alloc["allocated_amount_l"])

        dest_id = None
        if "destination_node_id" in alloc.keys() and alloc["destination_node_id"] is not None:
            dest_id = int(alloc["destination_node_id"])

        zone_inputs.append(
            ZoneDeliveryInput(
                zone_id=zid,
                zone_name=zname,
                destination_node_id=dest_id,
                allocated_volume_l=vol,
            )
        )

    # 6. Load network graph from the database
    graph = load_graph_from_db(db_module=db_module, directed=False)

    # 7. Evaluate delivery feasibility using pure DSA engine
    plan = assess_distribution(
        graph=graph,
        source_node_id=source_node_id,
        zones=zone_inputs,
        duration_minutes=duration_minutes,
        routing_strategy=routing_strategy,
    )

    # 8. Persist assessment and per-zone outcomes atomically
    assessment_id = db_module.record_distribution_assessment(
        plan_result=plan,
        run_id=run_id,
        notes=notes,
    )

    return assessment_id, plan

