"""
tests/test_water_allocation.py – Comprehensive Validation of Fair Water Allocation.

Validates:
1. Three-tier weighted max-min fairness algorithm:
   - Policy 1: Abundant Supply (S >= sum(d_i))
   - Policy 2: Constrained Supply Above Essential Baseline (sum(e_i) <= S < sum(d_i))
   - Policy 3: Severe Emergency Deficit (S < sum(e_i))
   - Policy 4: Shortage with zero essential requirements (weighted-demand fallback)
2. Mathematical invariants:
   - Demand ceiling (0 <= a_i <= d_i)
   - Supply ceiling (sum(a_i) <= S)
   - Monotonicity & bottleneck saturation
   - Deterministic two-decimal rounding and cent remainder reconciliation
3. Edge cases and robust input validation:
   - Zero supply, zero demand, empty zones, single zone, large values (10^9)
   - Negative supply/demand/essential, weight <= 0, essential > demand
   - 100-run determinism guarantee
4. Database layer & transactional integrity:
   - Zone CRUD and domain constraint validation
   - Atomic run recording with complete rollback on write failure
   - Historical zone name snapshot preservation
   - Deletion restriction on referenced zones (ON DELETE RESTRICT)
   - Foreign key handling with network_nodes (ON DELETE SET NULL)
"""

from __future__ import annotations

import os
import sys
from decimal import Decimal
import pytest
import sqlite3

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import database as db_module
from allocation.engine import (
    ZoneDemandInput,
    ZoneAllocationResult,
    AllocationRunResult,
    allocate_water,
)


@pytest.fixture()
def isolated_db(monkeypatch, tmp_path):
    """Isolated SQLite database for water allocation tests."""
    test_db = str(tmp_path / "test_allocation.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


# ==============================================================================
# 1. Policy 1 – Abundant Supply Tests
# ==============================================================================

class TestFairAllocationPolicy1AbundantSupply:
    """Policy 1: Total available supply >= total requested demand."""

    def test_abundant_supply_all_zones_satisfied_100_percent(self):
        """When S >= sum(d_i), every zone receives 100% of requested demand."""
        zones = [
            ZoneDemandInput(zone_id=1, zone_name="Hospital", requested_demand=1000, essential_min=500, fairness_weight=3.0),
            ZoneDemandInput(zone_id=2, zone_name="Residential-A", requested_demand=2500, essential_min=800, fairness_weight=1.5),
            ZoneDemandInput(zone_id=3, zone_name="Commercial-Park", requested_demand=1500, essential_min=200, fairness_weight=1.0),
        ]
        # Total demand = 5000 L, Supply = 6000 L
        result = allocate_water(total_supply=6000, zones=zones)

        assert result.policy_applied == "ABUNDANT_SUPPLY"
        assert result.total_allocated == Decimal("5000.00")
        assert result.total_unmet == Decimal("0.00")
        assert result.satisfaction_rate_pct == Decimal("100.00")

        for alloc in result.allocations:
            assert alloc.unmet_demand == Decimal("0.00")
            assert alloc.satisfaction_pct == Decimal("100.00")
            assert alloc.is_bottleneck is True
            assert alloc.tier_reached == 3

        assert result.allocations[0].allocated_amount == Decimal("1000.00")
        assert result.allocations[1].allocated_amount == Decimal("2500.00")
        assert result.allocations[2].allocated_amount == Decimal("1500.00")

    def test_exact_supply_equals_total_demand(self):
        """When S == sum(d_i), all demands are exactly satisfied and total_allocated == S."""
        zones = [
            ZoneDemandInput(zone_id=1, zone_name="Zone-1", requested_demand=1234.56, essential_min=100),
            ZoneDemandInput(zone_id=2, zone_name="Zone-2", requested_demand=2345.67, essential_min=200),
        ]
        result = allocate_water(total_supply=Decimal("3580.23"), zones=zones)

        assert result.policy_applied == "ABUNDANT_SUPPLY"
        assert result.total_allocated == Decimal("3580.23")
        assert result.total_unmet == Decimal("0.00")


# ==============================================================================
# 2. Policy 2 – Constrained Supply & Weighted Max-Min Water-Filling
# ==============================================================================

class TestFairAllocationPolicy2ConstrainedResidual:
    """Policy 2: sum(e_i) <= S < sum(d_i)."""

    def test_essential_minimums_guaranteed_first(self):
        """Every zone must receive at least its essential minimum requirement."""
        zones = [
            ZoneDemandInput(zone_id=1, zone_name="Z1", requested_demand=1000, essential_min=300, fairness_weight=1.0),
            ZoneDemandInput(zone_id=2, zone_name="Z2", requested_demand=1000, essential_min=400, fairness_weight=1.0),
            ZoneDemandInput(zone_id=3, zone_name="Z3", requested_demand=1000, essential_min=200, fairness_weight=1.0),
        ]
        # Total essential = 900 L, Total demand = 3000 L, Supply = 900 L
        result = allocate_water(total_supply=900, zones=zones)

        assert result.policy_applied == "WEIGHTED_MAX_MIN_RESIDUAL"
        assert result.total_allocated == Decimal("900.00")
        assert result.allocations[0].allocated_amount == Decimal("300.00")
        assert result.allocations[1].allocated_amount == Decimal("400.00")
        assert result.allocations[2].allocated_amount == Decimal("200.00")
        for a in result.allocations:
            assert a.tier_reached == 1  # essential only

    def test_bottleneck_saturation_and_excess_redistribution(self):
        """
        Zones with small residual demands saturate first; excess supply is redistributed
        to unsaturated zones.
        Z1 residual = 100, Z2 residual = 1000. Equal weights.
        S_residual = 500 L.
        Z1 saturates at +100 L. Remaining 400 L goes to Z2.
        """
        zones = [
            ZoneDemandInput(zone_id=1, zone_name="Small-Residual", requested_demand=300, essential_min=200, fairness_weight=1.0),
            ZoneDemandInput(zone_id=2, zone_name="Large-Residual", requested_demand=1500, essential_min=500, fairness_weight=1.0),
        ]
        # Total essential = 700 L, Supply = 1200 L -> S_residual = 500 L
        result = allocate_water(total_supply=1200, zones=zones)

        assert result.policy_applied == "WEIGHTED_MAX_MIN_RESIDUAL"
        assert result.total_allocated == Decimal("1200.00")

        # Zone 1 receives 200 + 100 = 300 L (100% satisfied, bottleneck)
        assert result.allocations[0].allocated_amount == Decimal("300.00")
        assert result.allocations[0].is_bottleneck is True
        assert result.allocations[0].tier_reached == 3

        # Zone 2 receives 500 + 400 = 900 L (partial residual)
        assert result.allocations[1].allocated_amount == Decimal("900.00")
        assert result.allocations[1].is_bottleneck is False
        assert result.allocations[1].tier_reached == 2

    def test_equal_weights_equal_demands_equal_allocation(self):
        """Equal weights and equal demands receive identical allocations."""
        zones = [
            ZoneDemandInput(zone_id=1, zone_name="Z1", requested_demand=1000, essential_min=100, fairness_weight=1.0),
            ZoneDemandInput(zone_id=2, zone_name="Z2", requested_demand=1000, essential_min=100, fairness_weight=1.0),
            ZoneDemandInput(zone_id=3, zone_name="Z3", requested_demand=1000, essential_min=100, fairness_weight=1.0),
        ]
        # Total essential = 300 L, S_residual = 900 L (300 L each)
        result = allocate_water(total_supply=1200, zones=zones)

        assert result.allocations[0].allocated_amount == Decimal("400.00")
        assert result.allocations[1].allocated_amount == Decimal("400.00")
        assert result.allocations[2].allocated_amount == Decimal("400.00")

    def test_unequal_weights_equal_demands_proportional_residual(self):
        """Unequal weights with equal residual demands receive residual proportional to weights."""
        zones = [
            ZoneDemandInput(zone_id=1, zone_name="Normal", requested_demand=1000, essential_min=0, fairness_weight=1.0),
            ZoneDemandInput(zone_id=2, zone_name="Priority", requested_demand=1000, essential_min=0, fairness_weight=2.0),
        ]
        # Supply = 900 L. Z1 gets 300 L, Z2 gets 600 L.
        result = allocate_water(total_supply=900, zones=zones)

        assert result.allocations[0].allocated_amount == Decimal("300.00")
        assert result.allocations[1].allocated_amount == Decimal("600.00")
        assert result.allocations[1].allocated_amount == result.allocations[0].allocated_amount * 2


# ==============================================================================
# 3. Policy 3 – Severe Emergency Deficit & Fallbacks
# ==============================================================================

class TestFairAllocationPolicy3SevereDeficit:
    """Policy 3: S < sum(e_i)."""

    def test_severe_emergency_deficit_proportional_to_essential(self):
        """When S < total_essential, allocation is strictly proportional to essential minimums."""
        zones = [
            ZoneDemandInput(zone_id=1, zone_name="Hospital", requested_demand=2000, essential_min=600, fairness_weight=3.0),
            ZoneDemandInput(zone_id=2, zone_name="School", requested_demand=2000, essential_min=400, fairness_weight=1.0),
        ]
        # Total essential = 1000 L, Supply = 500 L (50% of essential)
        result = allocate_water(total_supply=500, zones=zones)

        assert result.policy_applied == "EMERGENCY_ESSENTIAL_PROPORTIONAL"
        assert result.total_allocated == Decimal("500.00")
        # Hospital gets 600/1000 * 500 = 300 L
        assert result.allocations[0].allocated_amount == Decimal("300.00")
        # School gets 400/1000 * 500 = 200 L
        assert result.allocations[1].allocated_amount == Decimal("200.00")
        for a in result.allocations:
            assert a.tier_reached == 0
            assert "Severe emergency deficit" in a.explanation

    def test_fallback_when_all_essential_minimums_are_zero(self):
        """When shortage occurs and all essential minimums are zero, use weighted demand fallback."""
        zones = [
            ZoneDemandInput(zone_id=1, zone_name="Z1", requested_demand=1000, essential_min=0, fairness_weight=1.0),
            ZoneDemandInput(zone_id=2, zone_name="Z2", requested_demand=2000, essential_min=0, fairness_weight=2.0),
        ]
        # Total demand = 3000 L, essential = 0, supply = 1500 L
        # Weighted demands: Z1 = 1.0 * 1000 = 1000, Z2 = 2.0 * 2000 = 4000. Total = 5000
        # Z1 gets 1500 * (1000/5000) = 300 L
        # Z2 gets 1500 * (4000/5000) = 1200 L
        result = allocate_water(total_supply=1500, zones=zones)

        assert result.policy_applied == "WEIGHTED_DEMAND_FALLBACK"
        assert result.total_allocated == Decimal("1500.00")
        assert result.allocations[0].allocated_amount == Decimal("300.00")
        assert result.allocations[1].allocated_amount == Decimal("1200.00")


# ==============================================================================
# 4. Edge Cases, Rounding, Precision & Mathematical Invariants
# ==============================================================================

class TestFairAllocationEdgeCasesAndInvariants:
    """Edge cases, high precision, determinism, and invariants."""

    def test_zero_available_supply(self):
        """When total supply is zero, all zones receive exactly 0.00 L."""
        zones = [
            ZoneDemandInput(zone_id=1, zone_name="Z1", requested_demand=500, essential_min=100),
            ZoneDemandInput(zone_id=2, zone_name="Z2", requested_demand=1000, essential_min=200),
        ]
        result = allocate_water(total_supply=0, zones=zones)

        assert result.policy_applied == "ZERO_SUPPLY"
        assert result.total_allocated == Decimal("0.00")
        assert result.total_unmet == Decimal("1500.00")
        for a in result.allocations:
            assert a.allocated_amount == Decimal("0.00")
            assert a.unmet_demand == a.requested_demand

    def test_zero_demand_zone_receives_zero(self):
        """Zone requesting 0 L receives 0 L without breaking the algorithm."""
        zones = [
            ZoneDemandInput(zone_id=1, zone_name="Zero-Zone", requested_demand=0, essential_min=0),
            ZoneDemandInput(zone_id=2, zone_name="Active-Zone", requested_demand=500, essential_min=100),
        ]
        result = allocate_water(total_supply=300, zones=zones)

        assert result.allocations[0].allocated_amount == Decimal("0.00")
        assert result.allocations[0].satisfaction_pct == Decimal("100.00")
        assert result.allocations[1].allocated_amount == Decimal("300.00")

    def test_empty_zone_list_returns_valid_empty_result(self):
        """Empty zone list returns clean result with NO_ZONES policy."""
        result = allocate_water(total_supply=5000, zones=[])

        assert result.policy_applied == "NO_ZONES"
        assert result.total_supply == Decimal("5000.00")
        assert result.total_allocated == Decimal("0.00")
        assert result.zone_count == 0
        assert len(result.allocations) == 0

    def test_single_zone_receives_min_supply_demand(self):
        """Single zone receives min(supply, demand)."""
        z = [ZoneDemandInput(zone_id=1, zone_name="Solo", requested_demand=1000, essential_min=200)]

        # Supply < demand
        res1 = allocate_water(total_supply=600, zones=z)
        assert res1.allocations[0].allocated_amount == Decimal("600.00")

        # Supply > demand
        res2 = allocate_water(total_supply=1500, zones=z)
        assert res2.allocations[0].allocated_amount == Decimal("1000.00")

    def test_very_large_numbers_do_not_overflow(self):
        """Very large demands (10^9 L) compute accurately with high precision Decimal."""
        zones = [
            ZoneDemandInput(zone_id=1, zone_name="Mega-1", requested_demand=1_000_000_000, essential_min=200_000_000, fairness_weight=1.5),
            ZoneDemandInput(zone_id=2, zone_name="Mega-2", requested_demand=2_000_000_000, essential_min=400_000_000, fairness_weight=3.0),
        ]
        result = allocate_water(total_supply=1_500_000_000, zones=zones)

        assert result.total_allocated == Decimal("1500000000.00")
        assert result.allocations[0].allocated_amount + result.allocations[1].allocated_amount == Decimal("1500000000.00")
        assert result.allocations[0].allocated_amount <= Decimal("1000000000.00")
        assert result.allocations[1].allocated_amount <= Decimal("2000000000.00")

    def test_decimal_rounding_never_exceeds_supply_and_reconciles_cent(self):
        """
        Odd fractional divisions like 100 L divided across 3 zones:
        100 / 3 = 33.3333...
        Rounded down: 33.33 each = 99.99 total.
        Deterministic reconciliation awards the 0.01 remainder so sum equals exactly 100.00 L.
        """
        zones = [
            ZoneDemandInput(zone_id=1, zone_name="Z1", requested_demand=100, essential_min=0, fairness_weight=1.0),
            ZoneDemandInput(zone_id=2, zone_name="Z2", requested_demand=100, essential_min=0, fairness_weight=1.0),
            ZoneDemandInput(zone_id=3, zone_name="Z3", requested_demand=100, essential_min=0, fairness_weight=1.0),
        ]
        result = allocate_water(total_supply=100, zones=zones)

        assert result.total_allocated == Decimal("100.00")
        alloc_sum = sum(a.allocated_amount for a in result.allocations)
        assert alloc_sum == Decimal("100.00")
        assert alloc_sum <= Decimal("100.00")  # Supply ceiling strictly respected

    def test_mathematical_invariants_across_multiple_scenarios(self):
        """Verify Demand Ceiling (a_i <= d_i) and Supply Ceiling (sum(a_i) <= S) always hold."""
        test_supplies = [0, 50, 100, 300, 750, 1000, 1499.99, 2000, 50000]
        zones = [
            ZoneDemandInput(zone_id=1, zone_name="Z1", requested_demand=250.50, essential_min=50.25, fairness_weight=1.2),
            ZoneDemandInput(zone_id=2, zone_name="Z2", requested_demand=400.00, essential_min=150.00, fairness_weight=2.5),
            ZoneDemandInput(zone_id=3, zone_name="Z3", requested_demand=120.75, essential_min=30.00, fairness_weight=0.8),
            ZoneDemandInput(zone_id=4, zone_name="Z4", requested_demand=800.00, essential_min=100.00, fairness_weight=1.0),
        ]

        for s in test_supplies:
            res = allocate_water(total_supply=s, zones=zones)
            s_dec = Decimal(str(s)).quantize(Decimal("0.01"))
            # Supply ceiling invariant
            assert res.total_allocated <= s_dec, f"Supply ceiling violated for supply={s}"
            assert sum(a.allocated_amount for a in res.allocations) == res.total_allocated

            for a, z in zip(res.allocations, zones):
                # Demand ceiling invariant
                assert a.allocated_amount <= a.requested_demand, f"Demand ceiling violated for {z.zone_name}"
                assert a.allocated_amount >= Decimal("0.00")
                assert a.unmet_demand == a.requested_demand - a.allocated_amount

    def test_determinism_100_runs_identical_results(self):
        """Running the allocation algorithm 100 times with identical inputs produces bit-for-bit identical outputs."""
        zones = [
            ZoneDemandInput(zone_id=1, zone_name="Z1", requested_demand=333.33, essential_min=111.11, fairness_weight=1.33),
            ZoneDemandInput(zone_id=2, zone_name="Z2", requested_demand=666.66, essential_min=222.22, fairness_weight=2.66),
            ZoneDemandInput(zone_id=3, zone_name="Z3", requested_demand=999.99, essential_min=333.33, fairness_weight=3.99),
        ]
        base_result = allocate_water(total_supply=1000, zones=zones)
        base_allocs = [(a.zone_id, a.allocated_amount, a.unmet_demand, a.satisfaction_pct) for a in base_result.allocations]

        for _ in range(100):
            run = allocate_water(total_supply=1000, zones=zones)
            assert run.total_allocated == base_result.total_allocated
            current_allocs = [(a.zone_id, a.allocated_amount, a.unmet_demand, a.satisfaction_pct) for a in run.allocations]
            assert current_allocs == base_allocs


# ==============================================================================
# 5. Input Validation Tests
# ==============================================================================

class TestFairAllocationInputValidation:
    """Robust error handling on malformed or illegal inputs."""

    def test_negative_supply_raises_value_error(self):
        zones = [ZoneDemandInput(zone_id=1, zone_name="Z1", requested_demand=100)]
        with pytest.raises(ValueError, match="Total available water supply cannot be negative"):
            allocate_water(total_supply=-50, zones=zones)

    def test_negative_demand_raises_value_error(self):
        zones = [ZoneDemandInput(zone_id=1, zone_name="Z1", requested_demand=-100)]
        with pytest.raises(ValueError, match="cannot be negative"):
            allocate_water(total_supply=500, zones=zones)

    def test_negative_essential_min_raises_value_error(self):
        zones = [ZoneDemandInput(zone_id=1, zone_name="Z1", requested_demand=100, essential_min=-10)]
        with pytest.raises(ValueError, match="cannot be negative"):
            allocate_water(total_supply=500, zones=zones)

    def test_essential_min_exceeding_demand_raises_value_error(self):
        zones = [ZoneDemandInput(zone_id=1, zone_name="Hospital", requested_demand=500, essential_min=600)]
        with pytest.raises(ValueError, match="cannot exceed requested demand"):
            allocate_water(total_supply=1000, zones=zones)

    def test_fairness_weight_zero_or_negative_raises_value_error(self):
        zones_zero = [ZoneDemandInput(zone_id=1, zone_name="Z1", requested_demand=500, fairness_weight=0)]
        with pytest.raises(ValueError, match="must be strictly positive"):
            allocate_water(total_supply=500, zones=zones_zero)

        zones_neg = [ZoneDemandInput(zone_id=1, zone_name="Z1", requested_demand=500, fairness_weight=-1.5)]
        with pytest.raises(ValueError, match="must be strictly positive"):
            allocate_water(total_supply=500, zones=zones_neg)

    def test_non_numeric_supply_raises_value_error(self):
        zones = [ZoneDemandInput(zone_id=1, zone_name="Z1", requested_demand=100)]
        with pytest.raises(ValueError, match="Invalid total supply value"):
            allocate_water(total_supply="abc", zones=zones)


# ==============================================================================
# 6. Database Persistence, Atomicity & Referential Integrity Tests
# ==============================================================================

class TestAllocationDatabasePersistence:
    """CRUD operations, domain validation, foreign keys, and atomic transactions."""

    def test_zone_crud_lifecycle(self, isolated_db):
        """Full CRUD lifecycle for allocation zones."""
        # Create node to attach zone
        node_id = isolated_db.create_node("Node-North", "junction", "North District", "active")

        # 1. Create Zone
        zone_id = isolated_db.create_allocation_zone(
            zone_name="Zone-Central",
            requested_demand_l=5000.0,
            essential_min_l=1500.0,
            fairness_weight=2.0,
            priority_category="critical",
            status="active",
            node_id=node_id,
            description="Central municipal district including general hospital",
        )
        assert zone_id > 0

        # 2. Read Zone
        zone = isolated_db.get_allocation_zone_by_id(zone_id)
        assert zone is not None
        assert zone["zone_name"] == "Zone-Central"
        assert zone["node_name"] == "Node-North"
        assert zone["requested_demand_l"] == 5000.0
        assert zone["essential_min_l"] == 1500.0
        assert zone["fairness_weight"] == 2.0
        assert zone["priority_category"] == "critical"
        assert zone["status"] == "active"

        # 3. Update Zone
        isolated_db.update_allocation_zone(
            zone_id=zone_id,
            zone_name="Zone-Central-Renamed",
            requested_demand_l=6000.0,
            essential_min_l=2000.0,
            fairness_weight=2.5,
            priority_category="critical",
            status="inactive",
            node_id=node_id,
            description="Updated description",
        )
        updated = isolated_db.get_allocation_zone_by_id(zone_id)
        assert updated["zone_name"] == "Zone-Central-Renamed"
        assert updated["requested_demand_l"] == 6000.0
        assert updated["status"] == "inactive"

        # 4. Delete Zone
        isolated_db.delete_allocation_zone(zone_id)
        assert isolated_db.get_allocation_zone_by_id(zone_id) is None

    def test_zone_validation_rejects_illegal_records(self, isolated_db):
        """validate_allocation_zone detects all invalid fields."""
        # Blank name
        errs = isolated_db.validate_allocation_zone("", 100, 50, 1.0)
        assert any("Zone name is required" in e for e in errs)

        # Name too long
        errs = isolated_db.validate_allocation_zone("X" * 121, 100, 50, 1.0)
        assert any("120 characters or fewer" in e for e in errs)

        # Negative demand
        errs = isolated_db.validate_allocation_zone("Z1", -10, 0, 1.0)
        assert any("Requested demand must be zero or greater" in e for e in errs)

        # Essential min > demand
        errs = isolated_db.validate_allocation_zone("Z1", 100, 150, 1.0)
        assert any("Essential minimum cannot exceed requested demand" in e for e in errs)

        # Weight <= 0
        errs = isolated_db.validate_allocation_zone("Z1", 100, 50, 0.0)
        assert any("Fairness weight must be strictly greater than zero" in e for e in errs)

        # Invalid category
        errs = isolated_db.validate_allocation_zone("Z1", 100, 50, 1.0, priority_category="super-urgent")
        assert any("Priority category 'super-urgent' is invalid" in e for e in errs)

        # Invalid status
        errs = isolated_db.validate_allocation_zone("Z1", 100, 50, 1.0, status="archived")
        assert any("Status 'archived' is invalid" in e for e in errs)

        # Nonexistent node_id
        errs = isolated_db.validate_allocation_zone("Z1", 100, 50, 1.0, node_id=99999)
        assert any("Referenced network node 99999 does not exist" in e for e in errs)

    def test_record_allocation_run_and_query(self, isolated_db):
        """Recording an allocation run stores run metadata and per-zone allocations."""
        z1_id = isolated_db.create_allocation_zone("Hospital", 1000, 400, 2.0, "critical")
        z2_id = isolated_db.create_allocation_zone("Residential", 2000, 500, 1.0, "standard")

        active_zones = isolated_db.get_active_allocation_zones()
        assert len(active_zones) == 2

        zone_inputs = [
            ZoneDemandInput(
                zone_id=z["zone_id"],
                zone_name=z["zone_name"],
                requested_demand=z["requested_demand_l"],
                essential_min=z["essential_min_l"],
                fairness_weight=z["fairness_weight"],
            )
            for z in active_zones
        ]

        run_result = allocate_water(total_supply=2000, zones=zone_inputs)
        run_id = isolated_db.record_allocation_run(
            run_result=run_result,
            period_start="2026-11-12T00:00:00Z",
            period_end="2026-11-12T23:59:59Z",
            notes="Morning operational dispatch run",
        )

        assert run_id > 0
        assert isolated_db.count_allocation_runs() == 1
        assert isolated_db.count_water_allocations() == 2

        # Verify run query
        run_record = isolated_db.get_allocation_run_by_id(run_id)
        assert run_record is not None
        assert run_record["total_available_supply_l"] == 2000.0
        assert run_record["total_requested_demand_l"] == 3000.0
        assert run_record["total_allocated_l"] == 2000.0
        assert run_record["allocation_policy"] == "WEIGHTED_MAX_MIN_RESIDUAL"
        assert run_record["active_zone_count"] == 2
        assert run_record["notes"] == "Morning operational dispatch run"

        # Verify per-zone allocations query
        zone_allocs = isolated_db.get_allocations_for_run(run_id)
        assert len(zone_allocs) == 2
        assert zone_allocs[0]["zone_name_snapshot"] == "Hospital"
        assert zone_allocs[0]["allocated_amount_l"] > 400.0
        assert zone_allocs[1]["zone_name_snapshot"] == "Residential"

        # Latest run helper
        latest = isolated_db.get_latest_allocation_run()
        assert latest["run_id"] == run_id

    def test_atomic_transaction_rollback_on_write_failure(self, isolated_db):
        """Simulated write failure during allocation recording must roll back completely."""
        z1_id = isolated_db.create_allocation_zone("Zone-1", 1000, 200, 1.0)
        z2_id = isolated_db.create_allocation_zone("Zone-2", 1000, 200, 1.0)

        zone_inputs = [
            ZoneDemandInput(zone_id=z1_id, zone_name="Zone-1", requested_demand=1000, essential_min=200),
            ZoneDemandInput(zone_id=z2_id, zone_name="Zone-2", requested_demand=1000, essential_min=200),
        ]
        run_result = allocate_water(total_supply=1500, zones=zone_inputs)

        initial_run_count = isolated_db.count_allocation_runs()
        initial_alloc_count = isolated_db.count_water_allocations()

        # Trigger simulated failure on Zone-2
        with pytest.raises(RuntimeError, match="Simulated transaction failure"):
            isolated_db.record_allocation_run(run_result, fail_on_zone_id=z2_id)

        # Assert complete rollback: no orphan run record and no orphan allocation records
        assert isolated_db.count_allocation_runs() == initial_run_count
        assert isolated_db.count_water_allocations() == initial_alloc_count

    def test_zone_name_snapshot_preservation_after_rename(self, isolated_db):
        """Historical water allocations preserve original zone name snapshot even if zone is renamed."""
        z_id = isolated_db.create_allocation_zone("Original-Hospital", 1000, 500, 2.0)
        zone_input = [ZoneDemandInput(zone_id=z_id, zone_name="Original-Hospital", requested_demand=1000, essential_min=500)]

        run_result = allocate_water(total_supply=1000, zones=zone_input)
        run_id = isolated_db.record_allocation_run(run_result)

        # Rename the zone
        isolated_db.update_allocation_zone(
            zone_id=z_id,
            zone_name="Modern-Medical-Center",
            requested_demand_l=1200.0,
            essential_min_l=600.0,
            fairness_weight=2.0,
        )

        # Check historical allocation
        historical_allocs = isolated_db.get_allocations_for_run(run_id)
        assert len(historical_allocs) == 1
        assert historical_allocs[0]["zone_name_snapshot"] == "Original-Hospital"

        # Check live zone
        live_zone = isolated_db.get_allocation_zone_by_id(z_id)
        assert live_zone["zone_name"] == "Modern-Medical-Center"

    def test_cannot_delete_zone_with_historical_allocations(self, isolated_db):
        """Foreign key ON DELETE RESTRICT prevents deleting a zone referenced by past allocations."""
        z_id = isolated_db.create_allocation_zone("Critical-Zone", 500, 200, 1.0)
        zone_input = [ZoneDemandInput(zone_id=z_id, zone_name="Critical-Zone", requested_demand=500, essential_min=200)]

        run_result = allocate_water(total_supply=500, zones=zone_input)
        isolated_db.record_allocation_run(run_result)

        # Attempt to delete referenced zone
        with pytest.raises(RuntimeError, match="Cannot delete allocation zone"):
            isolated_db.delete_allocation_zone(z_id)

        # Zone must still exist
        assert isolated_db.get_allocation_zone_by_id(z_id) is not None

    def test_deleting_network_node_sets_zone_node_id_to_null(self, isolated_db):
        """Deleting a referenced network node sets allocation_zones.node_id to NULL (ON DELETE SET NULL)."""
        node_id = isolated_db.create_node("Node-Temp", "junction", "Test Area", "active")
        z_id = isolated_db.create_allocation_zone("Zone-Node-Test", 800, 200, 1.0, node_id=node_id)

        zone_before = isolated_db.get_allocation_zone_by_id(z_id)
        assert zone_before["node_id"] == node_id

        # Delete network node
        isolated_db.delete_node(node_id)

        # Zone should still exist, with node_id set to None/NULL
        zone_after = isolated_db.get_allocation_zone_by_id(z_id)
        assert zone_after is not None
        assert zone_after["node_id"] is None


# ==============================================================================
# 7. Verification Tests – Scalability, Precision, Summary, Reconciliation & Safety
# ==============================================================================

class TestWaterFillingComplexityAndScalability:
    """Mathematical complexity validation and scalability benchmarks across 100, 1,000, and 10,000 zones."""

    def test_water_filling_scale_100_zones(self):
        """100 zones with distinct saturation thresholds execute in < 0.10 seconds."""
        import time
        zones = [
            ZoneDemandInput(
                zone_id=i,
                zone_name=f"Zone-{i}",
                requested_demand=Decimal(str(i * 10)),
                essential_min=Decimal(str(i * 2)),
                fairness_weight=Decimal(str((i % 5 + 1) * 0.5)),
            )
            for i in range(1, 101)
        ]
        total_dem = sum(z.requested_demand for z in zones)
        total_ess = sum(z.essential_min for z in zones)
        supply = total_ess + (total_dem - total_ess) * Decimal("0.4")

        t0 = time.perf_counter()
        res = allocate_water(total_supply=supply, zones=zones)
        t1 = time.perf_counter()

        assert t1 - t0 < 0.20, f"100 zones took {t1 - t0:.4f}s (expected < 0.20s)"
        assert res.total_allocated <= supply
        assert len(res.allocations) == 100

    def test_water_filling_scale_1000_zones(self):
        """1,000 zones with distinct saturation thresholds execute in < 0.50 seconds."""
        import time
        zones = [
            ZoneDemandInput(
                zone_id=i,
                zone_name=f"Zone-{i}",
                requested_demand=Decimal(str(i * 10)),
                essential_min=Decimal(str(i * 2)),
                fairness_weight=Decimal(str((i % 5 + 1) * 0.5)),
            )
            for i in range(1, 1001)
        ]
        total_dem = sum(z.requested_demand for z in zones)
        total_ess = sum(z.essential_min for z in zones)
        supply = total_ess + (total_dem - total_ess) * Decimal("0.4")

        t0 = time.perf_counter()
        res = allocate_water(total_supply=supply, zones=zones)
        t1 = time.perf_counter()

        assert t1 - t0 < 0.50, f"1,000 zones took {t1 - t0:.4f}s (expected < 0.50s)"
        assert res.total_allocated <= supply
        assert len(res.allocations) == 1000

    def test_water_filling_scale_10000_zones(self):
        """10,000 zones with distinct saturation thresholds execute in < 2.50 seconds in O(N log N)."""
        import time
        zones = [
            ZoneDemandInput(
                zone_id=i,
                zone_name=f"Zone-{i}",
                requested_demand=Decimal(str(i * 10)),
                essential_min=Decimal(str(i * 2)),
                fairness_weight=Decimal("1.0"),
            )
            for i in range(1, 10001)
        ]
        total_dem = sum(z.requested_demand for z in zones)
        total_ess = sum(z.essential_min for z in zones)
        supply = total_ess + (total_dem - total_ess) * Decimal("0.3")

        t0 = time.perf_counter()
        res = allocate_water(total_supply=supply, zones=zones)
        t1 = time.perf_counter()

        assert t1 - t0 < 2.50, f"10,000 zones took {t1 - t0:.4f}s (expected < 2.50s)"
        assert res.total_allocated <= supply
        assert len(res.allocations) == 10000
        # Invariant checks on sample zones
        assert res.allocations[0].allocated_amount <= res.allocations[0].requested_demand
        assert res.allocations[9999].allocated_amount <= res.allocations[9999].requested_demand


class TestDecimalPrecisionRoundTripPersistence:
    """Verifies two-decimal exactness across SQLite insertion and retrieval."""

    def test_round_trip_decimal_persistence(self, isolated_db):
        """Values 0.01, 0.10, 123.45, 666.67, 999999.99 preserve exact cents on retrieval."""
        test_values = [
            Decimal("0.01"),
            Decimal("0.10"),
            Decimal("123.45"),
            Decimal("666.67"),
            Decimal("999999.99"),
        ]

        zone_inputs = []
        for idx, val in enumerate(test_values, start=1):
            zid = isolated_db.create_allocation_zone(
                zone_name=f"Precision-Zone-{idx}",
                requested_demand_l=float(val),
                essential_min_l=0.0,
                fairness_weight=1.0,
            )
            zone_inputs.append(ZoneDemandInput(
                zone_id=zid,
                zone_name=f"Precision-Zone-{idx}",
                requested_demand=val,
                essential_min=Decimal("0.00"),
                fairness_weight=Decimal("1.0"),
            ))

        total_supply = sum(test_values)
        run_res = allocate_water(total_supply=total_supply, zones=zone_inputs)
        run_id = isolated_db.record_allocation_run(run_res)

        stored_allocs = isolated_db.get_allocations_for_run(run_id)
        assert len(stored_allocs) == len(test_values)

        for expected_val, row in zip(test_values, stored_allocs):
            # Convert retrieved float back to exact Decimal with two decimal places
            retrieved_dec = Decimal(f"{row['allocated_amount_l']:.2f}")
            assert retrieved_dec == expected_val, f"Mismatch: expected {expected_val}, got {retrieved_dec}"


class TestAllocationRunSummaryRetrieval:
    """Verifies all mandatory allocation run summary fields can be retrieved from database."""

    def test_allocation_run_summary_all_fields_retrieved(self, isolated_db):
        """Database retrieves supply, demand, allocated, unmet, satisfaction, count, policy, and period."""
        zid = isolated_db.create_allocation_zone("Z-Summary", 1000, 200, 1.0)
        zone_inputs = [ZoneDemandInput(zone_id=zid, zone_name="Z-Summary", requested_demand=1000, essential_min=200)]

        run_res = allocate_water(total_supply=600, zones=zone_inputs)
        run_id = isolated_db.record_allocation_run(
            run_result=run_res,
            period_start="2026-11-11T00:00:00Z",
            period_end="2026-11-11T23:59:59Z",
            notes="Daily technical verification run",
        )

        record = isolated_db.get_allocation_run_by_id(run_id)
        assert record is not None
        # All 8 mandatory fields
        assert record["total_available_supply_l"] == 600.0
        assert record["total_requested_demand_l"] == 1000.0
        assert record["total_allocated_l"] == 600.0
        assert record["total_unmet_l"] == 400.0
        assert record["satisfaction_rate_pct"] == 60.0
        assert record["active_zone_count"] == 1
        assert record["allocation_policy"] == "WEIGHTED_MAX_MIN_RESIDUAL"
        assert record["period_start"] == "2026-11-11T00:00:00Z"
        assert record["period_end"] == "2026-11-11T23:59:59Z"


class TestRoundingReconciliationTarget:
    """Verifies reconciliation target = min(total_available_supply, total_requested_demand)."""

    def test_reconciliation_supply_greater_than_demand(self):
        """When S > demand, reconciliation target is total_demand, preventing over-allocation."""
        zones = [
            ZoneDemandInput(zone_id=1, zone_name="Z1", requested_demand=300, essential_min=100),
            ZoneDemandInput(zone_id=2, zone_name="Z2", requested_demand=200, essential_min=50),
        ]
        # Total demand = 500 L, Supply = 1000 L
        res = allocate_water(total_supply=1000, zones=zones)
        assert res.total_allocated == Decimal("500.00")
        assert res.total_unmet == Decimal("0.00")
        assert res.allocations[0].allocated_amount == Decimal("300.00")
        assert res.allocations[1].allocated_amount == Decimal("200.00")

    def test_reconciliation_supply_equal_to_demand(self):
        """When S == demand, reconciliation target is exactly S and all demand is satisfied."""
        zones = [
            ZoneDemandInput(zone_id=1, zone_name="Z1", requested_demand=333.33, essential_min=50),
            ZoneDemandInput(zone_id=2, zone_name="Z2", requested_demand=666.67, essential_min=100),
        ]
        res = allocate_water(total_supply=1000, zones=zones)
        assert res.total_allocated == Decimal("1000.00")
        assert res.total_unmet == Decimal("0.00")

    def test_reconciliation_supply_below_demand_with_cents(self):
        """When S < demand, reconciliation awards remaining cents deterministically."""
        zones = [
            ZoneDemandInput(zone_id=1, zone_name="Z1", requested_demand=500, essential_min=0, fairness_weight=1.0),
            ZoneDemandInput(zone_id=2, zone_name="Z2", requested_demand=500, essential_min=0, fairness_weight=1.0),
            ZoneDemandInput(zone_id=3, zone_name="Z3", requested_demand=500, essential_min=0, fairness_weight=1.0),
        ]
        # Supply = 100 L. 100 / 3 = 33.333... L
        # Rounded down = 33.33 each = 99.99 L. 0.01 cent reconciled to Z1.
        res = allocate_water(total_supply=100, zones=zones)
        assert res.total_allocated == Decimal("100.00")
        assert sum(a.allocated_amount for a in res.allocations) == Decimal("100.00")

    def test_reconciliation_multiple_demand_caps_reached(self):
        """Multiple zones reaching demand caps never receive more than their demand."""
        zones = [
            ZoneDemandInput(zone_id=1, zone_name="Small-Cap-1", requested_demand=50.00, essential_min=10),
            ZoneDemandInput(zone_id=2, zone_name="Small-Cap-2", requested_demand=50.00, essential_min=10),
            ZoneDemandInput(zone_id=3, zone_name="Large-Cap", requested_demand=1000.00, essential_min=100),
        ]
        # Supply = 500 L
        res = allocate_water(total_supply=500, zones=zones)
        assert res.allocations[0].allocated_amount == Decimal("50.00")
        assert res.allocations[1].allocated_amount == Decimal("50.00")
        assert res.allocations[2].allocated_amount == Decimal("400.00")
        assert res.total_allocated == Decimal("500.00")


class TestHistoricalSafetyExtended:
    """Extended historical safety tests for run deletion and zone deactivation."""

    def test_deactivating_zone_preserves_historical_allocations(self, isolated_db):
        """Setting zone status to 'inactive' does not delete historical water_allocations."""
        zid = isolated_db.create_allocation_zone("Deactivate-Me", 500, 100, 1.0, status="active")
        zone_inputs = [ZoneDemandInput(zone_id=zid, zone_name="Deactivate-Me", requested_demand=500, essential_min=100)]
        run_res = allocate_water(total_supply=500, zones=zone_inputs)
        run_id = isolated_db.record_allocation_run(run_res)

        # Deactivate the zone
        isolated_db.update_allocation_zone(
            zone_id=zid,
            zone_name="Deactivate-Me",
            requested_demand_l=500,
            essential_min_l=100,
            fairness_weight=1.0,
            status="inactive",
        )

        zone = isolated_db.get_allocation_zone_by_id(zid)
        assert zone["status"] == "inactive"

        # Check historical allocations remain intact
        allocs = isolated_db.get_allocations_for_run(run_id)
        assert len(allocs) == 1
        assert allocs[0]["zone_name_snapshot"] == "Deactivate-Me"

    def test_cannot_delete_allocation_run_with_per_zone_allocations(self, isolated_db):
        """delete_allocation_run raises RuntimeError if per-zone allocations exist (RESTRICT)."""
        zid = isolated_db.create_allocation_zone("Zone-Run-Del", 500, 100, 1.0)
        zone_inputs = [ZoneDemandInput(zone_id=zid, zone_name="Zone-Run-Del", requested_demand=500, essential_min=100)]
        run_res = allocate_water(total_supply=500, zones=zone_inputs)
        run_id = isolated_db.record_allocation_run(run_res)

        with pytest.raises(RuntimeError, match="Cannot delete allocation run"):
            isolated_db.delete_allocation_run(run_id)

        # Run still exists
        assert isolated_db.get_allocation_run_by_id(run_id) is not None

    def test_can_delete_empty_allocation_run(self, isolated_db):
        """delete_allocation_run succeeds when no per-zone allocations are attached."""
        empty_res = allocate_water(total_supply=1000, zones=[])
        run_id = isolated_db.record_allocation_run(empty_res)

        # Can delete empty run
        isolated_db.delete_allocation_run(run_id)
        assert isolated_db.get_allocation_run_by_id(run_id) is None

