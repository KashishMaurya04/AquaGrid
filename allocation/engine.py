"""
allocation/engine.py – Mathematically Explainable Weighted Max-Min Fair Water Allocation.

Implements three-tier resource rationing and progressive filling (water-filling)
for distributing scarce municipal water supplies across heterogeneous consumer zones.

Fairness Doctrine & Policy Decomposition:
------------------------------------------
Policy 1: Abundant Supply (S >= sum(d_i))
    - If total available water supply is sufficient to satisfy all requested demands,
      every zone receives 100% of its requested demand (a_i = d_i). Unmet demand is 0.

Policy 2: Constrained Supply Above Essential Baseline (sum(e_i) <= S < sum(d_i))
    - Step 2A (Essential Guarantee):
      Every zone receives its critical essential minimum requirement first: a_i^(0) = e_i.
    - Step 2B (Weighted Max-Min Water-Filling on Residual Demands):
      The remaining surplus supply S_residual = S - sum(e_i) is distributed across zones
      with unmet residual demand (r_i = d_i - e_i > 0) using weighted max-min fairness.
      Under weighted max-min fairness, the normalized rate x_i / w_i is leveled across all
      active unsaturated zones until a zone's residual demand is fully satisfied (bottleneck
      saturation), at which point it is removed from the active set and excess supply is
      redistributed to remaining zones.

Policy 3: Severe Emergency Deficit (S < sum(e_i))
    - Available supply is insufficient even for critical survival requirements.
    - Supply is rationed strictly proportionally to essential requirements:
      a_i = S * (e_i / sum(e_j)).
    - Fallback: If all essential requirements are zero, supply is allocated proportionally
      to weighted requested demand: a_i = S * (w_i * d_i / sum(w_j * d_j)).

Invariants & Guarantees:
------------------------
1. Demand Ceiling: 0 <= a_i <= d_i for all zones (never over-allocates).
2. Supply Ceiling: sum(a_i) <= S (total allocation never exceeds available supply).
3. Monotonicity: Higher weight or higher essential need yields higher satisfaction.
4. Precision & Reconciliation: High-precision Decimal arithmetic with deterministic
   two-decimal-place rounding and penny/cent reconciliation.

Algorithmic Complexity:
-----------------------
- Time Complexity: O(N log N), where N is the number of allocation zones.
  Sorting active normalized residual demands takes O(N log N); at each water-filling
  iteration at least one zone saturates, bounding iterations to at most N.
- Space Complexity: O(N), for storage of residual vectors, active sets, and results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Any, Sequence


TWO_PLACES = Decimal("0.01")
EPSILON = Decimal("0.000000000001")
ALGORITHM_VERSION = "1.0-weighted-maxmin"


@dataclass
class ZoneDemandInput:
    """Input parameters for a consumer distribution zone."""
    zone_id: int
    zone_name: str
    requested_demand: Decimal | float | int
    essential_min: Decimal | float | int = 0.0
    fairness_weight: Decimal | float | int = 1.0
    node_id: int | None = None
    priority_category: str = "standard"


@dataclass
class ZoneAllocationResult:
    """Detailed allocation outcome for an individual consumer zone."""
    zone_id: int
    zone_name: str
    requested_demand: Decimal
    essential_min: Decimal
    fairness_weight: Decimal
    allocated_amount: Decimal
    unmet_demand: Decimal
    satisfaction_pct: Decimal
    explanation: str
    policy_applied: str
    is_bottleneck: bool = False
    tier_reached: int = 1

    @property
    def requested_demand_l(self) -> Decimal:
        return self.requested_demand

    @property
    def essential_min_l(self) -> Decimal:
        return self.essential_min

    @property
    def allocated_amount_l(self) -> Decimal:
        return self.allocated_amount

    @property
    def unmet_demand_l(self) -> Decimal:
        return self.unmet_demand

    @property
    def satisfaction_rate_pct(self) -> Decimal:
        return self.satisfaction_pct

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "requested_demand": float(self.requested_demand),
            "requested_demand_l": float(self.requested_demand),
            "essential_min": float(self.essential_min),
            "essential_min_l": float(self.essential_min),
            "fairness_weight": float(self.fairness_weight),
            "allocated_amount": float(self.allocated_amount),
            "allocated_amount_l": float(self.allocated_amount),
            "unmet_demand": float(self.unmet_demand),
            "unmet_demand_l": float(self.unmet_demand),
            "satisfaction_pct": float(self.satisfaction_pct),
            "satisfaction_rate_pct": float(self.satisfaction_pct),
            "is_bottleneck": self.is_bottleneck,
            "tier_reached": self.tier_reached,
            "explanation": self.explanation,
            "policy_applied": self.policy_applied,
        }


@dataclass
class AllocationRunResult:
    """Summary of a multi-zone fair water allocation run."""
    total_supply: Decimal
    total_demand: Decimal
    total_allocated: Decimal
    total_unmet: Decimal
    zone_count: int
    policy_applied: str
    satisfaction_rate_pct: Decimal = Decimal("100.00")
    algorithm_version: str = ALGORITHM_VERSION
    allocations: list[ZoneAllocationResult] = field(default_factory=list)

    @property
    def total_supply_l(self) -> Decimal:
        return self.total_supply

    @property
    def total_demand_l(self) -> Decimal:
        return self.total_demand

    @property
    def total_allocated_l(self) -> Decimal:
        return self.total_allocated

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_supply": float(self.total_supply),
            "total_supply_l": float(self.total_supply),
            "total_demand": float(self.total_demand),
            "total_demand_l": float(self.total_demand),
            "total_allocated": float(self.total_allocated),
            "total_allocated_l": float(self.total_allocated),
            "total_unmet": float(self.total_unmet),
            "zone_count": self.zone_count,
            "satisfaction_rate_pct": float(self.satisfaction_rate_pct),
            "policy_applied": self.policy_applied,
            "algorithm_version": self.algorithm_version,
            "allocations": [a.to_dict() for a in self.allocations],
        }


def allocate_water(
    total_supply: Decimal | float | int | str,
    zones: Sequence[ZoneDemandInput | dict[str, Any]],
) -> AllocationRunResult:
    """
    Execute three-tier weighted max-min fair water allocation.

    Args:
        total_supply: Total water supply available (litres), must be >= 0.
        zones: List of ZoneDemandInput objects or dictionaries describing consumer zones.

    Returns:
        AllocationRunResult containing overall summary and per-zone allocations.

    Raises:
        ValueError: If supply is negative, or if zone inputs fail validation constraints
                    (e.g., negative demand, essential_min > demand, non-positive weight).
    """
    # ── 1. Validate and Parse Total Supply ────────────────────────────────────
    try:
        S = Decimal(str(total_supply)).quantize(TWO_PLACES, rounding=ROUND_DOWN)
    except Exception as exc:
        raise ValueError(f"Invalid total supply value '{total_supply}': {exc}") from exc

    if S < Decimal("0.00"):
        raise ValueError(f"Total available water supply cannot be negative (got {S} L).")

    # Empty zone list edge case
    if not zones:
        return AllocationRunResult(
            total_supply=S,
            total_demand=Decimal("0.00"),
            total_allocated=Decimal("0.00"),
            total_unmet=Decimal("0.00"),
            zone_count=0,
            policy_applied="NO_ZONES",
            allocations=[],
        )

    # ── 2. Validate and Parse Zone Inputs ─────────────────────────────────────
    parsed_zones: list[ZoneDemandInput] = []
    for idx, raw_z in enumerate(zones):
        if isinstance(raw_z, dict):
            zid = int(raw_z.get("zone_id", idx + 1))
            name = str(raw_z.get("zone_name", raw_z.get("name", f"Zone-{zid}")))
            req_d = raw_z.get("requested_demand", raw_z.get("requested_demand_l", 0.0))
            ess_m = raw_z.get("essential_min", raw_z.get("essential_min_l", 0.0))
            f_w = raw_z.get("fairness_weight", 1.0)
            node_id = raw_z.get("node_id")
            cat = str(raw_z.get("priority_category", "standard"))
        else:
            zid = raw_z.zone_id
            name = raw_z.zone_name
            req_d = raw_z.requested_demand
            ess_m = raw_z.essential_min
            f_w = raw_z.fairness_weight
            node_id = raw_z.node_id
            cat = raw_z.priority_category

        try:
            d_val = Decimal(str(req_d)).quantize(TWO_PLACES, rounding=ROUND_DOWN)
            e_val = Decimal(str(ess_m)).quantize(TWO_PLACES, rounding=ROUND_DOWN)
            w_val = Decimal(str(f_w))
        except Exception as exc:
            raise ValueError(f"Invalid numeric input in zone '{name}': {exc}") from exc

        if d_val < Decimal("0.00"):
            raise ValueError(f"Requested demand for zone '{name}' cannot be negative (got {d_val} L).")
        if e_val < Decimal("0.00"):
            raise ValueError(f"Essential minimum for zone '{name}' cannot be negative (got {e_val} L).")
        if e_val > d_val:
            raise ValueError(
                f"Essential minimum ({e_val} L) cannot exceed requested demand ({d_val} L) for zone '{name}'."
            )
        if w_val <= Decimal("0.00"):
            raise ValueError(f"Fairness weight for zone '{name}' must be strictly positive (got {w_val}).")

        parsed_zones.append(ZoneDemandInput(
            zone_id=zid,
            zone_name=name,
            requested_demand=d_val,
            essential_min=e_val,
            fairness_weight=w_val,
            node_id=node_id,
            priority_category=cat,
        ))

    total_demand = sum(z.requested_demand for z in parsed_zones)
    total_essential = sum(z.essential_min for z in parsed_zones)

    raw_allocations: dict[int, Decimal] = {}
    explanations: dict[int, str] = {}
    policy_applied: str

    # ── 3. Policy Execution ───────────────────────────────────────────────────

    # Case 0: Zero Available Supply
    if S == Decimal("0.00"):
        policy_applied = "ZERO_SUPPLY"
        for z in parsed_zones:
            raw_allocations[z.zone_id] = Decimal("0.00")
            explanations[z.zone_id] = "Zero water supply available. All requested demands unmet."

    # Policy 1: Abundant Supply (S >= total_demand)
    elif S >= total_demand:
        policy_applied = "ABUNDANT_SUPPLY"
        for z in parsed_zones:
            raw_allocations[z.zone_id] = z.requested_demand
            explanations[z.zone_id] = (
                f"Sufficient supply available. Full requested demand ({z.requested_demand:.2f} L) satisfied (100.0%)."
            )

    # Policy 4 Fallback: Zero essential requirements during shortage (S < total_demand and total_essential == 0)
    elif total_essential == Decimal("0.00"):
        policy_applied = "WEIGHTED_DEMAND_FALLBACK"
        weighted_sum = sum(z.fairness_weight * z.requested_demand for z in parsed_zones)
        for z in parsed_zones:
            if weighted_sum > Decimal("0.00"):
                share = (z.fairness_weight * z.requested_demand) / weighted_sum
                raw_allocations[z.zone_id] = S * share
            else:
                raw_allocations[z.zone_id] = S / Decimal(len(parsed_zones))
            explanations[z.zone_id] = (
                f"Supply shortage with zero essential requirements: allocated proportionally based on "
                f"weighted requested demand (weight {z.fairness_weight})."
            )

    # Policy 2: Constrained Supply Above Essential Baseline (total_essential <= S < total_demand)
    elif S >= total_essential:
        policy_applied = "WEIGHTED_MAX_MIN_RESIDUAL"

        # 2A: Guaranteed Essential Minimums
        residual_allocations: dict[int, Decimal] = {z.zone_id: Decimal("0.0") for z in parsed_zones}
        S_residual = S - total_essential

        # 2B: Weighted Max-Min Water-Filling on Residuals via One-Time Sort and Linear Sweep (O(N log N))
        active = [
            (
                (z.requested_demand - z.essential_min) / z.fairness_weight,  # gamma_i = r_i / w_i
                z.fairness_weight,
                z.requested_demand - z.essential_min,
                z.zone_id,
            )
            for z in parsed_zones
            if z.requested_demand > z.essential_min
        ]
        # One-time sort: lowest normalized capacity ratio first, then higher weight, then lower zone_id
        active.sort(key=lambda item: (item[0], -item[1], item[3]))

        total_active_weight = sum(item[1] for item in active)
        lambda_curr = Decimal("0.0")
        lambda_final = Decimal("0.0")

        i = 0
        m = len(active)
        while i < m and S_residual > EPSILON:
            gamma_k = active[i][0]
            delta_lambda = gamma_k - lambda_curr

            # Group all items with identical normalized capacity threshold
            j = i
            group_weight = Decimal("0.0")
            while j < m and active[j][0] == gamma_k:
                group_weight += active[j][1]
                j += 1

            delta_S = delta_lambda * total_active_weight
            if delta_S <= S_residual:
                S_residual -= delta_S
                lambda_curr = gamma_k
                total_active_weight -= group_weight
                i = j
                lambda_final = lambda_curr
            else:
                delta_lambda_final = S_residual / total_active_weight
                lambda_final = lambda_curr + delta_lambda_final
                S_residual = Decimal("0.0")
                break

        for z in parsed_zones:
            r_i = z.requested_demand - z.essential_min
            if r_i <= Decimal("0.0"):
                residual_allocations[z.zone_id] = Decimal("0.0")
            else:
                residual_allocations[z.zone_id] = min(r_i, lambda_final * z.fairness_weight)

        for z in parsed_zones:
            raw_allocations[z.zone_id] = z.essential_min + residual_allocations[z.zone_id]
            x_i = residual_allocations[z.zone_id]
            if raw_allocations[z.zone_id] >= z.requested_demand:
                explanations[z.zone_id] = (
                    f"Essential requirement guaranteed ({z.essential_min:.2f} L) and residual demand fully satisfied "
                    f"(+{x_i:.2f} L) under weighted max-min fairness (weight {z.fairness_weight})."
                )
            else:
                explanations[z.zone_id] = (
                    f"Essential requirement guaranteed ({z.essential_min:.2f} L) with partial residual allocation "
                    f"(+{x_i:.2f} L) under weighted max-min fairness (weight {z.fairness_weight})."
                )

    # Policy 3: Severe Emergency Deficit (S < total_essential)
    else:
        policy_applied = "EMERGENCY_ESSENTIAL_PROPORTIONAL"
        for z in parsed_zones:
            prop = z.essential_min / total_essential
            raw_allocations[z.zone_id] = S * prop
            explanations[z.zone_id] = (
                f"Severe emergency deficit: available supply ({S:.2f} L) is below total essential minimum "
                f"({total_essential:.2f} L). Supply rationed strictly proportional to essential survival "
                f"requirement ({z.essential_min:.2f} L / {total_essential:.2f} L)."
            )

    # ── 4. Decimal Rounding & Deterministic Reconciliation ─────────────────────
    rounded_allocations: dict[int, Decimal] = {}
    remainders: list[tuple[Decimal, Decimal, int]] = []  # (fractional_rem, weight, zone_id)

    for z in parsed_zones:
        raw_val = raw_allocations[z.zone_id]
        # Round down to prevent preliminary sum from ever exceeding S
        rounded = raw_val.quantize(TWO_PLACES, rounding=ROUND_DOWN)
        # Cap strictly at requested demand
        rounded = min(rounded, z.requested_demand)
        rounded_allocations[z.zone_id] = rounded

        frac = raw_val - rounded
        remainders.append((frac, z.fairness_weight, z.zone_id))

    # Calculate residual rounding gap against target_allocated = min(S, total_demand)
    target_allocated = min(S, total_demand)
    current_total = sum(rounded_allocations.values())
    R = target_allocated - current_total

    # Reconcile fractional cents/millilitres deterministically
    # Priority: largest fractional remainder first, then higher weight, then lower zone_id
    remainders.sort(key=lambda item: (-item[0], -item[1], item[2]))

    zone_dict = {z.zone_id: z for z in parsed_zones}
    idx = 0
    # Loop over eligible zones that can accept +0.01 without exceeding demand
    while R >= TWO_PLACES and idx < len(remainders):
        _, _, zid = remainders[idx]
        z_obj = zone_dict[zid]
        if rounded_allocations[zid] + TWO_PLACES <= z_obj.requested_demand:
            rounded_allocations[zid] += TWO_PLACES
            R -= TWO_PLACES
        idx += 1

    # ── 5. Assemble Structured Results ─────────────────────────────────────────
    results: list[ZoneAllocationResult] = []
    for z in parsed_zones:
        alloc_amt = rounded_allocations[z.zone_id]
        unmet = z.requested_demand - alloc_amt
        pct = (alloc_amt / z.requested_demand * 100) if z.requested_demand > Decimal("0.00") else Decimal("100.00")

        # Determine bottleneck status and tier reached
        is_bottleneck = False
        tier = 0
        if policy_applied == "ABUNDANT_SUPPLY":
            is_bottleneck = True
            tier = 3
        elif policy_applied == "WEIGHTED_MAX_MIN_RESIDUAL":
            if alloc_amt >= z.requested_demand:
                is_bottleneck = True
                tier = 3
            elif alloc_amt > z.essential_min:
                is_bottleneck = False
                tier = 2
            else:
                is_bottleneck = False
                tier = 1
        elif policy_applied == "EMERGENCY_ESSENTIAL_PROPORTIONAL":
            is_bottleneck = False
            tier = 0
        else:
            is_bottleneck = False
            tier = 0

        results.append(ZoneAllocationResult(
            zone_id=z.zone_id,
            zone_name=z.zone_name,
            requested_demand=z.requested_demand,
            essential_min=z.essential_min,
            fairness_weight=z.fairness_weight,
            allocated_amount=alloc_amt,
            unmet_demand=unmet,
            satisfaction_pct=pct.quantize(TWO_PLACES, rounding=ROUND_HALF_UP),
            explanation=explanations[z.zone_id],
            policy_applied=policy_applied,
            is_bottleneck=is_bottleneck,
            tier_reached=tier,
        ))

    final_allocated = sum(r.allocated_amount for r in results)
    final_unmet = sum(r.unmet_demand for r in results)
    overall_sat = (
        (final_allocated / total_demand * 100).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
        if total_demand > Decimal("0.00")
        else Decimal("100.00")
    )

    return AllocationRunResult(
        total_supply=S,
        total_demand=total_demand,
        total_allocated=final_allocated,
        total_unmet=final_unmet,
        zone_count=len(results),
        policy_applied=policy_applied,
        satisfaction_rate_pct=overall_sat,
        algorithm_version=ALGORITHM_VERSION,
        allocations=results,
    )
