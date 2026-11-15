"""
simulation/engine.py – Sensor Simulation Engine for AquaGrid.

Generates realistic synthetic telemetry readings for active water distribution sensors
(flow meters and pressure sensors) supporting:
  1. Normal Mode:
     - Synthetic flow readings centered around typical pipe delivery baselines with
       natural Gaussian fluctuation.
     - Synthetic pressure readings reflecting normal municipal water distribution
       mains pressure with small random noise.
     - Automatically converts values to sensor-specific measurement units (L/min, m³/h,
       L/s, psi, bar, kPa).
     - Inactive sensors are strictly excluded.
  2. Leak Scenario Mode:
     - The user designates a monitored pipe experiencing a simulated rupture/leak.
     - Flow meters on the affected pipe exhibit a severe flow surge (orifice outflow).
     - Pressure sensors on the affected pipe exhibit a marked pressure drop.
     - Sensors on unaffected pipes continue generating baseline readings.
     - Clearly flagged as synthetic demonstration data; does NOT claim to prove a physical leak.
  3. Reproducibility & Determinism:
     - Configurable random seed (`seed=...`) for repeatable test scenarios.
     - Modular, parameterizable `SimulatorConfig`.

Physical Assumptions & Generated Ranges:
-----------------------------------------
- Flow Meter Baselines:
  * Baseline flow: ~350.0 L/min (~21.0 m³/h or ~5.83 L/s).
  * Normal fluctuation: Gaussian std dev = 7.0 L/min (approx ±2-4%).
  * Normal Range (L/min): ~330.0 to 370.0 L/min.
  * Leak Surge Anomaly: Flow increases by ~70% (multiplier = 1.70, std = 14.0 L/min).
  * Leak Range (L/min): ~550.0 to 650.0 L/min (conspicuously higher than normal).
- Pressure Sensor Baselines:
  * Baseline pressure: ~52.0 psi (~3.59 bar or ~358.5 kPa).
  * Normal fluctuation: Gaussian std dev = 1.1 psi (approx ±2%).
  * Normal Range (psi): ~49.0 to 55.0 psi.
  * Leak Drop Anomaly: Pressure drops by ~26.0 psi down to ~26.0 psi (std = 1.8 psi).
  * Leak Range (psi): ~21.0 to 31.0 psi (conspicuously lower than normal 52 psi).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Sequence


@dataclass
class SimulatorConfig:
    """Configurable baseline parameters for flow and pressure simulation."""
    # Flow parameters (in L/min)
    flow_baseline_lpm: float = 350.0
    flow_noise_std_lpm: float = 7.0
    flow_leak_multiplier: float = 1.70       # ~70% surge in flow
    flow_leak_noise_std_lpm: float = 14.0

    # Pressure parameters (in psi)
    pressure_baseline_psi: float = 52.0
    pressure_noise_std_psi: float = 1.1
    pressure_leak_drop_psi: float = 26.0      # drops to ~26.0 psi
    pressure_leak_noise_std_psi: float = 1.8


DEFAULT_CONFIG = SimulatorConfig()


@dataclass
class GeneratedReading:
    """Represents a generated synthetic sensor reading."""
    sensor_id: int
    sensor_name: str
    sensor_type: str
    pipe_id: int
    pipe_name: str
    reading_value: float
    unit: str
    recorded_at: str
    is_simulated: int = 1
    is_anomaly: bool = False
    simulation_mode: str = "normal"  # "normal" or "leak"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "sensor_name": self.sensor_name,
            "sensor_type": self.sensor_type,
            "pipe_id": self.pipe_id,
            "pipe_name": self.pipe_name,
            "reading_value": self.reading_value,
            "unit": self.unit,
            "recorded_at": self.recorded_at,
            "is_simulated": self.is_simulated,
            "is_anomaly": self.is_anomaly,
            "simulation_mode": self.simulation_mode,
        }


class SensorSimulator:
    """
    Synthetic telemetry generator for water distribution sensors.
    """

    def __init__(
        self,
        config: SimulatorConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.rng = random.Random(seed)

    def set_seed(self, seed: int | None) -> None:
        """Reset the internal PRNG with a specific seed for reproducible output."""
        self.rng = random.Random(seed)

    # ── Unit conversion helpers ───────────────────────────────────────────────

    @staticmethod
    def _convert_flow_from_lpm(value_lpm: float, target_unit: str) -> float:
        """Convert flow from L/min to target unit."""
        u = target_unit.strip().lower()
        if u in ("l/min", "lpm"):
            return value_lpm
        if u in ("m³/h", "m3/h", "m3h"):
            # 1 m³/h = 1000 L / 60 min = 16.666667 L/min
            return value_lpm * 0.06
        if u in ("l/s", "lps"):
            # 1 L/s = 60 L/min
            return value_lpm / 60.0
        return value_lpm

    @staticmethod
    def _convert_pressure_from_psi(value_psi: float, target_unit: str) -> float:
        """Convert pressure from psi to target unit."""
        u = target_unit.strip().lower()
        if u in ("psi", "lb/in²"):
            return value_psi
        if u == "bar":
            # 1 psi = 0.0689476 bar
            return value_psi * 0.0689476
        if u in ("kpa", "kilopascal"):
            # 1 psi = 6.89476 kPa
            return value_psi * 6.89476
        return value_psi

    # ── Single reading generator ──────────────────────────────────────────────

    def generate_reading_for_sensor(
        self,
        sensor: dict[str, Any] | Any,
        is_leak_affected: bool = False,
        recorded_at: str | None = None,
    ) -> GeneratedReading:
        """
        Generate a synthetic reading for an individual active sensor.

        Args:
            sensor: Dict or sqlite3.Row containing sensor_id, sensor_name, sensor_type,
                    pipe_id, unit, status, and pipe_name.
            is_leak_affected: Whether this sensor is on a pipe experiencing a simulated leak.
            recorded_at: ISO timestamp for the reading. Defaults to current UTC time.

        Returns:
            GeneratedReading object.

        Raises:
            ValueError: If the sensor is inactive or has an unsupported sensor_type.
        """
        status = sensor["status"] if isinstance(sensor, dict) else sensor["status"]
        if status != "active":
            raise ValueError(
                f"Cannot generate reading for inactive sensor #{sensor['sensor_id']} ('{sensor['sensor_name']}')."
            )

        sensor_id = int(sensor["sensor_id"])
        sensor_name = str(sensor["sensor_name"])
        sensor_type = str(sensor["sensor_type"])
        pipe_id = int(sensor["pipe_id"])
        pipe_name = str(sensor.get("pipe_name", f"Pipe #{pipe_id}") if isinstance(sensor, dict) else sensor["pipe_name"])
        unit = str(sensor["unit"])

        ts = recorded_at or datetime.now(timezone.utc).isoformat()

        if sensor_type == "flow_meter":
            if is_leak_affected:
                # Flow surge during pipe rupture / leak
                mean_lpm = self.config.flow_baseline_lpm * self.config.flow_leak_multiplier
                raw_lpm = self.rng.gauss(mean_lpm, self.config.flow_leak_noise_std_lpm)
                is_anomaly = True
            else:
                # Normal operational flow
                mean_lpm = self.config.flow_baseline_lpm
                raw_lpm = self.rng.gauss(mean_lpm, self.config.flow_noise_std_lpm)
                is_anomaly = False

            # Flow must be strictly positive
            raw_lpm = max(1.0, raw_lpm)
            converted = self._convert_flow_from_lpm(raw_lpm, unit)
            final_value = round(converted, 2)

        elif sensor_type == "pressure_sensor":
            if is_leak_affected:
                # Pressure drop at rupture site
                mean_psi = self.config.pressure_baseline_psi - self.config.pressure_leak_drop_psi
                raw_psi = self.rng.gauss(mean_psi, self.config.pressure_leak_noise_std_psi)
                is_anomaly = True
            else:
                # Normal distribution pressure
                mean_psi = self.config.pressure_baseline_psi
                raw_psi = self.rng.gauss(mean_psi, self.config.pressure_noise_std_psi)
                is_anomaly = False

            # Pressure must be strictly positive
            raw_psi = max(0.1, raw_psi)
            converted = self._convert_pressure_from_psi(raw_psi, unit)
            final_value = round(converted, 2)

        else:
            raise ValueError(f"Unsupported sensor type '{sensor_type}'.")

        return GeneratedReading(
            sensor_id=sensor_id,
            sensor_name=sensor_name,
            sensor_type=sensor_type,
            pipe_id=pipe_id,
            pipe_name=pipe_name,
            reading_value=final_value,
            unit=unit,
            recorded_at=ts,
            is_simulated=1,
            is_anomaly=is_anomaly,
            simulation_mode="leak" if is_leak_affected else "normal",
        )

    # ── Batch generator ───────────────────────────────────────────────────────

    def generate_batch(
        self,
        active_sensors: Sequence[dict[str, Any] | Any],
        mode: str = "normal",
        leak_pipe_id: int | None = None,
        iterations: int = 1,
        base_time: datetime | None = None,
    ) -> list[GeneratedReading]:
        """
        Generate readings for all active sensors over `iterations` steps.

        Args:
            active_sensors: List of active sensors (dicts or sqlite3.Row).
            mode: "normal" or "leak".
            leak_pipe_id: Required if mode is "leak"; sensors on this pipe exhibit anomaly.
            iterations: Number of sequential reading cycles (e.g. 1 or 10).
            base_time: Starting timestamp. Defaults to current UTC time.

        Returns:
            List of GeneratedReading instances ordered by step and sensor.

        Raises:
            ValueError: If mode == "leak" and leak_pipe_id is None.
        """
        if mode == "leak" and leak_pipe_id is None:
            raise ValueError("leak_pipe_id is required when mode is 'leak'.")

        # Exclude any inactive sensors if passed by mistake
        valid_sensors = [s for s in active_sensors if s["status"] == "active"]
        if not valid_sensors:
            return []

        start_dt = base_time or datetime.now(timezone.utc)
        results: list[GeneratedReading] = []

        for step in range(iterations):
            # Advance each step by 30 seconds for a realistic time sequence
            step_time = (start_dt + timedelta(seconds=step * 30)).isoformat()

            for s in valid_sensors:
                s_pipe = int(s["pipe_id"])
                is_leak = (mode == "leak" and s_pipe == leak_pipe_id)
                reading = self.generate_reading_for_sensor(
                    sensor=s,
                    is_leak_affected=is_leak,
                    recorded_at=step_time,
                )
                results.append(reading)

        return results

