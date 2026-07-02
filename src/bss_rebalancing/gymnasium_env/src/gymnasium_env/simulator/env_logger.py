from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

from gymnasium_env.simulator.station import Station
from gymnasium_env.simulator.trip import Trip
from gymnasium_env.simulator.truck import Truck


class EnvLogger:
    """
    Lightweight, domain-specific logger for the environment.

    - Call `init(...)` once to configure handlers (only if logging is enabled).
    - If not initialized, all public methods are safe no-ops.
    - `enabled` can be toggled at runtime (e.g. per-episode) without
      touching the underlying logging configuration.
    - Call `set_env_time(t)` at each step to prefix all log messages with
      the internal environment timestamp.
    """

    def __init__(self, name: str = "env"):
        self._name: str = name
        self._logger: Optional[logging.Logger] = None
        self._initialized: bool = False
        self._enabled: bool = False
        self._env_time: str | None = None  # internal env timestamp prefix

    # ------------------------------------------------------------------
    # Initialization & configuration
    # ------------------------------------------------------------------

    def init(
        self,
        log_dir: str | None,
        filename: str = "env.log",
        level: int = logging.INFO,
        enabled: bool = True,
        overwrite: bool = False,
    ) -> None:
        if not enabled or log_dir is None:
            self._initialized = False
            self._enabled = False
            self._logger = None
            return

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_path = log_path / filename

        logger = logging.getLogger(f"gym_env.{self._name}")
        logger.setLevel(level)

        if overwrite:
            logger.handlers.clear()

        if not logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
            )
            file_handler = logging.FileHandler(file_path, mode="w")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self._logger = logger
        self._initialized = True
        self._enabled = True

    def set_enabled(self, enabled: bool) -> None:
        """Toggle runtime logging without changing configuration."""
        self._enabled = enabled

    def reconfigure(
        self,
        log_dir: str,
        filename: str = "env.log",
        level: int | None = None,
    ) -> None:
        """Change the log directory/file at runtime."""
        if not self._initialized and not self._enabled:
            return
        self.init(
            log_dir=log_dir,
            filename=filename,
            level=level
            if level is not None
            else (self._logger.level if self._logger else logging.INFO),
            enabled=self._enabled,
            overwrite=True,
        )

    # ------------------------------------------------------------------
    # Env time stamping
    # ------------------------------------------------------------------

    def set_env_time(self, env_time: str | None) -> None:
        """
        Set the internal environment timestamp to prepend to all log messages.
        Call this once per step, e.g.:
            self._env_logger.set_env_time(convert_seconds_to_hours_minutes_day(self._env_time))
        Pass None to clear the prefix.
        """
        self._env_time = env_time

    def _prefix(self, message: str) -> str:
        """Prepend env timestamp to message if set."""
        if self._env_time is not None:
            return f"[{self._env_time}] {message}"
        return message

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def can_log(self, level: int = logging.INFO) -> bool:
        if not self._initialized or not self._enabled or self._logger is None:
            return False
        return self._logger.isEnabledFor(level)

    # ------------------------------------------------------------------
    # Public logging API
    # ------------------------------------------------------------------

    def new_log_line(self, timeslot: int | None = None) -> None:
        if not self.can_log(logging.INFO):
            return
        msg = "--------------------------------------------------------"
        if timeslot is not None:
            msg += f" Timeslot = {timeslot}"
        self._logger.info(self._prefix(msg))

    def log_starting_action(
        self, action: str, t: int, cell_id: int, invalid: bool
    ) -> None:
        if not self.can_log(logging.INFO):
            return
        invalid_str = "INVALID" if invalid else ""
        steps = int(math.ceil(t / 30))
        self._logger.info(
            self._prefix(
                f"ACTION: {action} on cell {cell_id} {invalid_str}"
                f" --> Time to complete: {t}s - Steps needed: {steps}"
            )
        )

    def log_ending_action(self, invalid: bool, time: str) -> None:
        if not self.can_log(logging.INFO):
            return
        if invalid:
            self._logger.info(self._prefix(f"### ACTION IS INVALID ### - Time: {time}"))
        else:
            self._logger.info(
                self._prefix(f"Action completed successfully - Time: {time}")
            )

    def log_state(self, step: int, time: str) -> None:
        if not self.can_log(logging.INFO):
            return
        self._logger.info(self._prefix(f"State S_{step} - Time: {time}"))

    def log_truck(self, truck: Truck, depot_bikes: int) -> None:
        if not self.can_log(logging.INFO):
            return
        self._logger.info(
            self._prefix(
                f"TRUCK in CELL: {truck.get_cell().get_id()}"
                f" (center_node={truck.get_cell().get_center_node()},"
                f" cell_bikes={truck.get_cell().get_total_bikes()},"
                f" critic_score={truck.get_cell().get_critic_score()})"
                f" - POSITION: {truck.get_position()}"
                f" - LOAD: {truck.get_load()}-{depot_bikes}"
            )
        )

    def log_no_available_bikes(
        self, start_station: Station, end_station: Station
    ) -> None:
        if not self.can_log(logging.WARNING):
            return
        # End station cell
        end_cell = (
            end_station.get_cell().get_id()
            if end_station.get_station_id() != 10000 and end_station.get_cell() is not None
            else "-"
        )

        # Start station cell (same sentinel logic)
        start_cell = (
            start_station.get_cell().get_id()
            if start_station.get_station_id() != 10000 and start_station.get_cell() is not None
            else "-"
        )
        self._logger.warning(
            self._prefix(
                f"TRIP FAILED: No bikes from station {start_station.get_station_id()}"
                f" (cell: {start_cell})"
                f" to station {end_station.get_station_id()}"
                f" (cell: {end_cell})"
            )
        )

    def log_trip(self, trip: Trip) -> None:
        if not self.can_log(logging.INFO):
            return
        self._logger.info(self._prefix(f"Trip: {trip}"))

    def log_terminated(self, time: str) -> None:
        if not self.can_log(logging.INFO):
            return
        self._logger.info(
            self._prefix(
                f"##################################################"
                f"##### Slot TERMINATED - Time: {time} "
                f"##################################################"
            )
        )

    def log_done(self, time: str) -> None:
        if not self.can_log(logging.INFO):
            return
        self._logger.info(
            self._prefix(
                f"##################################################"
                f"##### THE EPISODE IS DONE - Time: {time} "
                f"##################################################"
            )
        )

    def info(self, message: str) -> None:
        if not self.can_log(logging.INFO):
            return
        self._logger.info(self._prefix(message))

    def warning(self, message: str) -> None:
        if not self.can_log(logging.WARNING):
            return
        self._logger.warning(self._prefix(message))

    def debug(self, message: str) -> None:
        if not self.can_log(logging.DEBUG):
            return
        self._logger.debug(self._prefix(message))
