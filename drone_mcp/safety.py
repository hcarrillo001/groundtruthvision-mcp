"""The safety gate. Every command passes through here before it reaches a driver.

The rule: Claude decides where to go, the gate decides whether that is allowed,
and the gate always wins. A refused command raises Refused with a reason Claude
can read and act on; nothing is sent to the drone.

Limits come from environment variables so they can be tightened without a code
change. The defaults suit a small drone indoors.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .drone import DIRECTIONS, DroneState, offset


class Refused(Exception):
    """A command the gate would not send. The message says why."""


def _env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Limits:
    max_height: float = _env("DRONE_MAX_HEIGHT_CM", 200)
    min_height: float = _env("DRONE_MIN_HEIGHT_CM", 30)
    min_move: float = 20          # the Tello ignores moves under 20 cm
    max_move: float = _env("DRONE_MAX_MOVE_CM", 200)
    wall_margin: float = _env("DRONE_WALL_MARGIN_CM", 40)
    battery_floor: float = _env("DRONE_BATTERY_FLOOR", 20)
    takeoff_battery: float = _env("DRONE_TAKEOFF_BATTERY", 30)


@dataclass
class FlightBox:
    """The area the drone may fly in, in cm. In the simulator it is the room;
    on a real drone it is a box drawn around the take-off point."""
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def around(cls, x: float, y: float, half_width: float, half_depth: float):
        return cls(x - half_width, y - half_depth, x + half_width, y + half_depth)


class SafetyGate:
    def __init__(self, box: FlightBox, limits: Limits | None = None):
        self.box = box
        self.limits = limits or Limits()

    # -- checks -----------------------------------------------------------------
    def check_takeoff(self, s: DroneState):
        if s.flying:
            raise Refused("Already flying.")
        if s.battery < self.limits.takeoff_battery:
            raise Refused(f"Battery {s.battery:.0f}% is below the "
                          f"{self.limits.takeoff_battery:.0f}% needed to take off.")

    def check_move(self, s: DroneState, direction: str, cm: float):
        self._airborne(s)
        self._battery(s)
        if direction not in DIRECTIONS:
            raise Refused(f"Unknown direction '{direction}'. Use one of: {', '.join(DIRECTIONS)}.")
        if not self.limits.min_move <= cm <= self.limits.max_move:
            raise Refused(f"Moves must be {self.limits.min_move:.0f}-{self.limits.max_move:.0f} cm "
                          f"(asked for {cm:.0f}). Split long moves into steps.")
        dx, dy, dz = offset(s.heading, direction, cm)
        x, y, z = s.x + dx, s.y + dy, s.z + dz
        m = self.limits.wall_margin
        if not (self.box.x0 + m <= x <= self.box.x1 - m and self.box.y0 + m <= y <= self.box.y1 - m):
            raise Refused(f"That move would leave the flight box (or come within "
                          f"{m:.0f} cm of a wall). Turn or move a shorter distance.")
        if z > self.limits.max_height:
            raise Refused(f"That would put the drone at {z:.0f} cm, above the "
                          f"{self.limits.max_height:.0f} cm ceiling.")
        if z < self.limits.min_height:
            raise Refused(f"That would put the drone at {z:.0f} cm, below the "
                          f"{self.limits.min_height:.0f} cm minimum. Use land instead.")

    def check_rotate(self, s: DroneState, degrees: float):
        self._airborne(s)
        self._battery(s)
        if not -360 <= degrees <= 360 or degrees == 0:
            raise Refused("Rotate by -360 to 360 degrees (not 0).")

    def check_hover(self, s: DroneState):
        self._airborne(s)

    # landing and emergency stop are never refused while flying
    def check_land(self, s: DroneState):
        if not s.flying:
            raise Refused("Not flying.")

    def battery_low(self, s: DroneState) -> bool:
        return s.flying and s.battery < self.limits.battery_floor

    # -- helpers ----------------------------------------------------------------
    @staticmethod
    def _airborne(s: DroneState):
        if not s.flying:
            raise Refused("Not flying. Call takeoff first.")

    def _battery(self, s: DroneState):
        if s.battery < self.limits.battery_floor:
            raise Refused(f"Battery {s.battery:.0f}% is below the "
                          f"{self.limits.battery_floor:.0f}% floor. Land now.")
