"""Drone drivers.

A driver only knows how to fly and how to take a picture. It never decides
whether a command is safe: that is the SafetyGate's job, which runs first.

Coordinates (simulation): centimetres, x to the east, y to the north, z up.
Heading is a compass heading in degrees: 0 faces north (+y), 90 faces east.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Protocol


DIRECTIONS = ("forward", "back", "left", "right", "up", "down")


@dataclass
class DroneState:
    flying: bool = False
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    heading: float = 0.0
    battery: float = 100.0
    clock: float = 0.0          # seconds since the drone was created


class Drone(Protocol):
    """What every driver provides. SimDrone and TelloDrone both implement it."""
    state: DroneState

    def takeoff(self) -> None: ...
    def land(self) -> None: ...
    def move(self, direction: str, cm: int) -> None: ...
    def rotate(self, degrees: int) -> None: ...
    def hover(self, seconds: float) -> None: ...
    def emergency(self) -> None: ...
    def capture(self, tag: str) -> str: ...      # path to a saved frame


def offset(heading: float, direction: str, cm: float):
    """(dx, dy, dz) for a move relative to where the drone is facing."""
    h = math.radians(heading)
    fwd = (math.sin(h), math.cos(h))
    right = (math.cos(h), -math.sin(h))
    return {
        "forward": (fwd[0] * cm, fwd[1] * cm, 0.0),
        "back": (-fwd[0] * cm, -fwd[1] * cm, 0.0),
        "right": (right[0] * cm, right[1] * cm, 0.0),
        "left": (-right[0] * cm, -right[1] * cm, 0.0),
        "up": (0.0, 0.0, cm),
        "down": (0.0, 0.0, -cm),
    }[direction]


# ---------------------------------------------------------------------------
# Simulated world
# ---------------------------------------------------------------------------

@dataclass
class WorldObject:
    label: str                 # what a detector would call it, e.g. "backpack"
    description: str           # what it really is, e.g. "blue backpack"
    x: float
    y: float
    photo: str                 # picture the simulated camera shows when it sees it


@dataclass
class World:
    name: str
    width: float               # cm, x runs 0..width
    depth: float               # cm, y runs 0..depth
    start: tuple               # (x, y, heading)
    objects: List[WorldObject] = field(default_factory=list)
    empty_photo: str = ""      # what the camera shows when nothing is in view

    @classmethod
    def load(cls, path: str) -> "World":
        import json
        from pathlib import Path
        p = Path(path)
        d = json.loads(p.read_text())
        base = p.parent

        def rel(f):
            return str((base / f).resolve()) if f else ""
        objs = [WorldObject(o["label"], o["description"], o["x"], o["y"], rel(o["photo"]))
                for o in d.get("objects", [])]
        return cls(d["name"], d["width"], d["depth"], tuple(d["start"]), objs,
                   rel(d.get("empty_photo", "")))


def bearing_to(state: DroneState, x: float, y: float):
    """(relative angle in degrees, distance in cm) from the drone to a point.
    The angle is 0 straight ahead, negative to the left, positive to the right."""
    dx, dy = x - state.x, y - state.y
    absolute = math.degrees(math.atan2(dx, dy))
    rel = (absolute - state.heading + 180) % 360 - 180
    return rel, math.hypot(dx, dy)


class SimDrone:
    """A drone that flies in a simulated room. No hardware, no network.

    Its camera returns real image files: the photo of whatever object is most
    centred in view, or the empty-room photo. Swap the photos in sim/photos for
    pictures of your own room and the vision model sees real images."""

    FOV = 82.0          # Tello's horizontal field of view, degrees
    RANGE = 450.0       # beyond this the camera can't make anything out, cm
    SPEED = 50.0        # cm per second
    TURN = 90.0         # degrees per second

    def __init__(self, world: World):
        self.world = world
        x, y, heading = world.start
        self.state = DroneState(x=x, y=y, heading=heading)

    # -- flight ---------------------------------------------------------------
    def takeoff(self):
        self.state.flying, self.state.z = True, 80.0
        self._spend(2.0, 1.0)

    def land(self):
        self.state.flying, self.state.z = False, 0.0
        self._spend(2.0, 0.5)

    def move(self, direction: str, cm: int):
        dx, dy, dz = offset(self.state.heading, direction, cm)
        self.state.x += dx
        self.state.y += dy
        self.state.z += dz
        self._spend(cm / self.SPEED, 1.0)

    def rotate(self, degrees: int):
        self.state.heading = (self.state.heading + degrees) % 360
        self._spend(abs(degrees) / self.TURN, 0.5)

    def hover(self, seconds: float):
        self._spend(seconds, seconds * 0.1)

    def emergency(self):
        # a real drone cuts its motors and drops; in the sim it just stops
        self.state.flying, self.state.z = False, 0.0

    def _spend(self, seconds: float, battery: float):
        self.state.clock += seconds
        self.state.battery = max(0.0, self.state.battery - battery)

    # -- camera ---------------------------------------------------------------
    def visible(self):
        """Objects in the camera's view, nearest-to-centre first."""
        if not self.state.flying:
            return []
        out = []
        for o in self.world.objects:
            rel, dist = bearing_to(self.state, o.x, o.y)
            if abs(rel) <= self.FOV / 2 and dist <= self.RANGE:
                out.append((o, rel, dist))
        return sorted(out, key=lambda t: abs(t[1]))

    def capture(self, tag: str = "") -> str:
        seen = self.visible()
        return seen[0][0].photo if seen else self.world.empty_photo

    def frame_for(self, obj: Optional[WorldObject]) -> str:
        """The frame the camera shows when it is pointed at this object."""
        return obj.photo if obj else self.world.empty_photo
