"""The MCP server: the tools Claude (or any MCP client) calls to fly the drone.

Run it:
  python -m drone_mcp.server                 # stdio, for Claude Desktop / Cursor
  DRONE_DRIVER=tello python -m drone_mcp.server

Configuration (environment variables):
  DRONE_DRIVER         "sim" (default) or "tello"
  DRONE_WORLD          world file for the simulator (default sim/worlds/living_room.json)
  DRONE_VERIFIER       "sim" (default with the simulator), "ollama", or "openai"
  OLLAMA_VISION_MODEL  e.g. qwen2.5vl:7b or moondream, for DRONE_VERIFIER=ollama
  DRONE_BOX_CM         size of the flight box around take-off on a real drone (default 400)
  DRONE_RUNS_DIR       where flight logs are saved (default runs/ in this project)
"""
from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .drone import SimDrone, World
from .mission import Mission
from .perception import OllamaVerifier, SimDetector, SimVerifier
from .safety import FlightBox, SafetyGate

ROOT = Path(__file__).resolve().parent.parent
mcp = FastMCP("drone-mcp")
_mission: Mission | None = None


def _build(goal: str = "") -> Mission:
    driver = os.environ.get("DRONE_DRIVER", "sim").lower()
    verifier_name = os.environ.get("DRONE_VERIFIER", "sim" if driver == "sim" else "ollama").lower()
    if driver == "tello":
        from .tello import TelloDrone
        drone = TelloDrone()
        half = float(os.environ.get("DRONE_BOX_CM", "400")) / 2
        box = FlightBox.around(0, 0, half, half)
        # a real detector (open-vocabulary YOLO) plugs in here once installed
        from .yolo import YoloWorldDetector
        detector = YoloWorldDetector()
    else:
        world = World.load(os.environ.get("DRONE_WORLD", str(ROOT / "sim/worlds/living_room.json")))
        drone = SimDrone(world)
        box = FlightBox(0, 0, world.width, world.depth)
        detector = SimDetector()
    if verifier_name == "ollama":
        verifier = OllamaVerifier()
    elif verifier_name == "openai":
        from .perception import OpenAIVisionVerifier
        verifier = OpenAIVisionVerifier()
    else:
        verifier = SimVerifier()
    runs = os.environ.get("DRONE_RUNS_DIR", str(ROOT / "runs"))
    return Mission(drone, SafetyGate(box), detector, verifier, runs_dir=runs, goal=goal)


def mission() -> Mission:
    global _mission
    if _mission is None:
        _mission = _build()
    return _mission


@mcp.tool()
def start_mission(goal: str) -> dict:
    """Begin a new mission with a goal in plain words, e.g. "find the blue
    backpack". Resets the flight log. Call this first."""
    global _mission
    if _mission is not None and _mission.drone.state.flying:
        return {"ok": False, "refused": "The drone is still flying. Land before starting a new mission."}
    _mission = _build(goal)
    return {"ok": True, "mission_id": _mission.id, "goal": goal,
            "status": _mission._status(),
            "how": "takeoff, then scan_for the target; it hovers and verifies each "
                   "candidate. approach a confirmed target, land, then report."}


@mcp.tool()
def takeoff() -> dict:
    """Take off and hover at about 80 cm."""
    return mission().takeoff()


@mcp.tool()
def land() -> dict:
    """Land where the drone is."""
    return mission().land()


@mcp.tool()
def move(direction: str, cm: int) -> dict:
    """Move relative to where the drone faces: forward, back, left, right, up or
    down, 20-200 cm. The safety gate refuses moves that leave the flight box,
    come too close to a wall, or break the height limits."""
    return mission().move(direction, cm)


@mcp.tool()
def rotate(degrees: int) -> dict:
    """Turn on the spot. Positive = clockwise (right), negative = left."""
    return mission().rotate(degrees)


@mcp.tool()
def look(target: str) -> dict:
    """What the fast detector sees right now that might be `target`. Detections
    are candidates, not confirmations: the detector can't tell colours or
    details apart. Use scan_for to confirm."""
    return mission().look(target)


@mcp.tool()
def scan_for(target: str, step_degrees: int = 45) -> dict:
    """Turn in place looking for `target` (e.g. "blue backpack"). Each candidate
    the detector finds is checked by the vision model while the drone hovers.
    Returns found=true with the distance once a candidate is confirmed (and the
    drone turned to face it), plus any rejected look-alikes."""
    return mission().scan_for(target, step_degrees)


@mcp.tool()
def approach(target: str) -> dict:
    """Fly toward a target that scan_for confirmed, stopping about a metre away."""
    return mission().approach(target)


@mcp.tool()
def status() -> dict:
    """Position (estimated), height, heading, battery, and mission clock."""
    m = mission()
    return {"mission_id": m.id, "goal": m.goal, "status": m._status(),
            "confirmed_targets": sorted(m.confirmed)}


@mcp.tool()
def emergency_stop() -> dict:
    """Stop the motors immediately. On a real drone it DROPS. Only for danger."""
    return mission().emergency_stop()


@mcp.tool()
def report_mission(summary: str) -> dict:
    """After landing: write what the drone found, in plain words. Saves the full
    flight log and returns a retriEVAL case pairing your summary with the frames
    the camera actually captured, so the summary can be checked with
    visual_faithfulness. Only claim what the verifications confirmed."""
    return mission().report(summary)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
