"""The MCP server: the tools Claude (or any MCP client) calls to fly the drone.

Run it:
  python -m owlsight_mcp.server                 # stdio, for Claude Desktop / Cursor
  DRONE_DRIVER=tello python -m owlsight_mcp.server

Configuration (environment variables):
  DRONE_DRIVER         "sim" (default) or "tello"
  DRONE_WORLD          world file for the simulator (default sim/worlds/living_room.json)
  DRONE_VERIFIER       "retrieval" (default when retriEVAL is found: it scores
                       every candidate and the report, live), "sim", "ollama", "openai"
  RETRIEVAL_SERVER     path to retriEVAL's server.py (see owlsight_mcp/retrieval.py)
  RETRIEVAL_JUDGE      retriEVAL's judge for these checks, e.g. "nemotron"
  OLLAMA_VISION_MODEL  e.g. qwen2.5vl:7b or moondream, for DRONE_VERIFIER=ollama
  DRONE_BOX_CM         size of the flight box around take-off on a real drone (default 400)
  DRONE_RUNS_DIR       where flight logs are saved (default runs/ in this project)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from . import flights

from .drone import SimDrone, World
from .mission import Mission
from .perception import OllamaVerifier, SimDetector, SimVerifier
from .safety import FlightBox, SafetyGate

ROOT = Path(__file__).resolve().parent.parent
mcp = FastMCP("owlsight")
_mission: Mission | None = None


def runs_dir() -> Path:
    return Path(os.environ.get("DRONE_RUNS_DIR", str(ROOT / "runs"))).expanduser()


def _build(goal: str = "") -> Mission:
    driver = os.environ.get("DRONE_DRIVER", "sim").lower()
    verifier_name = os.environ.get("DRONE_VERIFIER", "").lower() or default_evaluator()
    if driver == "tello":
        from .tello import TelloDrone
        # On home Wi-Fi (station mode) the router gives the Tello its own address;
        # find it in the router's device list or with tello_join_wifi.py.
        drone = TelloDrone(frames_dir=str(ROOT / "runs" / "frames"),
                           host=os.environ.get("TELLO_HOST") or None)
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
    if verifier_name == "retrieval":
        verifier = retrieval_verifier()
    elif verifier_name == "ollama":
        verifier = OllamaVerifier()
    elif verifier_name == "openai":
        from .perception import OpenAIVisionVerifier
        verifier = OpenAIVisionVerifier()
    else:
        verifier = SimVerifier()
    runs = str(runs_dir())
    return Mission(drone, SafetyGate(box), detector, verifier, runs_dir=runs, goal=goal)


_retrieval = None


def retrieval_verifier():
    """One retriEVAL session for the life of this server, shared by missions."""
    global _retrieval
    if _retrieval is None:
        from .retrieval import RetrievalVerifier
        _retrieval = RetrievalVerifier()
    return _retrieval


def default_evaluator() -> str:
    """retriEVAL when it can be found; otherwise the simulator's stand-in."""
    from .retrieval import DEFAULT_SERVER
    path = Path(os.environ.get("RETRIEVAL_SERVER", DEFAULT_SERVER)).expanduser()
    return "retrieval" if path.is_file() else "sim"


def evaluator_name(m: Mission) -> str:
    return getattr(m.verifier, "model", type(m.verifier).__name__)


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
            "evaluator": evaluator_name(_mission),
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
def check_claim(claim: str) -> dict:
    """Mid-flight: have retriEVAL check a claim against what the camera is
    seeing now, e.g. "The backpack is on the couch." Returns a score, pass/fail,
    and any part of the claim the frames don't support. Use it before you
    tell the user something about what the drone sees."""
    return mission().check_claim(claim)


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
    """After landing: write what the drone found, in plain words. retriEVAL
    checks the summary against the frames the camera captured; if it finds
    unsupported claims, accepted=false lists them, so rewrite and report again.
    Saves the flight log either way. Only claim what was confirmed."""
    return mission().report(summary)


@mcp.tool()
def list_flights(limit: int = 10) -> dict:
    """Saved missions, newest first: goal, when, how long, what was confirmed,
    how many look-alikes retriEVAL rejected, and whether the report passed.
    Use a mission_id from here with get_flight_log."""
    rows = flights.listing(runs_dir(), limit)
    return {"flights": rows, "runs_dir": str(runs_dir()),
            **({} if rows else {"note": "No saved flights yet. A flight is saved by report_mission."})}


@mcp.tool()
def get_flight_log(mission_id: str = "", show_frames: bool = False, max_frames: int = 4,
                   include_events: bool = False):
    """Open a saved flight. Leave mission_id empty for the most recent one.

    Returns a summary: every hover check retriEVAL made (what it looked for,
    distance, confirmed or rejected, score, why, seconds), claims checked
    mid-flight, safety refusals, distance flown, and the report verdict.
    show_frames=true also returns the camera frames behind those checks as
    images, each with a caption, so you can see what retriEVAL saw.
    include_events=true adds the raw event log."""
    rd = runs_dir()
    try:
        if mission_id.strip():
            path = flights.log_path(rd, mission_id)
        else:
            files = flights.saved(rd)
            if not files:
                return {"ok": False, "error": "No saved flights yet. A flight is saved by report_mission."}
            path = files[0]
        data = flights.load(path)
    except (flights.FlightNotFound, OSError, ValueError) as e:
        return {"ok": False, "error": str(e)}
    out = {"ok": True, **flights.summarize(data, path)}
    if include_events:
        out["events"] = data.get("events", [])
    if not show_frames:
        return out
    pics = flights.frames(data, max(1, min(max_frames, 10)))
    out["frames_shown"] = len(pics)
    content = [TextContent(type="text", text=json.dumps(out, indent=2))]
    for f in pics:
        content.append(TextContent(type="text", text=f["caption"]))
        content.append(ImageContent(type="image", data=f["data"], mimeType=f["mime"]))
    return content


def main():
    mcp.run()


if __name__ == "__main__":
    main()
