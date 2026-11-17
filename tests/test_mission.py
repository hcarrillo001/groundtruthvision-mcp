"""owlsightMCP tests: the safety gate, the hover-and-verify loop, and the report.
All in the simulator: no drone, no network.

    cd owlsightMCP && python -m pytest -q
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owlsight_mcp.drone import SimDrone, World, bearing_to          # noqa: E402
from owlsight_mcp.mission import Mission                              # noqa: E402
from owlsight_mcp.perception import (SimDetector, SimVerifier, Verdict,  # noqa: E402
                                  _parse_verdict)
from owlsight_mcp.safety import FlightBox, Limits, SafetyGate         # noqa: E402

WORLD = str(ROOT / "sim/worlds/living_room.json")


@pytest.fixture
def m(tmp_path):
    world = World.load(WORLD)
    gate = SafetyGate(FlightBox(0, 0, world.width, world.depth), Limits())
    return Mission(SimDrone(world), gate, SimDetector(), SimVerifier(),
                   runs_dir=str(tmp_path), goal="find the blue backpack")


# ---- safety gate ----------------------------------------------------------------

def test_cannot_move_before_takeoff(m):
    r = m.move("forward", 100)
    assert r["ok"] is False and "takeoff" in r["refused"]
    assert m.drone.state.y == 80                       # nothing was flown


def test_move_size_limits(m):
    m.takeoff()
    assert "20-200" in m.move("forward", 10)["refused"]
    assert "20-200" in m.move("forward", 500)["refused"]


def test_walls_are_refused(m):
    m.takeoff()
    m.rotate(180)                                      # face the near wall (y=0)
    r = m.move("forward", 60)                          # would reach y=20, inside the margin
    assert r["ok"] is False and "flight box" in r["refused"]
    assert round(m.drone.state.y) == 80


def test_height_limits(m):
    m.takeoff()                                        # 80 cm
    assert "ceiling" in m.move("up", 150)["refused"]
    assert "minimum" in m.move("down", 60)["refused"]


def test_unknown_direction(m):
    m.takeoff()
    assert "Unknown direction" in m.move("sideways", 50)["refused"]


def test_low_battery_blocks_takeoff_and_auto_lands(m):
    m.drone.state.battery = 25
    assert "needed to take off" in m.takeoff()["refused"]
    m.drone.state.battery = 40
    m.takeoff()
    m.drone.state.battery = 20.5
    r = m.move("forward", 100)                         # costs 1%, drops below 20
    assert r["auto_landed"]
    assert m.drone.state.flying is False
    assert any(e["kind"] == "auto_land" for e in m.events)


def test_refusals_are_logged(m):
    m.move("forward", 100)
    assert m.events[-1]["kind"] == "refused"


def test_emergency_stop_always_works(m):
    m.takeoff()
    assert m.emergency_stop()["ok"] and m.drone.state.flying is False


# ---- perception: the decoy is caught by the verifier --------------------------------

def test_detector_cannot_tell_backpacks_apart(m):
    m.takeoff()
    labels = {d.ref.description for d in SimDetector().detect(m.drone, "blue backpack")}
    assert labels == {"blue backpack", "black backpack"}


def test_scan_rejects_decoy_and_confirms_target(m):
    m.takeoff()
    r = m.scan_for("blue backpack")
    assert r["found"] is True
    assert [x["verify"]["why"] for x in r["rejected"]] == ["the object is a black backpack"]
    angle, _ = bearing_to(m.drone.state, 470, 400)
    assert abs(angle) < 5                              # turned to face it
    kinds = [e["kind"] for e in m.events]
    assert kinds.count("verify") == 2                  # hovered for both candidates


def test_hover_time_is_spent_while_verifying(m):
    m.takeoff()
    before = m.drone.state.clock
    m.scan_for("blue backpack")
    hover = sum(e["hover_s"] for e in m.events if e["kind"] == "verify")
    assert hover == pytest.approx(3.0)                 # 2 checks x 1.5 s
    assert m.drone.state.clock - before >= hover


def test_scan_finds_nothing_for_absent_target(m):
    m.takeoff()
    r = m.scan_for("red bicycle", step=90)
    assert r["found"] is False and r["rejected"] == []


def test_approach_needs_a_confirmed_target(m):
    m.takeoff()
    assert "scan_for" in m.approach("blue backpack")["refused"]


def test_approach_stops_about_a_metre_away(m):
    m.takeoff()
    m.scan_for("blue backpack")
    r = m.approach("blue backpack")
    assert r["arrived"] is True
    _, dist = bearing_to(m.drone.state, 470, 400)
    assert 80 <= dist <= 120


# ---- report: the log becomes a retriEVAL case ----------------------------------------

def test_report_builds_visual_faithfulness_case(m, tmp_path):
    m.takeoff()
    m.scan_for("blue backpack")
    m.land()
    r = m.report("Found a blue backpack; rejected a black one.")
    case = r["retrieval_case"]
    assert case["input"] == "find the blue backpack"
    srcs = [Path(i["src"]).name for i in case["images"]]
    assert "blue_backpack.png" in srcs and "black_backpack.png" in srcs
    assert all("t" in i for i in case["images"])
    saved = json.loads(Path(r["log"]).read_text())
    assert saved["retrieval_case"] == case
    assert saved["events"][-1]["kind"] == "report"


def test_report_frames_are_capped(m):
    m.takeoff()
    for i in range(30):
        m._log("verify", frame=f"/frames/{i}.jpg")
    assert len(m.frames_for_report()) == 20


# ---- verifier parsing (shared by the Ollama and NVIDIA verifiers) ---------------------

def test_parse_verdict_handles_thinking_and_bad_json():
    v = _parse_verdict('<think>{"match": false}</think>{"match": true, "confidence": 0.9, "why": "blue"}',
                       1.2, "m")
    assert v.match is True and v.confidence == 0.9
    bad = _parse_verdict("I think so", 1.0, "m")
    assert bad.match is False and "JSON" in bad.why


def test_ollama_verifier_sends_image(monkeypatch, tmp_path):
    import io
    from owlsight_mcp.perception import OllamaVerifier
    img = tmp_path / "f.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 8)
    sent = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, timeout=None):
        sent.update(json.loads(req.data))
        return Resp(json.dumps({"message": {"content": '{"match": true, "confidence": 0.8, "why": "ok"}'}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake)
    v = OllamaVerifier(model="moondream").verify(str(img), "blue backpack")
    assert v.match and v.model == "moondream"
    assert sent["messages"][0]["images"] and "blue backpack" in sent["messages"][0]["content"]


def test_nvidia_verifier_defaults_to_nemotron(monkeypatch):
    from owlsight_mcp.perception import OpenAIVisionVerifier
    monkeypatch.delenv("DRONE_VISION_MODEL", raising=False)
    v = OpenAIVisionVerifier(key="k")
    assert v.model.startswith("nvidia/nemotron") and "openrouter" in v.base_url


# ---- MCP server ---------------------------------------------------------------------

def test_server_tools_fly_a_mission(monkeypatch, tmp_path):
    monkeypatch.setenv("DRONE_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("DRONE_VERIFIER", "sim")    # no retriEVAL or judge key needed
    from owlsight_mcp import server as S
    assert S.start_mission("find the blue backpack")["ok"]
    S.takeoff()
    assert S.scan_for("blue backpack")["found"] is True
    assert "still flying" in S.start_mission("again")["refused"]   # can't reset mid-flight
    S.land()
    assert S.report_mission("Found it.")["ok"]


def test_server_lists_expected_tools():
    import asyncio
    from owlsight_mcp import server as S
    names = {t.name for t in asyncio.run(S.mcp.list_tools())}
    assert {"start_mission", "takeoff", "land", "move", "rotate", "look", "scan_for",
            "approach", "check_claim", "status", "emergency_stop", "report_mission"} <= names
