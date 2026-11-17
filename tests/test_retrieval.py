"""GroundTruthVision <-> retriEVAL over MCP: a real stdio session to a stand-in server."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from groundtruthvision_mcp.drone import SimDrone, World                     # noqa: E402
from groundtruthvision_mcp.mission import Mission                            # noqa: E402
from groundtruthvision_mcp.perception import SimDetector                     # noqa: E402
from groundtruthvision_mcp.retrieval import (RetrievalClient, RetrievalError,  # noqa: E402
                                    RetrievalVerifier)
from groundtruthvision_mcp.safety import FlightBox, SafetyGate               # noqa: E402

FAKE = str(Path(__file__).with_name("fake_retrieval_server.py"))
WORLD = str(ROOT / "sim/worlds/living_room.json")


@pytest.fixture(scope="module")
def evaluator():
    v = RetrievalVerifier(RetrievalClient(command=sys.executable, args=[FAKE]), judge="nemotron")
    yield v
    v.client.close()


@pytest.fixture
def m(evaluator, tmp_path):
    world = World.load(WORLD)
    return Mission(SimDrone(world), SafetyGate(FlightBox(0, 0, world.width, world.depth)),
                   SimDetector(), evaluator, runs_dir=str(tmp_path), goal="find the blue backpack")


def test_verify_goes_through_retrieval(evaluator):
    photos = ROOT / "sim/photos"
    blue = evaluator.verify(str(photos / "blue_backpack.png"), "blue backpack")
    black = evaluator.verify(str(photos / "black_backpack.png"), "blue backpack")
    assert blue.match and blue.confidence == 1.0
    assert not black.match and black.confidence == 0.5
    assert "It is blue." in black.why
    assert evaluator.model == "retriEVAL/nemotron"


def test_scan_rejects_decoy_via_retrieval(m):
    m.takeoff()
    r = m.scan_for("blue backpack")
    assert r["found"] is True
    assert r["verify"]["model"] == "retriEVAL/nemotron"
    assert len(r["rejected"]) == 1 and r["rejected"][0]["verify"]["confidence"] == 0.5


def test_check_claim_mid_flight(m):
    m.takeoff()
    m.scan_for("blue backpack")                 # now facing the blue backpack
    good = m.check_claim("The backpack is blue.")
    bad = m.check_claim("The backpack is red.")
    assert good["passed"] is True
    assert bad["passed"] is False and bad["unsupported"] == ["It is red."]
    assert [e["kind"] for e in m.events].count("check_claim") == 2


def test_report_gate_rejects_unsupported_claims(m):
    m.takeoff()
    m.scan_for("blue backpack")
    m.land()
    bad = m.report("Found a red backpack.")
    assert bad["accepted"] is False and "red" in bad["next"]
    good = m.report("Found a blue backpack.")
    assert good["accepted"] is True
    assert good["retrieval_check"]["report_url"] == "https://example.test/report/r-1"


def test_missing_retrieval_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_SERVER", "/nope/server.py")
    with pytest.raises(RetrievalError, match="RETRIEVAL_SERVER"):
        RetrievalClient()


def test_evaluator_failure_does_not_crash_the_flight(tmp_path):
    class Broken:
        model = "broken"

        def verify(self, *a, **k):
            raise RuntimeError("no judge key")

    world = World.load(WORLD)
    m = Mission(SimDrone(world), SafetyGate(FlightBox(0, 0, world.width, world.depth)),
                SimDetector(), Broken(), runs_dir=str(tmp_path))
    m.takeoff()
    r = m.scan_for("blue backpack")
    assert r["found"] is False and "no judge key" in r["error"]
    assert m.drone.state.flying                # still hovering, ready for land()


def _fly(m):
    m.takeoff()
    m.scan_for("blue backpack")
    a = m.approach("blue backpack")
    m.land()
    return a


def test_flight_facts_record_what_a_camera_cannot_show(m):
    a = _fly(m)
    facts = " ".join(m.flight_facts())
    assert "climbed to 80 cm" in facts
    assert "rejected it" in facts and "confirmed it" in facts
    assert f"stopped {a['distance_cm']} cm from the blue backpack" in facts
    assert f"after flying {a['moved_cm']} cm" in facts
    assert "landed" in facts


def test_report_with_distances_passes_on_the_flight_log(m):
    """The real-flight bug: true distances were rejected because no frame shows them."""
    a = _fly(m)
    r = m.report(f"Found the blue backpack and flew {a['moved_cm']} cm toward it.")
    assert r["accepted"] is True
    assert r["retrieval_case"]["context"] == m.flight_facts()


def test_report_with_a_wrong_distance_fails(m):
    a = _fly(m)
    r = m.report(f"Found the blue backpack and flew {a['moved_cm'] + 999} cm toward it.")
    assert r["accepted"] is False
    assert "flight log" in r["next"]
