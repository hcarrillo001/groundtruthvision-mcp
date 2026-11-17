"""A mission: the drone, the safety gate, the two perception tiers, and the log.

Every action goes the same way:
  1. the SafetyGate checks it (a refusal is logged and returned, never flown),
  2. the driver does it,
  3. the flight log records it with the drone's clock and any frame.

The log is the ground truth. At the end, report() turns Claude's summary of the
mission plus the frames it looked at into a retriEVAL visual_faithfulness case,
so the report can be checked against what the camera actually saw.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .drone import DIRECTIONS
from .perception import Detection
from .safety import Refused, SafetyGate

DETECT_MIN = 0.40      # below this a detection is noise and not worth a hover
CONFIRM_AT = 0.70      # a verifier confidence at or above this confirms a target
STOP_DISTANCE = 100    # approach() stops this far from the target, cm


class Mission:
    def __init__(self, drone, gate: SafetyGate, detector, verifier,
                 runs_dir: str | None = None, goal: str = ""):
        self.drone, self.gate = drone, gate
        self.detector, self.verifier = detector, verifier
        self.id = time.strftime("m-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.goal = goal
        self.events: list[dict] = []
        self.confirmed: dict[str, dict] = {}
        self.runs_dir = Path(runs_dir or os.environ.get("DRONE_RUNS_DIR", "runs"))

    # -- logging ----------------------------------------------------------------
    def _log(self, kind: str, frame: str = "", **detail) -> dict:
        ev = {"t": round(self.drone.state.clock, 2), "kind": kind, **detail}
        if frame:
            ev["frame"] = frame
        self.events.append(ev)
        return ev

    def _status(self) -> dict:
        s = self.drone.state
        return {"flying": s.flying, "x_cm": round(s.x), "y_cm": round(s.y),
                "height_cm": round(s.z), "heading": round(s.heading),
                "battery": round(s.battery), "clock_s": round(s.clock, 1)}

    def _guard(self, check, *args) -> Optional[dict]:
        """Run a gate check; on refusal, log it and return the refusal."""
        try:
            check(self.drone.state, *args)
            return None
        except Refused as e:
            self._log("refused", reason=str(e))
            return {"ok": False, "refused": str(e), "status": self._status()}

    def _after(self, out: dict) -> dict:
        """After every action: land automatically if the battery hit the floor."""
        if self.gate.battery_low(self.drone.state):
            self.drone.land()
            self._log("auto_land", reason="battery below floor")
            out["auto_landed"] = "Battery fell below the floor, so the drone landed."
        out["status"] = self._status()
        return out

    # -- flight -----------------------------------------------------------------
    def takeoff(self) -> dict:
        if (r := self._guard(self.gate.check_takeoff)):
            return r
        self.drone.takeoff()
        self._log("takeoff")
        return self._after({"ok": True})

    def land(self) -> dict:
        if (r := self._guard(self.gate.check_land)):
            return r
        self.drone.land()
        self._log("land")
        return self._after({"ok": True})

    def emergency_stop(self) -> dict:
        self.drone.emergency()
        self._log("emergency_stop")
        return {"ok": True, "status": self._status()}

    def move(self, direction: str, cm: int) -> dict:
        direction = direction.lower().strip()
        if (r := self._guard(self.gate.check_move, direction, cm)):
            return r
        self.drone.move(direction, cm)
        self._log("move", direction=direction, cm=cm)
        return self._after({"ok": True})

    def rotate(self, degrees: int) -> dict:
        if (r := self._guard(self.gate.check_rotate, degrees)):
            return r
        self.drone.rotate(degrees)
        self._log("rotate", degrees=degrees)
        return self._after({"ok": True})

    # -- perception -------------------------------------------------------------
    def look(self, query: str) -> dict:
        """Tier 1 only: what the detector sees right now. Fast, easily fooled."""
        if not self.drone.state.flying:
            return {"ok": False, "refused": "Not flying. Call takeoff first."}
        dets = self.detector.detect(self.drone, query)
        frame = self.drone.capture("look")
        self._log("look", frame=frame, query=query,
                  detections=[_det(d) for d in dets])
        return {"ok": True, "detections": [_det(d) for d in dets],
                "frame": frame, "status": self._status()}

    def verify(self, d: Detection, query: str) -> dict:
        """Tier 2: hover and let the vision model look at the candidate."""
        if (r := self._guard(self.gate.check_hover)):
            return r
        t0 = time.perf_counter()
        try:
            v = self.verifier.verify(d.frame, query, ref=d.ref)
        except Exception as e:                # evaluator down, no judge key, ...
            self.drone.hover(time.perf_counter() - t0)
            self._log("verify_error", frame=d.frame, query=query, error=str(e))
            return {"confirmed": False, "error": f"Could not evaluate the candidate: {e}"}
        self.drone.hover(v.seconds)          # the drone held still for this long
        ok = v.match and v.confidence >= CONFIRM_AT
        self._log("verify", frame=d.frame, query=query, candidate=_det(d),
                  match=v.match, confidence=round(v.confidence, 2), confirmed=ok,
                  why=v.why, hover_s=round(v.seconds, 2), model=v.model)
        return {"confirmed": ok, "match": v.match, "confidence": round(v.confidence, 2),
                "why": v.why, "hover_seconds": round(v.seconds, 2), "model": v.model}

    def scan_for(self, query: str, step: int = 45) -> dict:
        """Turn on the spot in steps. At each step the detector looks; any
        candidate gets a hover-and-verify. Stops at the first confirmed match
        and turns to face it. Rejected candidates are reported too, because a
        decoy that fooled the detector is worth knowing about."""
        if not self.drone.state.flying:
            return {"ok": False, "refused": "Not flying. Call takeoff first."}
        rejected = []
        for i in range(max(1, 360 // step)):
            if i:
                r = self.rotate(step)
                if not r.get("ok"):
                    return {**r, "found": False, "rejected": rejected}
            dets = [d for d in self.detector.detect(self.drone, query) if d.confidence >= DETECT_MIN]
            self._log("look", frame=self.drone.capture("scan"), query=query,
                      detections=[_det(d) for d in dets])
            for d in dets:
                v = self.verify(d, query)
                if v.get("refused") or v.get("error"):
                    return self._after({**v, "ok": False, "found": False, "rejected": rejected})
                if v["confirmed"]:
                    if abs(d.angle) >= 5:
                        self.rotate(round(d.angle))
                    self.confirmed[query] = {"distance_cm": d.distance_cm, "frame": d.frame,
                                             "confidence": v["confidence"], "ref": d.ref}
                    return self._after({"ok": True, "found": True, "query": query,
                                        "distance_cm": d.distance_cm,
                                        "verify": v, "rejected": rejected, "frame": d.frame})
                rejected.append({"candidate": _det(d), "verify": v})
        return self._after({"ok": True, "found": False, "query": query, "rejected": rejected})

    def approach(self, query: str) -> dict:
        """Fly toward a target that scan_for confirmed, stopping about a metre away."""
        t = self.confirmed.get(query)
        if not t:
            return {"ok": False, "refused": f"No confirmed '{query}'. Run scan_for first."}
        moved = 0
        while True:
            dets = [d for d in self.detector.detect(self.drone, query) if d.ref is t["ref"]] \
                if t["ref"] is not None else self.detector.detect(self.drone, query)[:1]
            if not dets:
                return self._after({"ok": False, "lost_target": True, "moved_cm": moved,
                                    "hint": "Target is out of view. scan_for it again."})
            d = dets[0]
            if abs(d.angle) >= 5:
                r = self.rotate(round(d.angle))
                if not r.get("ok"):
                    return {**r, "moved_cm": moved}
            remaining = d.distance_cm - STOP_DISTANCE
            if remaining < 20:
                return self._after({"ok": True, "arrived": True, "distance_cm": d.distance_cm,
                                    "moved_cm": moved, "frame": d.frame})
            step = int(min(remaining, self.gate.limits.max_move))
            r = self.move("forward", step)
            if not r.get("ok"):
                return {**r, "moved_cm": moved}
            moved += step

    # -- reporting --------------------------------------------------------------
    def frames_for_report(self, limit: int = 20) -> list:
        """The frames that back the mission's claims: every verified candidate
        first, then frames where something was detected, oldest first."""
        seen, out = set(), []
        ranked = [e for e in self.events if e["kind"] == "verify"] + \
                 [e for e in self.events if e["kind"] == "look" and e.get("detections")]
        for e in sorted(ranked, key=lambda e: e["t"]):
            f = e.get("frame")
            if f and f not in seen:
                seen.add(f)
                out.append({"src": f, "t": e["t"]})
        return out[:limit]

    def recent_frames(self, n: int = 3) -> list:
        frames = [{"src": e["frame"], "t": e["t"]} for e in self.events if e.get("frame")]
        if self.drone.state.flying:
            frames.append({"src": self.drone.capture("check"), "t": round(self.drone.state.clock, 2)})
        seen, out = set(), []
        for f in reversed(frames):
            if f["src"] not in seen:
                seen.add(f["src"])
                out.append(f)
        return list(reversed(out[:n]))

    def check_claim(self, claim: str) -> dict:
        """Mid-flight: score a claim against what the camera is seeing now."""
        check = getattr(self.verifier, "check_claims", None)
        if check is None:
            return {"ok": False, "refused": "Live claim checks need retriEVAL as the evaluator "
                                            "(DRONE_VERIFIER=retrieval)."}
        frames = self.recent_frames()
        t0 = time.perf_counter()
        try:
            r = check(claim, frames, self.goal or "Drone mission", save=False)
        except Exception as e:
            self._log("check_error", claim=claim, error=str(e))
            return {"ok": False, "error": f"retriEVAL could not check the claim: {e}"}
        if self.drone.state.flying:
            self.drone.hover(time.perf_counter() - t0)
        self._log("check_claim", claim=claim, score=r["score"], passed=r["passed"],
                  unsupported=r["unsupported"], frames=[f["src"] for f in frames])
        return self._after({"ok": True, "claim": claim, **r})

    def report(self, summary: str) -> dict:
        """Check the summary with retriEVAL (when it is the evaluator), then save
        the flight log and the retriEVAL case."""
        case = {"input": self.goal or "Drone mission", "actual_output": summary,
                "images": self.frames_for_report()}
        check, check_out = getattr(self.verifier, "check_claims", None), None
        if check is not None and case["images"]:
            try:
                check_out = check(summary, case["images"], case["input"], save=True)
            except Exception as e:
                check_out = {"error": f"retriEVAL could not check the report: {e}"}
        self._log("report", summary=summary,
                  **({"retrieval": {k: v for k, v in check_out.items()
                                    if k in ("score", "passed", "unsupported", "error")}}
                     if check_out else {}))
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.runs_dir / f"{self.id}.json"
        path.write_text(json.dumps({"mission_id": self.id, "goal": self.goal,
                                    "events": self.events, "retrieval_case": case},
                                   indent=2))
        out = {"ok": True, "mission_id": self.id, "log": str(path), "retrieval_case": case}
        if check_out and "error" not in check_out:
            out["retrieval_check"] = check_out
            out["accepted"] = check_out["passed"]
            if not check_out["passed"]:
                out["next"] = ("retriEVAL found claims the camera frames don't support: "
                               + "; ".join(c.rstrip(". ") for c in check_out["unsupported"])
                               + ". Rewrite the report with only what was confirmed and report again.")
        else:
            if check_out:
                out["retrieval_check"] = check_out
            out["next"] = ("Check this report with retriEVAL: run_eval with metrics "
                           "['visual_faithfulness'] and cases = [retrieval_case].")
        return out


def _det(d: Detection) -> dict:
    out = asdict(d)
    out.pop("ref", None)
    return out
