"""Reading saved flights back: the list of past missions and one mission's log,
summarised so Claude (and later the showcase page) can see what happened.

A flight log is runs/<mission_id>.json, written by report_mission:
  {"mission_id", "goal", "events": [...], "retrieval_case": {...}}
"""
from __future__ import annotations

import base64
import json
import re
from datetime import datetime
from pathlib import Path

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
MAX_FRAME_BYTES = 5 * 1024 * 1024
_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".webp": "image/webp", ".gif": "image/gif"}


class FlightNotFound(ValueError):
    pass


def log_path(runs_dir: Path, mission_id: str) -> Path:
    mission_id = mission_id.strip().removesuffix(".json")
    if not _ID.match(mission_id) or ".." in mission_id:
        raise FlightNotFound(f"'{mission_id}' is not a mission id (like m-20261116-231651-9f9f).")
    path = (runs_dir / f"{mission_id}.json").resolve()
    if path.parent != runs_dir.resolve() or not path.is_file():
        raise FlightNotFound(f"No saved flight '{mission_id}'. Use list_flights to see the saved ones.")
    return path


def saved(runs_dir: Path) -> list[Path]:
    if not runs_dir.is_dir():
        return []
    return sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _report(events: list) -> dict | None:
    reports = [e for e in events if e.get("kind") == "report"]
    if not reports:
        return None
    last = reports[-1]
    r = last.get("retrieval") or {}
    out = {"summary": last.get("summary", ""), "attempts": len(reports)}
    if "score" in r:
        out.update(score=r["score"], accepted=r.get("passed"),
                   unsupported=r.get("unsupported", []))
    if "error" in r:
        out["error"] = r["error"]
    return out


def summarize(data: dict, path: Path | None = None) -> dict:
    """The parts of a flight a person cares about, in order."""
    events = data.get("events", [])
    checks, claims, safety, errors, moved_cm = [], [], [], [], 0
    for e in events:
        kind = e.get("kind")
        if kind == "verify":
            c = e.get("candidate") or {}
            checks.append({
                "t": e.get("t"), "looking_for": e.get("query"),
                "distance_cm": c.get("distance_cm"), "angle": c.get("angle"),
                "confirmed": bool(e.get("confirmed")), "score": e.get("confidence"),
                "why": e.get("why", ""), "hover_s": e.get("hover_s"),
                "evaluator": e.get("model"), "frame": Path(e.get("frame", "")).name,
            })
        elif kind == "move":
            moved_cm += int(e.get("cm") or 0)
        elif kind == "check_claim":
            claims.append({"t": e.get("t"), "claim": e.get("claim"),
                           "passed": e.get("passed"), "score": e.get("score")})
        elif kind in ("refused", "auto_land", "emergency_stop"):
            safety.append({"t": e.get("t"), "kind": kind, "why": e.get("reason", "")})
        elif kind in ("verify_error", "check_error"):
            errors.append({"t": e.get("t"), "kind": kind, "error": e.get("error", "")})
    out = {
        "mission_id": data.get("mission_id"),
        "goal": data.get("goal", ""),
        "duration_s": events[-1].get("t") if events else 0,
        "landed": any(e.get("kind") == "land" for e in events),
        "confirmed": sorted({c["looking_for"] for c in checks if c["confirmed"]}),
        "rejected": sum(1 for c in checks if not c["confirmed"]),
        "checks": checks,
        "moved_cm": moved_cm,
        "claims_checked": claims,
        "safety_events": safety,
        "errors": errors,
        "report": _report(events),
    }
    if path is not None:
        out["log"] = str(path)
        out["saved_at"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    return out


def listing(runs_dir: Path, limit: int = 10) -> list[dict]:
    rows = []
    for p in saved(runs_dir)[:max(1, min(limit, 100))]:
        try:
            s = summarize(load(p), p)
        except (OSError, ValueError):
            continue
        rep = s["report"] or {}
        rows.append({"mission_id": s["mission_id"] or p.stem, "goal": s["goal"],
                     "saved_at": s["saved_at"], "duration_s": s["duration_s"],
                     "confirmed": s["confirmed"], "rejected": s["rejected"],
                     "report_accepted": rep.get("accepted"), "report_score": rep.get("score")})
    return rows


def frames(data: dict, limit: int = 4) -> list[dict]:
    """The frames worth looking at: each hover check first, then the report's
    frames. Only files the log points at are read."""
    seen, out = set(), []
    for e in data.get("events", []):
        if e.get("kind") == "verify" and e.get("frame"):
            verdict = "confirmed" if e.get("confirmed") else "rejected"
            out.append((e["frame"], f"t={e.get('t')}s: checked for {e.get('query')!r}, "
                                    f"{verdict} (score {e.get('confidence')}). {e.get('why', '')}".strip()))
    for im in (data.get("retrieval_case") or {}).get("images", []):
        src = im.get("src") if isinstance(im, dict) else im
        t = im.get("t") if isinstance(im, dict) else None
        out.append((src, f"t={t}s: frame used for the report check" if t is not None
                    else "frame used for the report check"))
    picked = []
    for src, caption in out:
        if not src or src in seen:
            continue
        seen.add(src)
        p = Path(src)
        mime = _MIME.get(p.suffix.lower())
        if not mime or not p.is_file() or p.stat().st_size > MAX_FRAME_BYTES:
            continue
        picked.append({"caption": caption, "mime": mime,
                       "data": base64.b64encode(p.read_bytes()).decode()})
        if len(picked) >= limit:
            break
    return picked
