# drone-mcp

Fly a small drone from Claude, Cursor or any MCP client, with a safety gate the
AI can't talk its way past, a hover-and-verify loop so it doesn't believe the
first thing it sees, and a flight log that [retriEVAL](https://github.com/hcarrillo001/retrieval-mcp)
can check its mission report against.

It runs in a **simulator today** and is built to swap in a **Tello EDU**.

## The idea

> "Find the blue backpack."

1. The drone takes off and turns on the spot, looking.
2. A fast detector spots candidates on every frame. It knows *backpack*, not
   *blue*, so a black backpack fools it.
3. For each candidate the drone **hovers** while a vision model looks closer:
   "is this a blue backpack?" The black one is rejected; the blue one confirmed.
4. It turns to face the target and flies to about a metre away.
5. After landing, Claude writes a report. `report_mission` pairs it with the
   frames the camera actually captured, as a retriEVAL `visual_faithfulness`
   case, so any claim the camera doesn't back up gets caught.

## Layers

```
Claude ──MCP──> server.py     the tools Claude calls
                mission.py    runs every action through the steps below, keeps the log
                safety.py     SafetyGate: refuses unsafe commands before they fly
                perception.py Detector (every frame) + Verifier (hover, look closer)
                drone.py      SimDrone, or tello.py for the real Tello EDU
```

The rule: **Claude decides where to go, the safety gate decides whether it's
allowed, and the gate always wins.** Refusals come back as plain sentences
Claude can act on ("that move would leave the flight box"), and nothing is flown.

## Tools

| Tool | What it does |
|---|---|
| `start_mission(goal)` | New mission and a fresh flight log |
| `takeoff` / `land` | Take off to ~80 cm; land |
| `move(direction, cm)` | forward, back, left, right, up, down; 20-200 cm |
| `rotate(degrees)` | Positive turns right, negative left |
| `look(target)` | What the detector sees right now (candidates only) |
| `scan_for(target)` | Turn and look; hover-and-verify each candidate; stop on a confirmed match |
| `approach(target)` | Fly toward a confirmed target, stop ~1 m away |
| `status` | Position, height, heading, battery, clock |
| `emergency_stop` | Cut the motors. On a real drone it drops |
| `report_mission(summary)` | Save the log; return the retriEVAL case for the summary |

## Safety limits

Set by environment variables; defaults suit a small drone indoors.

| Variable | Default | Meaning |
|---|---|---|
| `DRONE_MAX_HEIGHT_CM` | 200 | Ceiling |
| `DRONE_MIN_HEIGHT_CM` | 30 | Lowest it may fly; below that, land |
| `DRONE_MAX_MOVE_CM` | 200 | Longest single move |
| `DRONE_WALL_MARGIN_CM` | 40 | Keep-out distance from the flight box edge |
| `DRONE_BATTERY_FLOOR` | 20 | Below this it refuses moves and lands itself |
| `DRONE_TAKEOFF_BATTERY` | 30 | Minimum battery to take off |
| `DRONE_BOX_CM` | 400 | Real drone: size of the flight box around take-off |

Landing and emergency stop are never refused.

## Run it in the simulator

```bash
pip install -r requirements.txt
python -m pytest -q                 # 21 tests, no drone or network needed
python run.py                       # MCP over stdio
```

Cursor (`~/.cursor/mcp.json`) or Claude Desktop:

```json
{
  "mcpServers": {
    "drone": {
      "command": "python3",
      "args": ["/Users/hcarrillo/Workspace/drone-mcp/run.py"]
    }
  }
}
```

Use the full path to the Python that has the requirements installed if
`python3` isn't it (for a virtualenv: `/Users/hcarrillo/Workspace/drone-mcp/.venv/bin/python`).

Then ask: *"Start a mission to find the blue backpack, take off, scan for it,
approach it, land, and report what you found."*

The simulated room is `sim/worlds/living_room.json`. Its camera shows the
pictures in `sim/photos`; replace them with photos of your own room and the
vision model sees real images.

## Choosing the verifier

| `DRONE_VERIFIER` | Uses | When |
|---|---|---|
| `sim` | The simulated world's ground truth | Default with the simulator |
| `ollama` | A vision model on your own computer (`OLLAMA_VISION_MODEL`, e.g. `qwen2.5vl:7b` or `moondream`) | Real flights: free, private, no rate limits, ~1-3 s per check |
| `openai` | Any OpenAI-compatible vision model; default NVIDIA `nemotron-3.5-super-vl:free` on OpenRouter (`OPENROUTER_API_KEY`) | Developing with test photos. Free tier is rate limited and may log images, so not for flights over your home |

Each check's time is recorded as `hover_s` in the log, so you can measure your
own setup before flying.

## Checking the report with retriEVAL

`report_mission` returns:

```json
{"retrieval_case": {"input": "find the blue backpack",
                    "actual_output": "Found a blue backpack ...",
                    "images": [{"src": ".../frame.jpg", "t": 5.0}]}}
```

Pass it to retriEVAL's `run_eval` with `metrics: ["visual_faithfulness"]`
(use `judge="nemotron"` for NVIDIA's free vision model, or Claude). Frames are
file paths, so run retriEVAL locally (Cursor or Docker) for this step.

## Real Tello EDU (next)

```bash
pip install djitellopy opencv-python ultralytics
DRONE_DRIVER=tello DRONE_VERIFIER=ollama python -m drone_mcp.server
```

`tello.py` and `yolo.py` (YOLO-World, an open-vocabulary detector) implement the
same interfaces as the simulator but are **untested until they meet real
hardware**. The Tello has no GPS: position is dead reckoning from the commanded
moves, so keep the wall margin generous. Put the Tello EDU on your home Wi-Fi
(station mode) so this computer keeps its internet connection.

## Not built yet

- Railway relay: the server running on Railway with the laptop bridge dialling
  out to it over a WebSocket with signed commands.
- Distance sensing on the Tello: it has no rangefinder, so `approach` relies on
  an estimate from the camera and the wall margin.
- Jev for fusing detections into decisions, to be compared against plain rules
  on recorded flights.
