# owlsightMCP

*An owl doesn't swoop on the first shape it sees. It watches, makes sure, then commits.*

Fly a small drone from Claude, Cursor or any MCP client. **owlsight flies;
[retriEVAL](https://github.com/hcarrillo001/retrieval-mcp) evaluates, live.**
A safety gate the AI can't talk its way past keeps the flight safe; retriEVAL
decides what the camera really shows before the drone acts on it or reports it.

It runs in a **simulator today** and is built to swap in a **Tello EDU**.

## The idea

> "Find the blue backpack."

1. The drone takes off and turns on the spot, looking.
2. A fast detector spots candidates on every frame. It knows *backpack*, not
   *blue*, so a black backpack fools it.
3. For each candidate the drone **hovers** while **retriEVAL** scores the claim
   "There is a blue backpack in view" against the frame with its
   `visual_faithfulness` metric. It splits the claim into parts ("a backpack",
   "it is blue") and checks each: the black backpack scores 0.5 and is
   rejected, the blue one scores 1.0 and is confirmed.
4. It turns to face the target and flies to about a metre away. Mid-flight,
   `check_claim` lets Claude have retriEVAL check anything it is about to say
   ("the backpack is on the couch") against what the camera sees now.
5. After landing, `report_mission` sends Claude's report to retriEVAL with two
   kinds of evidence: the frames the camera captured, and the flight log as
   plain sentences. "Found the blue backpack" passes because a frame shows it;
   "flew 262 cm" passes because the log records it. Each claim says which backed
   it. Anything neither supports comes back `accepted: false` with the claims to fix.

## Layers

```
Claude ──MCP──> server.py     the tools Claude calls
                mission.py    runs every action through the steps below, keeps the log
                safety.py     SafetyGate: refuses unsafe commands before they fly
                perception.py Detector: fast candidates on every frame
                retrieval.py  ──MCP──> retriEVAL: scores candidates, claims, the report
                drone.py      SimDrone, or tello.py for the real Tello EDU
```

owlsight talks to retriEVAL MCP to MCP: it starts retriEVAL as a local server
and keeps one session open for the flight. Local, because the frames are files
on this computer.

The rule: **Claude decides where to go, the safety gate decides whether it's
allowed, and the gate always wins.** Refusals come back as plain sentences
Claude can act on ("that move would leave the flight box"), and nothing is flown.

## Tools

| Tool | What it does |
|---|---|
| `start_mission(goal)` | New mission and a fresh flight log; says which evaluator is in use |
| `takeoff` / `land` | Take off to ~80 cm; land |
| `move(direction, cm)` | forward, back, left, right, up, down; 20-200 cm |
| `rotate(degrees)` | Positive turns right, negative left |
| `look(target)` | What the detector sees right now (candidates only) |
| `scan_for(target)` | Turn and look; hover while retriEVAL checks each candidate; stop on a confirmed match |
| `approach(target)` | Fly toward a confirmed target, stop ~1 m away |
| `check_claim(claim)` | retriEVAL checks a claim against what the camera sees now |
| `status` | Position, height, heading, battery, clock |
| `emergency_stop` | Cut the motors. On a real drone it drops |
| `report_mission(summary)` | retriEVAL checks the report against the mission's frames; save the log |
| `list_flights()` | Saved missions, newest first: goal, what was confirmed, look-alikes rejected, report verdict |
| `get_flight_log(mission_id, show_frames)` | Open a saved flight: every retriEVAL check with its score and reason; `show_frames` returns the camera frames as images |

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
python -m pytest -q                 # 33 tests, no drone, network or API key needed
python run.py                       # MCP over stdio
```

Cursor (`~/.cursor/mcp.json`) or Claude Desktop:

```json
{
  "mcpServers": {
    "owlsight": {
      "command": "/Users/hcarrillo/Workspace/owlsight-mcp/.venv/bin/python",
      "args": ["/Users/hcarrillo/Workspace/owlsight-mcp/run.py"],
      "env": {
        "RETRIEVAL_SERVER": "/Users/hcarrillo/Workspace/retrieval-mcp/retrieval-mcp/server.py",
        "RETRIEVAL_PYTHON": "/Users/hcarrillo/Workspace/retrieval-mcp/.venv/bin/python",
        "RETRIEVAL_JUDGE": "nemotron",
        "OPENROUTER_API_KEY": "sk-or-..."
      }
    }
  }
}
```

Then ask: *"Start a mission to find the blue backpack, take off, scan for it,
approach it, land, and report what you found."*

The simulated room is `sim/worlds/living_room.json`. Its camera shows the
pictures in `sim/photos`; replace them with photos of your own room and
retriEVAL judges real images.

## The evaluator: retriEVAL

retriEVAL is the evaluator whenever owlsight can find it. Point owlsight at
your retriEVAL checkout and give retriEVAL a judge that can see images:

| Variable | Meaning |
|---|---|
| `RETRIEVAL_SERVER` | Path to retriEVAL's `server.py` (default `~/Workspace/retrieval-mcp/retrieval-mcp/server.py`) |
| `RETRIEVAL_PYTHON` | A Python with retriEVAL's requirements installed (default: owlsight's own) |
| `RETRIEVAL_JUDGE` | `nemotron` for NVIDIA via OpenRouter; empty for retriEVAL's default judge |

Judges for these checks:

| Judge | Setup | When |
|---|---|---|
| NVIDIA Nemotron | `RETRIEVAL_JUDGE=nemotron` + `OPENROUTER_API_KEY` | Developing: free, rate limited (~20/min), may log images |
| Claude | `ANTHROPIC_API_KEY`, leave `RETRIEVAL_JUDGE` empty | Accurate; paid |
| Local vision model | `RETRIEVAL_JUDGE_BACKEND=ollama`, `RETRIEVAL_JUDGE_MODEL=qwen2.5vl:7b` | Real flights over your home: free, private, no rate limits |

Each check's time is recorded as `hover_s` in the flight log, so you can
measure your setup before flying.

`DRONE_VERIFIER` overrides the choice: `retrieval` (the default when retriEVAL
is found), `sim` (the simulator's stand-in, no keys needed), or `ollama` /
`openai` to call a vision model directly without retriEVAL.

## Real Tello EDU (next)

```bash
pip install djitellopy opencv-python ultralytics
python tello_join_wifi.py "Home WiFi" "password"   # once, while joined to the Tello's own Wi-Fi
DRONE_DRIVER=tello TELLO_HOST=192.168.1.57 python run.py   # the address your router gave the Tello
```

The Tello and this computer must be on the same 2.4 GHz network, not a guest
network (guest networks usually stop devices seeing each other). Reserve the
Tello's address in your router so `TELLO_HOST` stays right. The computer's
firewall must let UDP ports 8889, 8890 and 11111 in (commands, state, video).

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
- Showing the report's frames on retriEVAL's report page (it lists the claims
  and verdicts today, not the images).
