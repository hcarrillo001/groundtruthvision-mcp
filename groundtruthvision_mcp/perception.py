"""Seeing things, in two tiers.

Tier 1, Detector: runs on every frame and is fast but easily fooled. It finds
  CANDIDATES: "there's a backpack about 30 degrees to the right". It knows
  object types, not details, so a black backpack and a blue one look the same.

Tier 2, Verifier: runs only on a candidate, while the drone hovers. A vision
  model looks at the frame and answers the actual question ("is this a BLUE
  backpack?") with a confidence. It is slower (about 1-3 s locally), which is
  fine because the drone is holding still.

Both are interfaces. The simulator uses SimDetector and SimVerifier, which read
the simulated world. On real flights use a real detector, and OllamaVerifier
(a vision model running on your own computer: free, private, no rate limits).
"""
from __future__ import annotations

import base64
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import List, Protocol


@dataclass
class Detection:
    label: str
    confidence: float
    angle: float          # degrees from straight ahead; negative = left
    distance_cm: float    # estimated
    frame: str            # path to the frame it was seen in
    ref: object = field(default=None, repr=False)   # sim only: the world object


@dataclass
class Verdict:
    match: bool
    confidence: float
    why: str
    seconds: float        # how long the check took (this is the hover time)
    model: str


class Detector(Protocol):
    def detect(self, drone, query: str) -> List[Detection]: ...


class Verifier(Protocol):
    def verify(self, frame: str, query: str, ref=None) -> Verdict: ...


def _words(text: str):
    return set(re.findall(r"[a-z]+", text.lower()))


# ---------------------------------------------------------------------------
# Simulator implementations
# ---------------------------------------------------------------------------

class SimDetector:
    """Plays the part of an object detector. It reports every visible object
    whose TYPE appears in the query, with confidence falling off with distance
    and a little noise, and it cannot tell a blue backpack from a black one.
    That is deliberate: it is what the verifier is for."""

    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)

    def detect(self, drone, query: str) -> List[Detection]:
        q = _words(query)
        out = []
        for obj, angle, dist in drone.visible():
            if obj.label.lower() not in q:
                continue
            conf = max(0.05, min(0.99, 0.95 - dist / 900 + self.rng.uniform(-0.05, 0.05)))
            out.append(Detection(obj.label, round(conf, 2), round(angle, 1), round(dist),
                                 drone.frame_for(obj), obj))
        return sorted(out, key=lambda d: -d.confidence)


class SimVerifier:
    """Plays the part of a vision model: matches when every word of the query
    appears in the object's true description."""

    def __init__(self, seconds: float = 1.5):
        self.seconds = seconds

    def verify(self, frame: str, query: str, ref=None) -> Verdict:
        desc = getattr(ref, "description", "") if ref is not None else ""
        ok = bool(desc) and _words(query) <= _words(desc)
        return Verdict(ok, 0.92 if ok else 0.08,
                       f"the object is a {desc}" if desc else "nothing recognisable in view",
                       self.seconds, "sim")


# ---------------------------------------------------------------------------
# Real verifier: a local vision model through Ollama
# ---------------------------------------------------------------------------

_PROMPT = """Look at this image from a drone's camera.
Question: is there a {query} clearly visible?
Answer only if you can actually see it; if the image is dark, blurry, or the
object only partly matches (wrong colour, wrong kind), the answer is no.
Reply with JSON only: {{"match": true or false, "confidence": 0.0-1.0, "why": "one short sentence"}}"""


class OllamaVerifier:
    """Asks a vision model running locally in Ollama. Install Ollama, then e.g.
    `ollama pull qwen2.5vl:7b` (more accurate) or `ollama pull moondream`
    (faster), and set OLLAMA_VISION_MODEL to the model name."""

    def __init__(self, model: str | None = None, url: str | None = None, timeout: float = 30):
        self.model = model or os.environ.get("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
        self.url = url or os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
        self.timeout = timeout

    def verify(self, frame: str, query: str, ref=None) -> Verdict:
        import urllib.request
        with open(frame, "rb") as f:
            image = base64.b64encode(f.read()).decode("ascii")
        payload = {"model": self.model, "stream": False, "format": "json",
                   "messages": [{"role": "user", "content": _PROMPT.format(query=query),
                                 "images": [image]}],
                   "options": {"temperature": 0}}
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            content = json.loads(r.read())["message"]["content"]
        return _parse_verdict(content, time.perf_counter() - t0, self.model)


def _parse_verdict(content: str, seconds: float, model: str) -> Verdict:
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.S).strip()
    start, end = content.find("{"), content.rfind("}")
    try:
        d = json.loads(content[start:end + 1] if start != -1 else content)
    except json.JSONDecodeError:
        return Verdict(False, 0.0, f"model did not return JSON: {content[:80]}", seconds, model)
    conf = float(d.get("confidence", 0.0) or 0.0)
    return Verdict(bool(d.get("match")), max(0.0, min(1.0, conf)), str(d.get("why", "")),
                   seconds, model)


# ---------------------------------------------------------------------------
# Cloud verifier: any OpenAI-compatible vision model, NVIDIA Nemotron by default
# ---------------------------------------------------------------------------

class OpenAIVisionVerifier:
    """A cloud vision model through an OpenAI-compatible API. Defaults to NVIDIA
    Nemotron 3.5 Super VL on OpenRouter's free endpoint.

    Good for developing with test photos. For real flights over your home prefer
    OllamaVerifier: a cloud call adds network delay, free endpoints are rate
    limited (about 20 a minute) and may log what you send.

    Env: DRONE_VISION_BASE_URL (default https://openrouter.ai/api/v1),
         DRONE_VISION_API_KEY (falls back to OPENROUTER_API_KEY),
         DRONE_VISION_MODEL (default nvidia/nemotron-3.5-super-vl:free)."""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 key: str | None = None, timeout: float = 60):
        self.model = model or os.environ.get("DRONE_VISION_MODEL", "nvidia/nemotron-3.5-super-vl:free")
        self.base_url = (base_url or os.environ.get("DRONE_VISION_BASE_URL",
                                                     "https://openrouter.ai/api/v1")).rstrip("/")
        self.key = key or os.environ.get("DRONE_VISION_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
        self.timeout = timeout

    def verify(self, frame: str, query: str, ref=None) -> Verdict:
        import urllib.request
        with open(frame, "rb") as f:
            raw = f.read()
        mt = "image/png" if raw.startswith(b"\x89PNG") else "image/jpeg"
        payload = {"model": self.model, "max_tokens": 1024, "temperature": 0,
                   "messages": [{"role": "user", "content": [
                       {"type": "text", "text": _PROMPT.format(query=query)},
                       {"type": "image_url", "image_url": {
                           "url": f"data:{mt};base64,{base64.b64encode(raw).decode('ascii')}"}}]}]}
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}",
                     "User-Agent": "GroundTruthVisionMCP/0.1"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            content = json.loads(r.read())["choices"][0]["message"].get("content") or ""
        return _parse_verdict(content, time.perf_counter() - t0, self.model)
