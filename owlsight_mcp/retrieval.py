"""retriEVAL as owlsight's evaluator, live, during the flight.

owlsight flies; retriEVAL judges. While the drone hovers over a candidate,
owlsight asks retriEVAL to score a claim ("There is a blue backpack in view.")
against the camera frame with its visual_faithfulness metric. retriEVAL splits
the claim into parts ("a backpack", "it is blue") and checks each against the
image, so a black backpack scores 0.5 and is rejected. The same connection
checks Claude's mission report before it is accepted, against the frames and
the flight log's facts (distances, heights, actions) together.

The two talk MCP to MCP: owlsight starts retriEVAL as a local MCP server over
stdio and keeps one session open for the whole flight. Local, because the
frames are files on this computer and retriEVAL needs to read them.

Env:
  RETRIEVAL_SERVER   path to retriEVAL's server.py
                     (default ~/Workspace/retrieval-mcp/retrieval-mcp/server.py)
  RETRIEVAL_PYTHON   Python that has retriEVAL's requirements (default: this one)
  RETRIEVAL_JUDGE    retriEVAL's judge for these checks: "" (its default),
                     "nemotron" (NVIDIA, free via OpenRouter), or "llm"
  The judge's own keys (ANTHROPIC_API_KEY, OPENROUTER_API_KEY, ...) and
  RETRIEVAL_JUDGE_BACKEND are passed through from this environment.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path

from .perception import Verdict

DEFAULT_SERVER = "~/Workspace/retrieval-mcp/retrieval-mcp/server.py"


class RetrievalError(RuntimeError):
    pass


class RetrievalClient:
    """One long-lived MCP session to a local retriEVAL server.

    MCP's Python client is async; owlsight is not. A small background thread
    owns the event loop and the session, and call() hands work to it."""

    def __init__(self, command: str | None = None, args: list | None = None,
                 env: dict | None = None, start_timeout: float = 60):
        from mcp import StdioServerParameters
        server = Path(os.environ.get("RETRIEVAL_SERVER", DEFAULT_SERVER)).expanduser()
        if args is None:
            if not server.is_file():
                raise RetrievalError(
                    f"retriEVAL not found at {server}. Set RETRIEVAL_SERVER to the "
                    f"path of retriEVAL's server.py.")
            args = [str(server)]
        self.params = StdioServerParameters(
            command=command or os.environ.get("RETRIEVAL_PYTHON", sys.executable),
            args=args,
            env={**os.environ, **(env or {})},
            cwd=str(Path(args[0]).parent) if args and Path(args[0]).is_file() else None,
        )
        self._loop = asyncio.new_event_loop()
        self._ready: concurrent.futures.Future = concurrent.futures.Future()
        self._session = None
        self._stop = None
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._run(), self._loop)
        try:
            self._ready.result(timeout=start_timeout)
        except concurrent.futures.TimeoutError:
            raise RetrievalError("retriEVAL did not start in time.") from None

    async def _run(self):
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client
        self._stop = asyncio.Event()
        try:
            async with stdio_client(self.params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set_result(True)
                    await self._stop.wait()
        except BaseException as e:   # noqa: BLE001 - surface any startup failure
            if not self._ready.done():
                self._ready.set_exception(RetrievalError(f"could not start retriEVAL: {e}"))

    def call(self, tool: str, arguments: dict, timeout: float = 120) -> dict:
        fut = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(tool, arguments), self._loop)
        result = fut.result(timeout=timeout)
        text = "".join(getattr(c, "text", "") for c in result.content)
        if result.isError:
            raise RetrievalError(text.strip() or f"retriEVAL {tool} failed")
        if isinstance(result.structuredContent, dict) and result.structuredContent:
            data = result.structuredContent
            return data.get("result", data) if set(data) == {"result"} else data
        return json.loads(text)

    def close(self):
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._loop.call_soon_threadsafe(self._loop.stop)


def _claim_for(query: str) -> str:
    return f"There is a {query.strip()} in view."


class RetrievalVerifier:
    """The verifier owlsight uses during the hover: retriEVAL's
    visual_faithfulness on one frame. Confidence is retriEVAL's score."""

    def __init__(self, client: RetrievalClient | None = None, judge: str | None = None):
        self.client = client or RetrievalClient()
        self.judge = os.environ.get("RETRIEVAL_JUDGE", "") if judge is None else judge
        self.model = "retriEVAL" + (f"/{self.judge}" if self.judge else "")

    def _score(self, text: str, images: list, goal: str, save: bool,
               facts: list | None = None) -> dict:
        args = {"input": goal, "actual_output": text, "metrics": ["visual_faithfulness"],
                "images": images, "judge": self.judge, "save": save}
        if facts:
            args["context"] = facts
        out = self.client.call("evaluate_case", args)
        r = out["scores"]["visual_faithfulness"]
        v = (out.get("verdict") or {}).get("visual_faithfulness") or {}
        claims = [{k: c[k] for k in ("claim", "supported", "image", "source", "why")
                   if c.get(k) not in (None, "")}
                  for c in (r.get("details") or {}).get("claims", []) if isinstance(c, dict)]
        return {"score": r["score"], "passed": bool(r["success"]), "claims": claims,
                "unsupported": (r.get("details") or {}).get("unsupported_claims", []),
                "reason": r.get("reason", ""), "headline": v.get("headline", ""),
                "drivers": v.get("drivers", []), "report_url": out.get("report_url"),
                "view_url": out.get("view_url")}

    def verify(self, frame: str, query: str, ref=None) -> Verdict:
        t0 = time.perf_counter()
        s = self._score(_claim_for(query), [frame], f"Is there a {query} in view?", save=False)
        why = ("; ".join(s["unsupported"]) and f"not supported: {'; '.join(s['unsupported'])}") \
            or s["reason"] or s["headline"]
        return Verdict(s["passed"], s["score"], why, time.perf_counter() - t0, self.model)

    def check_claims(self, text: str, frames: list, goal: str, save: bool = True,
                     facts: list | None = None) -> dict:
        """Score free text (a mid-flight claim or the mission report) against the
        frames and, when given, the flight log's facts."""
        return self._score(text, frames, goal, save, facts)
