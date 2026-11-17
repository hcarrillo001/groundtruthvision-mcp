"""A stand-in retriEVAL MCP server for tests: same evaluate_case tool and result
shape, no judge. A claim's colour words must appear in the image file names, and
its numbers in the recorded facts (context)."""
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("retrieval-mcp")
COLOURS = {"blue", "black", "red", "grey", "green"}


@mcp.tool()
def evaluate_case(input: str, actual_output: str, metrics: list, images: list = None,
                  judge: str = "", save: bool = True, context: list = None) -> dict:
    names = " ".join(Path(i["src"] if isinstance(i, dict) else i).stem for i in images or [])
    claims, bad = [], []
    for colour in COLOURS & set(re.findall(r"[a-z]+", actual_output.lower())):
        ok = colour in names
        claims.append(ok)
        if not ok:
            bad.append(f"It is {colour}.")
    facts = " ".join(context or [])
    for num in re.findall(r"\b\d+\b", actual_output):
        ok = re.search(rf"\b{num}\b", facts) is not None
        claims.append(ok)
        if not ok:
            bad.append(f"The number {num} is not in the flight log.")
    claims.append(True)                                  # "there is an object"
    score = sum(claims) / len(claims)
    out = {"scores": {"visual_faithfulness": {
        "score": score, "success": score >= 0.7, "reason": f"judge={judge or 'default'}",
        "details": {"unsupported_claims": bad, "facts": len(context or [])}}},
        "verdict": {"visual_faithfulness": {"headline": "ok" if score >= 0.7 else "fails",
                                            "drivers": bad}}}
    if save:
        out["report_url"] = "https://example.test/report/r-1"
    return out


if __name__ == "__main__":
    mcp.run()
