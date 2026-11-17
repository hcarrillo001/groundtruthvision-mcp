"""Start the MCP server from any working directory (handy for Cursor and Claude
Desktop configs, which may not run from this folder):

    python /path/to/owlsight-mcp/run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from owlsight_mcp.server import main  # noqa: E402

main()
