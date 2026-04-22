import os
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient


def _discover_tool_files() -> list[Path]:
    return sorted(Path(__file__).parent.glob("tool_*.py"))


def return_mcp_client(**env_kwargs) -> MultiServerMCPClient:
    tool_files = _discover_tool_files()
    servers = {}
    for path in tool_files:
        name = path.stem
        servers[name] = {
            "transport": "stdio",
            "command": "python",
            "args": [str(path.resolve())],
            "env": {
                **os.environ,
                **env_kwargs,
            },
        }
    return MultiServerMCPClient(servers)
