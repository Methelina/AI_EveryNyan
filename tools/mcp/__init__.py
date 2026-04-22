import os
import sys
import logging
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

def _discover_tool_files() -> list[Path]:
    files = sorted(Path(__file__).parent.glob("tool_*.py"))
    if files:
        logger.info(f"[MCP] Discovered tool files: {[f.name for f in files]}")
    else:
        logger.warning("[MCP] No tool_*.py files found in tools directory")
    return files

def return_mcp_client(**env_kwargs) -> MultiServerMCPClient:
    tool_files = _discover_tool_files()
    servers = {}
    
    for path in tool_files:
        name = path.stem  # например, "tool_searxng"
        
        try:
            if path.stat().st_size < 10:
                logger.warning(f"[MCP] Tool file {path} is too small (size {path.stat().st_size} bytes), skipping")
                continue
        except OSError as e:
            logger.warning(f"[MCP] Cannot stat {path}: {e}, skipping")
            continue
        
        servers[name] = {
            "transport": "stdio",
            "command": sys.executable,   
            "args": [str(path.resolve())],
            "env": {
                **os.environ,
                **env_kwargs,
            },
        }
        logger.info(f"[MCP] Configured server '{name}' command: {sys.executable}")
    
    if not servers:
        logger.error("[MCP] No valid tool servers configured. MCP will be unavailable.")
    else:
        logger.info(f"[MCP] Total servers configured: {len(servers)}")
    
    return MultiServerMCPClient(servers)