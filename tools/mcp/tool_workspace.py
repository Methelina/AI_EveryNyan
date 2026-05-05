"""
MCP server providing read-only filesystem access within a sandboxed workspace directory.
All operations are strictly confined to workspace_dir — path traversal and symlink escapes are blocked.
No write, delete, move, or modify operations are available.

Tools:
  - read_file:       read file content (text lines or binary bytes with offset/limit)
  - check_path:      check existence and type of a path
  - list_directory:  list directory contents with sorting and filtering
  - search_files:    search files by glob pattern
  - directory_tree:  show directory tree structure
  - file_info:       detailed file metadata (size, dates, line count, checksum)
  - grep_content:    search file contents by regex pattern

/tools/mcp/tool_workspace.py

Version:     0.1.0
Author:      pytraveler
Created:     2026-05-05

Patch Notes v0.1.0 (by pytraveler):
  [+] MCP server for read-only filesystem access within a sandboxed workspace directory.
  [+] Tools: read_file, check_path, list_directory, search_files, directory_tree, file_info, grep_content.
  [+] All paths resolved and validated against workspace_dir — path traversal and symlink escapes blocked.
  [+] No write/delete/move operations available; sandbox enforced at tool layer.
  [+] Configurable via settings.yaml (workspace section) with safe defaults (project root, read-only).
  [+] Debug logging to logs/mcp_debug.log (call params + result summary, no file content dumped).
"""

import os
import sys
import base64
import hashlib
import stat
import re
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastmcp import FastMCP

# ============================================================================
# PATH RESOLUTION
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"

# ============================================================================
# LOGGING
# ============================================================================
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_LOG = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/mcp_debug.log"))


def _log(msg: str):
    ts = datetime.now().isoformat()
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"{ts} {msg}\n")


# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULTS = {
    "workspace_dir": ".",
    "max_file_size": 10 * 1024 * 1024,
    "max_read_lines": 2000,
    "max_read_bytes": 1024 * 1024,
    "max_search_results": 200,
    "allow_hidden_files": False,
    "max_line_count_size": 1024 * 1024,
}


def _load_config() -> dict:
    cfg = {**DEFAULTS}
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            ws = data.get("workspace", {})
            if isinstance(ws, dict):
                for key in DEFAULTS:
                    if key in ws:
                        cfg[key] = ws[key]
    except Exception as e:
        _log(f"Config load failed: {e}, using defaults")

    p = Path(cfg["workspace_dir"])
    if not p.is_absolute():
        cfg["workspace_dir"] = str(REPO_ROOT / p)
    return cfg


_config = _load_config()
WORKSPACE_DIR = Path(_config["workspace_dir"])
MAX_FILE_SIZE: int = _config["max_file_size"]
MAX_READ_LINES: int = _config["max_read_lines"]
MAX_READ_BYTES: int = _config["max_read_bytes"]
MAX_SEARCH_RESULTS: int = _config["max_search_results"]
ALLOW_HIDDEN: bool = _config["allow_hidden_files"]
MAX_LINE_COUNT_SIZE: int = _config["max_line_count_size"]

_resolved_workspace = WORKSPACE_DIR.resolve()

# ============================================================================
# SANDBOX SECURITY
# ============================================================================


def _is_within_workspace(target: Path) -> bool:
    try:
        target.resolve().relative_to(_resolved_workspace)
        return True
    except (ValueError, OSError):
        return False


def _safe_resolve(rel_path: str) -> Path:
    if not rel_path or rel_path.strip() == ".":
        return _resolved_workspace
    cleaned = rel_path.replace("\\", "/").strip("/")
    target = (_resolved_workspace / cleaned).resolve()
    try:
        target.relative_to(_resolved_workspace)
    except ValueError:
        raise ValueError(f"Access denied: path '{rel_path}' escapes workspace")
    return target


# ============================================================================
# HELPERS
# ============================================================================


def _should_skip(name: str) -> bool:
    if not ALLOW_HIDDEN and name.startswith('.'):
        return True
    return False


def _count_lines(filepath: Path) -> Optional[int]:
    try:
        if filepath.stat().st_size > MAX_LINE_COUNT_SIZE:
            return None
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def _format_size(size: float) -> str:
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _format_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _is_text_file(filepath: Path) -> bool:
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(8192)
        return b'\x00' not in chunk
    except Exception:
        return False


# ============================================================================
# FastMCP instance
# ============================================================================
mcp = FastMCP("workspace")

WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MCP TOOLS
# ============================================================================


@mcp.tool()
async def read_file(
    path: str,
    offset: int = 0,
    limit: int = 0,
    binary: bool = False,
    byte_offset: int = 0,
    byte_limit: int = 0,
    encoding: str = "utf-8",
) -> str:
    """
    Read file contents from the sandboxed workspace directory. Text mode by default.

    Text mode: reads text lines with line numbers. Use 'offset' (0-based line number)
    and 'limit' (max lines to return).
    Binary mode (binary=True): reads raw bytes returned as base64. Use 'byte_offset'
    and 'byte_limit' to control the range.

    If limit or byte_limit is 0, reads up to the configured maximum.

    Parameters:
    - path: Relative path to the file within the workspace.
    - offset: Starting line number (0-based) in text mode. Default: 0.
    - limit: Maximum number of lines to read. 0 = up to max (2000).
    - binary: If true, read as binary bytes (base64 output) instead of text.
    - byte_offset: Starting byte position in binary mode. Default: 0.
    - byte_limit: Maximum bytes to read in binary mode. 0 = up to max (1 MB).
    - encoding: Text encoding for text mode. Default: 'utf-8'.
    """
    _log(f"=== read_file called === path={path}, binary={binary}, offset={offset}, limit={limit}, byte_offset={byte_offset}, byte_limit={byte_limit}, encoding={encoding}")

    try:
        target = _safe_resolve(path)
    except ValueError as e:
        _log(f"read_file ERROR: {e}")
        return f"Error: {e}"

    if not target.exists():
        _log(f"read_file ERROR: File not found: {path}")
        return f"Error: File not found: {path}"
    if not target.is_file():
        _log(f"read_file ERROR: Not a file: {path}")
        return f"Error: Not a file: {path}"

    file_size = target.stat().st_size
    if file_size > MAX_FILE_SIZE:
        _log(f"read_file ERROR: File too large ({_format_size(file_size)})")
        return f"Error: File too large ({_format_size(file_size)}). Maximum: {_format_size(MAX_FILE_SIZE)}."

    if binary:
        eff = byte_limit if byte_limit > 0 else MAX_READ_BYTES
        eff = min(eff, MAX_READ_BYTES)
        try:
            with open(target, 'rb') as f:
                if byte_offset > 0:
                    f.seek(byte_offset)
                data = f.read(eff)
        except Exception as e:
            _log(f"read_file binary ERROR: {e}")
            return f"Error reading binary: {e}"

        _log(f"read_file binary OK: {len(data)} bytes from offset {byte_offset}, total file {file_size} bytes")
        b64 = base64.b64encode(data).decode('ascii')
        return (
            f"Binary read: {len(data)} bytes from offset {byte_offset}\n"
            f"Total file size: {file_size} bytes\n"
            f"Base64:\n{b64}"
        )

    eff = limit if limit > 0 else MAX_READ_LINES
    eff = min(eff, MAX_READ_LINES)
    try:
        with open(target, 'r', encoding=encoding, errors='replace') as f:
            lines = f.readlines()
    except Exception as e:
        _log(f"read_file text ERROR: {e}")
        return f"Error reading text: {e}"

    total = len(lines)
    start = max(0, offset)
    end = min(total, start + eff)
    selected = lines[start:end]

    numbered = []
    for i, line in enumerate(selected, start=start + 1):
        numbered.append(f"{i:>6} | {line.rstrip()}")

    header = f"File: {path} | Lines {start + 1}-{end} of {total} | Size: {_format_size(file_size)}"
    if start + eff < total:
        header += f" | Use offset={start + eff} to continue"

    _log(f"read_file text OK: lines {start + 1}-{end}/{total}, size={_format_size(file_size)}")
    return header + "\n" + "\n".join(numbered)


@mcp.tool()
async def check_path(path: str) -> str:
    """
    Check if a file or directory exists in the workspace.
    Returns type (file/directory), size, modification time, and extension.
    Reports clearly when the path does not exist.

    Parameters:
    - path: Relative path within the workspace to check.
    """
    _log(f"=== check_path called === path={path}")

    try:
        target = _safe_resolve(path)
    except ValueError as e:
        _log(f"check_path ERROR: {e}")
        return f"Error: {e}"

    if not target.exists():
        _log(f"check_path: does not exist: {path}")
        return f"Path does not exist: {path}"

    st = target.stat()
    rel = str(target.relative_to(_resolved_workspace))

    if target.is_file():
        ptype = "file"
    elif target.is_dir():
        ptype = "directory"
    else:
        ptype = "other (symlink/socket/device)"

    lines = [
        f"Path: {rel}",
        f"Type: {ptype}",
        f"Size: {_format_size(st.st_size)}",
        f"Modified: {_format_time(st.st_mtime)}",
        f"Created: {_format_time(st.st_ctime)}",
    ]

    if target.is_file():
        ext = target.suffix.lower() if target.suffix else "(none)"
        lines.append(f"Extension: {ext}")
        if _is_text_file(target):
            lc = _count_lines(target)
            if lc is not None:
                lines.append(f"Lines: {lc}")
            lines.append(f"Content type: text")
        else:
            lines.append(f"Content type: binary")

    _log(f"check_path OK: {ptype} {rel}")
    return "\n".join(lines)


@mcp.tool()
async def list_directory(
    path: str = "",
    sort_by: str = "name",
    order: str = "asc",
    limit: int = 50,
    extensions: str = "",
    files_only: bool = False,
    dirs_only: bool = False,
) -> str:
    """
    List contents of a directory in the workspace with sorting and filtering.
    Returns name, type, size, modification date, extension, and line count for text files.

    Parameters:
    - path: Subdirectory within the workspace (empty = workspace root).
    - sort_by: "name", "date", "size", or "extension". Default: "name".
    - order: "asc" or "desc". Default: "asc".
    - limit: Maximum entries to return. 0 = all. Default: 50.
    - extensions: Comma-separated extension filter, e.g. "py,txt,json". Empty = all.
    - files_only: Only show files. Default: false.
    - dirs_only: Only show directories. Default: false.
    """
    _log(f"=== list_directory called === path='{path}', sort_by={sort_by}, order={order}, limit={limit}, extensions={extensions}, files_only={files_only}, dirs_only={dirs_only}")

    try:
        target = _safe_resolve(path)
    except ValueError as e:
        return f"Error: {e}"

    if not target.exists():
        return f"Error: Directory not found: {path}"
    if not target.is_dir():
        return f"Error: Not a directory: {path}"

    ext_filter = set()
    if extensions.strip():
        ext_filter = {
            e.strip().lstrip('.').lower()
            for e in extensions.split(',')
            if e.strip()
        }

    entries = []
    try:
        for entry in target.iterdir():
            if _should_skip(entry.name):
                continue

            is_file = entry.is_file()
            is_dir = entry.is_dir()

            if files_only and not is_file:
                continue
            if dirs_only and not is_dir:
                continue

            if ext_filter and is_file:
                ext = entry.suffix.lstrip('.').lower()
                if ext not in ext_filter:
                    continue

            try:
                st = entry.stat()
            except OSError:
                continue

            entries.append({
                "name": entry.name,
                "path": str(entry.relative_to(_resolved_workspace)),
                "is_file": is_file,
                "is_dir": is_dir,
                "size": st.st_size if is_file else 0,
                "mtime": st.st_mtime,
                "ext": entry.suffix.lower() if is_file else "",
            })
    except PermissionError:
        return f"Error: Permission denied accessing: {path}"

    reverse = (order == "desc")
    key_map = {
        "name": lambda e: e["name"].lower(),
        "date": lambda e: e["mtime"],
        "size": lambda e: e["size"],
        "extension": lambda e: e["ext"],
    }
    sort_key = key_map.get(sort_by, key_map["name"])
    entries.sort(key=sort_key, reverse=reverse)

    eff_limit = limit if limit > 0 else len(entries)
    total = len(entries)
    entries = entries[:eff_limit]

    if not entries:
        return f"Directory '{path or '/'}' is empty or no entries match filters."

    compute_lines = len(entries) <= 100

    rel_dir = str(target.relative_to(_resolved_workspace)) or "/"
    lines = [
        f"Contents of '{rel_dir}' ({total} total, showing {len(entries)}, sorted by {sort_by} {order}):",
        f"{'Type':<5} {'Size':>10} {'Modified':<20} {'Lines':>7} {'Name'}",
        "-" * 70,
    ]

    for e in entries:
        type_char = "F" if e["is_file"] else "D"
        size_str = _format_size(e["size"]) if e["is_file"] else "<dir>"
        mod_str = _format_time(e["mtime"])

        lc_str = ""
        if e["is_file"] and compute_lines:
            full_path = _resolved_workspace / e["path"]
            if _is_text_file(full_path):
                lc = _count_lines(full_path)
                lc_str = str(lc) if lc is not None else "?"
            else:
                lc_str = "bin"

        lines.append(f"{type_char:<5} {size_str:>10} {mod_str:<20} {lc_str:>7} {e['name']}")

    _log(f"list_directory OK: {len(entries)}/{total} entries in '{path or '/'}'")
    return "\n".join(lines)


@mcp.tool()
async def search_files(
    pattern: str,
    path: str = "",
    max_depth: int = 0,
    limit: int = 100,
) -> str:
    """
    Search for files matching a glob pattern in the workspace.
    Returns matched paths with size, modification date, and line count.

    Parameters:
    - pattern: Glob pattern, e.g. "*.py", "**/*.json", "report_*.csv".
    - path: Subdirectory to search in (empty = workspace root).
    - max_depth: Maximum recursion depth. 0 = unlimited. Default: 0.
    - limit: Maximum results. Default: 100.
    """
    _log(f"=== search_files called === pattern={pattern}, path='{path}', max_depth={max_depth}, limit={limit}")

    try:
        target = _safe_resolve(path)
    except ValueError as e:
        _log(f"search_files ERROR: {e}")
        return f"Error: {e}"

    if not target.exists():
        _log(f"search_files ERROR: Directory not found: {path}")
        return f"Error: Directory not found: {path}"
    if not target.is_dir():
        _log(f"search_files ERROR: Not a directory: {path}")
        return f"Error: Not a directory: {path}"

    try:
        matches = list(target.glob(pattern))
    except Exception as e:
        _log(f"search_files ERROR: Invalid glob '{pattern}': {e}")
        return f"Error: Invalid glob pattern '{pattern}': {e}"

    filtered = []
    for m in matches:
        if not _is_within_workspace(m):
            continue
        if _should_skip(m.name):
            continue
        filtered.append(m)

    if max_depth > 0:
        root_depth = len(target.resolve().parts)
        filtered = [
            m for m in filtered
            if len(m.resolve().parts) - root_depth <= max_depth
        ]

    filtered.sort(key=lambda p: str(p.relative_to(_resolved_workspace)).lower())
    filtered = filtered[:limit]

    if not filtered:
        _log(f"search_files: no matches for '{pattern}' in '{path or '/'}'")
        return f"No files matching '{pattern}' found in '{path or '/'}'."

    lines = [
        f"Search results for '{pattern}' in '{path or '/'}' ({len(filtered)} matches):",
        f"{'Size':>10} {'Modified':<20} {'Lines':>7} {'Path'}",
        "-" * 80,
    ]

    for m in filtered:
        rel = str(m.relative_to(_resolved_workspace))
        if m.is_file():
            st = m.stat()
            size_str = _format_size(st.st_size)
            mod_str = _format_time(st.st_mtime)
            lc_str = ""
            if _is_text_file(m):
                lc = _count_lines(m)
                lc_str = str(lc) if lc is not None else "?"
            else:
                lc_str = "bin"
            lines.append(f"{size_str:>10} {mod_str:<20} {lc_str:>7} {rel}")
        elif m.is_dir():
            lines.append(f"{'<dir>':>10} {'':20} {'':>7} {rel}/")

    _log(f"search_files OK: {len(filtered)} matches for '{pattern}' in '{path or '/'}'")
    return "\n".join(lines)


@mcp.tool()
async def directory_tree(
    path: str = "",
    max_depth: int = 3,
    limit: int = 200,
) -> str:
    """
    Display a tree visualization of a directory in the workspace.

    Parameters:
    - path: Subdirectory (empty = workspace root).
    - max_depth: Maximum depth to traverse. Default: 3.
    - limit: Maximum entries to display. Default: 200.
    """
    _log(f"=== directory_tree called === path='{path}', max_depth={max_depth}, limit={limit}")

    try:
        target = _safe_resolve(path)
    except ValueError as e:
        _log(f"directory_tree ERROR: {e}")
        return f"Error: {e}"

    if not target.exists():
        _log(f"directory_tree ERROR: Directory not found: {path}")
        return f"Error: Directory not found: {path}"
    if not target.is_dir():
        _log(f"directory_tree ERROR: Not a directory: {path}")
        return f"Error: Not a directory: {path}"

    rel_root = str(target.relative_to(_resolved_workspace)) or "workspace"
    output_lines = [f"{rel_root}/"]
    count = 0

    def _walk(directory: Path, prefix: str, depth: int):
        nonlocal count
        if depth > max_depth or count >= limit:
            return

        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda e: (not e.is_dir(), e.name.lower()),
            )
        except PermissionError:
            output_lines.append(f"{prefix}[permission denied]")
            return

        visible = [e for e in entries if not _should_skip(e.name)]

        for i, entry in enumerate(visible):
            if count >= limit:
                output_lines.append(f"{prefix}... (truncated, limit reached)")
                return

            is_last = (i == len(visible) - 1)
            connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "

            if entry.is_dir():
                output_lines.append(f"{prefix}{connector}{entry.name}/")
                count += 1
                ext = "    " if is_last else "\u2502   "
                _walk(entry, prefix + ext, depth + 1)
            else:
                try:
                    size = entry.stat().st_size
                    size_str = _format_size(size)
                except OSError:
                    size_str = "?"
                output_lines.append(f"{prefix}{connector}{entry.name}  ({size_str})")
                count += 1

    _walk(target, "", 0)

    if count >= limit:
        output_lines.append(f"\n(Showing {limit} entries, more exist)")

    _log(f"directory_tree OK: {count} entries in '{path or '/'}'")
    return "\n".join(output_lines)


@mcp.tool()
async def file_info(path: str) -> str:
    """
    Get detailed metadata about a file in the workspace.
    Returns size, dates, extension, content type, line count, encoding, and SHA-256 checksum.

    Parameters:
    - path: Relative path to the file within the workspace.
    """
    _log(f"=== file_info called === path={path}")

    try:
        target = _safe_resolve(path)
    except ValueError as e:
        _log(f"file_info ERROR: {e}")
        return f"Error: {e}"

    if not target.exists():
        _log(f"file_info ERROR: File not found: {path}")
        return f"Error: File not found: {path}"
    if not target.is_file():
        _log(f"file_info ERROR: Not a file: {path}")
        return f"Error: Not a file: {path}"

    st = target.stat()
    rel = str(target.relative_to(_resolved_workspace))
    is_text = _is_text_file(target)

    lines = [
        f"Path: {rel}",
        f"Name: {target.name}",
        f"Extension: {target.suffix.lower() or '(none)'}",
        f"Size: {_format_size(st.st_size)} ({st.st_size:,} bytes)",
        f"Modified: {_format_time(st.st_mtime)}",
        f"Created: {_format_time(st.st_ctime)}",
        f"Content type: {'text' if is_text else 'binary'}",
    ]

    if is_text:
        lc = _count_lines(target)
        if lc is not None:
            lines.append(f"Line count: {lc:,}")

        try:
            with open(target, 'rb') as f:
                raw = f.read(min(4096, st.st_size))
            if raw.startswith(b'\xef\xbb\xbf'):
                lines.append("Encoding: UTF-8 (BOM)")
            elif raw.startswith(b'\xff\xfe'):
                lines.append("Encoding: UTF-16 LE (BOM)")
            elif raw.startswith(b'\xfe\xff'):
                lines.append("Encoding: UTF-16 BE (BOM)")
            else:
                try:
                    raw.decode('utf-8')
                    lines.append("Encoding: UTF-8 (no BOM)")
                except UnicodeDecodeError:
                    lines.append("Encoding: non-UTF-8")
        except Exception:
            pass

    if st.st_size <= MAX_FILE_SIZE:
        try:
            h = hashlib.sha256()
            with open(target, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
            lines.append(f"SHA-256: {h.hexdigest()}")
        except Exception:
            pass

    lines.append(f"Permissions: {stat.filemode(st.st_mode)}")

    _log(f"file_info OK: {rel}, size={_format_size(st.st_size)}, {'text' if is_text else 'binary'}")
    return "\n".join(lines)


@mcp.tool()
async def grep_content(
    pattern: str,
    path: str = "",
    file_pattern: str = "*",
    max_results: int = 50,
    case_sensitive: bool = True,
    context_lines: int = 0,
) -> str:
    """
    Search file contents for a regex pattern within the workspace.
    Returns matching lines with file path, line number, and optional context.

    Parameters:
    - pattern: Regular expression to search for.
    - path: Subdirectory or file to search in (empty = workspace root).
    - file_pattern: Glob filter for files to search, e.g. "*.py". Default: "*" (all text files).
    - max_results: Maximum matching lines to return. Default: 50.
    - case_sensitive: Case-sensitive search. Default: true.
    - context_lines: Lines of context around each match. Default: 0.
    """
    _log(f"=== grep_content called === pattern={pattern}, path='{path}', file_pattern={file_pattern}, max_results={max_results}, case_sensitive={case_sensitive}, context_lines={context_lines}")

    try:
        target = _safe_resolve(path)
    except ValueError as e:
        _log(f"grep_content ERROR: {e}")
        return f"Error: {e}"

    if not target.exists():
        _log(f"grep_content ERROR: Path not found: {path}")
        return f"Error: Path not found: {path}"

    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
    except re.error as e:
        _log(f"grep_content ERROR: Invalid regex: {e}")
        return f"Error: Invalid regex: {e}"

    files_to_search = []
    if target.is_file():
        if _is_text_file(target) and _is_within_workspace(target):
            files_to_search.append(target)
    elif target.is_dir():
        for match in target.rglob(file_pattern):
            if not _is_within_workspace(match):
                continue
            if _should_skip(match.name):
                continue
            if match.is_file() and _is_text_file(match):
                try:
                    if match.stat().st_size > MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue
                files_to_search.append(match)

    _log(f"grep_content: searching {len(files_to_search)} file(s) for '{pattern}'")

    results = []
    total = 0

    for filepath in files_to_search:
        if total >= max_results:
            break
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                file_lines = f.readlines()
        except Exception:
            continue

        rel = str(filepath.relative_to(_resolved_workspace))

        for line_idx, line in enumerate(file_lines):
            if regex.search(line):
                total += 1
                if total > max_results:
                    break

                if context_lines > 0:
                    start = max(0, line_idx - context_lines)
                    end = min(len(file_lines), line_idx + context_lines + 1)
                    block = []
                    for i in range(start, end):
                        marker = ">>>" if i == line_idx else "   "
                        block.append(f"  {marker} {i + 1:>6} | {file_lines[i].rstrip()}")
                    results.append(f"\n{rel}:{line_idx + 1}:\n" + "\n".join(block))
                else:
                    results.append(f"{rel}:{line_idx + 1}: {line.rstrip()}")

    if not results:
        _log(f"grep_content: no matches for '{pattern}' in '{path or '/'}'")
        return f"No matches for '{pattern}' in '{path or '/'}'."

    header = f"Found {total} match(es) for '{pattern}':"
    if total > max_results:
        header += f" (showing first {max_results})"

    _log(f"grep_content OK: {total} match(es) for '{pattern}' in '{path or '/'}', searched {len(files_to_search)} file(s)")
    return header + "\n" + "\n".join(results)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    print(f"[MCP] workspace: Starting MCP server (tool_workspace.py)", file=sys.stderr)
    print(f"[MCP] workspace: Workspace dir: {WORKSPACE_DIR}", file=sys.stderr)
    print(f"[MCP] workspace: Max file size: {_format_size(MAX_FILE_SIZE)}", file=sys.stderr)
    print(f"[MCP] workspace: Max read lines: {MAX_READ_LINES}", file=sys.stderr)
    print(f"[MCP] workspace: Max read bytes: {_format_size(MAX_READ_BYTES)}", file=sys.stderr)
    print(f"[MCP] workspace: Allow hidden: {ALLOW_HIDDEN}", file=sys.stderr)
    print(f"[MCP] workspace: Ready to accept stdio MCP connections", file=sys.stderr)
    mcp.run(transport="stdio")
