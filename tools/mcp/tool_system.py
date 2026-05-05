"""
MCP server providing system information tools.

Tools:
  - get_system_info: OS, hostname, uptime, Python version, CPU, RAM, disk usage

/tools/mcp/tool_system.py

Version:     0.1.0
Author:      pytraveler
Created:     2026-05-05
"""

import os
import sys
import platform
import socket
from datetime import datetime, timedelta
from typing import Optional

from fastmcp import FastMCP

mcp = FastMCP("system")


def _get_uptime() -> str:
    try:
        if platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            ms = kernel32.GetTickCount64()
            return str(timedelta(milliseconds=ms)).split(".")[0]
        else:
            with open("/proc/uptime", "r") as f:
                secs = float(f.read().split()[0])
            return str(timedelta(seconds=int(secs)))
    except Exception:
        return "unknown"


def _get_cpu_info() -> str:
    try:
        if platform.system() == "Windows":
            import subprocess
            result = subprocess.run(
                ["wmic", "cpu", "get", "Name"],
                capture_output=True, text=True, timeout=10,
            )
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            if len(lines) > 1:
                return lines[1]
        elif platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout.strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return f"{platform.processor() or 'unknown'}"


def _get_cpu_count() -> str:
    try:
        physical = os.cpu_count() or 0
        return str(physical)
    except Exception:
        return "?"


def _get_ram_info() -> dict:
    try:
        if platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_gb = stat.ullTotalPhys / (1024 ** 3)
            avail_gb = stat.ullAvailPhys / (1024 ** 3)
            used_pct = stat.dwMemoryLoad
            return {"total_gb": total_gb, "avail_gb": avail_gb, "used_pct": used_pct}
        else:
            with open("/proc/meminfo", "r") as f:
                info = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        info[parts[0].rstrip(":")] = int(parts[1])
            total_kb = info.get("MemTotal", 0)
            avail_kb = info.get("MemAvailable", info.get("MemFree", 0))
            total_gb = total_kb / (1024 ** 2)
            avail_gb = avail_kb / (1024 ** 2)
            used_pct = ((total_kb - avail_kb) / total_kb * 100) if total_kb else 0
            return {"total_gb": total_gb, "avail_gb": avail_gb, "used_pct": used_pct}
    except Exception:
        return {"total_gb": 0, "avail_gb": 0, "used_pct": 0}


def _get_disk_info() -> list[dict]:
    try:
        import shutil
        disks = []
        if platform.system() == "Windows":
            import string
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    try:
                        usage = shutil.disk_usage(drive)
                        disks.append({
                            "path": drive,
                            "total_gb": usage.total / (1024 ** 3),
                            "used_gb": usage.used / (1024 ** 3),
                            "free_gb": usage.free / (1024 ** 3),
                            "used_pct": usage.used / usage.total * 100 if usage.total else 0,
                        })
                    except (PermissionError, OSError):
                        pass
        else:
            usage = shutil.disk_usage("/")
            disks.append({
                "path": "/",
                "total_gb": usage.total / (1024 ** 3),
                "used_gb": usage.used / (1024 ** 3),
                "free_gb": usage.free / (1024 ** 3),
                "used_pct": usage.used / usage.total * 100 if usage.total else 0,
            })
        return disks
    except Exception:
        return []


def _fmt_gb(val: float) -> str:
    return f"{val:.1f} GB"


@mcp.tool()
async def get_system_info(
    sections: str = "",
) -> str:
    """
    Get information about the operating system and hardware.

    Returns OS details, hostname, uptime, CPU, RAM, and disk usage.
    This helps understand the environment the application is running in.

    Use this when you need to know about the user's system — OS version,
    available memory, disk space, CPU model, etc.

    Parameters:
    - sections: Comma-separated filter for which sections to include.
                Available: "os", "cpu", "ram", "disk", "uptime", "python".
                Empty string = all sections. Example: "os,ram"
    """
    try:
        section_filter = set()
        if sections.strip():
            section_filter = {s.strip().lower() for s in sections.split(",") if s.strip()}

        lines = []
        show_all = not section_filter

        if show_all or "os" in section_filter:
            uname = platform.uname()
            lines.append("=== Operating System ===")
            lines.append(f"System: {uname.system}")
            lines.append(f"Release: {uname.release}")
            lines.append(f"Version: {uname.version}")
            lines.append(f"Machine: {uname.machine}")
            lines.append(f"Hostname: {socket.gethostname()}")
            if uname.system == "Windows":
                lines.append(f"Edition: {platform.win32_edition() if hasattr(platform, 'win32_edition') else 'N/A'}")
            lines.append("")

        if show_all or "python" in section_filter:
            lines.append("=== Python ===")
            lines.append(f"Version: {platform.python_version()}")
            lines.append(f"Implementation: {platform.python_implementation()}")
            lines.append(f"Executable: {sys.executable}")
            lines.append("")

        if show_all or "uptime" in section_filter:
            lines.append("=== Uptime ===")
            lines.append(f"System uptime: {_get_uptime()}")
            lines.append(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            tz_name = datetime.now().astimezone().strftime("%Z") or "Local"
            lines.append(f"Timezone: {tz_name}")
            lines.append("")

        if show_all or "cpu" in section_filter:
            lines.append("=== CPU ===")
            lines.append(f"Model: {_get_cpu_info()}")
            lines.append(f"Logical cores: {_get_cpu_count()}")
            try:
                load = os.getloadavg()
                lines.append(f"Load average (1/5/15 min): {load[0]:.2f} / {load[1]:.2f} / {load[2]:.2f}")
            except (AttributeError, OSError):
                if platform.system() == "Windows":
                    try:
                        import subprocess
                        r = subprocess.run(
                            ["wmic", "cpu", "get", "LoadPercentage"],
                            capture_output=True, text=True, timeout=10,
                        )
                        vals = [l.strip() for l in r.stdout.strip().split("\n") if l.strip() and l.strip() != "LoadPercentage"]
                        if vals:
                            lines.append(f"CPU usage: {vals[0]}%")
                    except Exception:
                        pass
            lines.append("")

        if show_all or "ram" in section_filter:
            ram = _get_ram_info()
            lines.append("=== Memory ===")
            lines.append(f"Total: {_fmt_gb(ram['total_gb'])}")
            lines.append(f"Available: {_fmt_gb(ram['avail_gb'])}")
            lines.append(f"Used: {ram['used_pct']:.0f}%")
            lines.append("")

        if show_all or "disk" in section_filter:
            disks = _get_disk_info()
            if disks:
                lines.append("=== Disk Usage ===")
                for d in disks:
                    lines.append(
                        f"  {d['path']}  "
                        f"{_fmt_gb(d['used_gb'])} / {_fmt_gb(d['total_gb'])}  "
                        f"({d['used_pct']:.0f}% used, {_fmt_gb(d['free_gb'])} free)"
                    )
                lines.append("")

        if not lines:
            return "No system information sections matched."

        return "\n".join(lines).rstrip()

    except Exception as e:
        return f"Error getting system info: {type(e).__name__}: {e}"


if __name__ == "__main__":
    print(f"[MCP] system: Starting MCP server (tool_system.py)", file=sys.stderr)
    print(f"[MCP] system: Ready to accept stdio MCP connections", file=sys.stderr)
    mcp.run(transport="stdio")
