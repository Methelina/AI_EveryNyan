"""
MCP server providing system date and time tools with timezone support.

Tools:
  - get_datetime:      current date/time with timezone, formatted
  - get_timestamp:     current Unix timestamp and ISO 8601
  - format_datetime:   format a timestamp or "now" into a custom string
  - time_diff:         human-readable difference between two dates/times

/tools/mcp/tool_datetime.py

Version:     0.1.0
Author:      pytraveler
Created:     2026-05-05
"""

import sys
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastmcp import FastMCP

mcp = FastMCP("datetime")


@mcp.tool()
async def get_datetime(
    timezone_offset: Optional[float] = None,
    format_str: str = "",
) -> str:
    """
    Get the current date, time, and timezone information.

    Use this tool whenever you need to know the current time, date, day of week,
    or timezone. This is essential for greetings ("good morning/evening"),
    scheduling references, and any time-aware responses.

    Parameters:
    - timezone_offset: UTC offset in hours (e.g. 3.0 for UTC+3, -5.0 for UTC-5).
                       If omitted, uses the system local timezone.
    - format_str: Custom strftime format string. If empty, returns a full
                  multi-line report with date, time, weekday, timezone, etc.
                  Example: "%Y-%m-%d %H:%M:%S" → "2026-05-05 22:12:25"
    """
    try:
        if timezone_offset is not None:
            tz = timezone(timedelta(hours=timezone_offset))
            now = datetime.now(tz)
            tz_name = f"UTC{'+' if timezone_offset >= 0 else ''}{timezone_offset}"
        else:
            now = datetime.now().astimezone()
            tz_name = now.strftime("%Z") or "Local"
            offset_h = now.utcoffset()
            if offset_h is not None:
                total_sec = offset_h.total_seconds()
                tz_name = f"UTC{'+' if total_sec >= 0 else ''}{total_sec / 3600:g}"

        if format_str:
            return now.strftime(format_str)

        iso = now.isoformat()
        unix = now.timestamp()

        lines = [
            f"Date: {now.strftime('%Y-%m-%d')}",
            f"Time: {now.strftime('%H:%M:%S')}",
            f"Weekday: {now.strftime('%A')}",
            f"Timezone: {tz_name}",
            f"ISO 8601: {iso}",
            f"Unix timestamp: {unix:.0f}",
        ]
        return "\n".join(lines)

    except Exception as e:
        return f"Error getting datetime: {type(e).__name__}: {e}"


@mcp.tool()
async def get_timestamp() -> str:
    """
    Get the current Unix timestamp, ISO 8601 string, and basic date/time parts.

    Returns a compact summary: unix epoch, ISO 8601, date, time, timezone offset.

    Use this when you need a machine-readable timestamp or need to calculate
    time differences later.
    """
    try:
        now = datetime.now().astimezone()
        return (
            f"Unix: {now.timestamp():.0f}\n"
            f"ISO 8601: {now.isoformat()}\n"
            f"Date: {now.strftime('%Y-%m-%d')}\n"
            f"Time: {now.strftime('%H:%M:%S')}\n"
            f"TZ offset: {now.strftime('%z')}"
        )
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def format_datetime(
    timestamp: float = 0.0,
    format_str: str = "%Y-%m-%d %H:%M:%S",
    timezone_offset: Optional[float] = None,
) -> str:
    """
    Format a Unix timestamp into a human-readable date/time string.

    If timestamp is 0 or omitted, formats the current time.

    Parameters:
    - timestamp: Unix timestamp (seconds since epoch). 0 = now.
    - format_str: strftime format string. Default: "%Y-%m-%d %H:%M:%S".
                  Common codes: %Y=year, %m=month, %d=day, %H=hour(24),
                  %M=minute, %S=second, %A=weekday, %B=month name, %Z=timezone.
    - timezone_offset: UTC offset in hours. If omitted, uses system local timezone.
    """
    try:
        if timezone_offset is not None:
            tz = timezone(timedelta(hours=timezone_offset))
        else:
            tz = None

        if timestamp == 0.0:
            dt = datetime.now(tz)
        else:
            dt = datetime.fromtimestamp(timestamp, tz=tz)

        return dt.strftime(format_str)

    except Exception as e:
        return f"Error formatting datetime: {type(e).__name__}: {e}"


@mcp.tool()
async def time_diff(
    start: str,
    end: str = "",
) -> str:
    """
    Calculate the human-readable difference between two dates or times.

    Accepts ISO 8601 strings (e.g. "2026-05-05T22:00:00") or date-only
    strings (e.g. "2026-05-01"). If end is empty, uses the current time.

    Returns the difference in days, hours, minutes, seconds, and total seconds.

    Parameters:
    - start: Start date/time in ISO 8601 or "YYYY-MM-DD" format.
    - end: End date/time in same format. Empty string = now.
    """
    try:
        def _parse(s: str) -> datetime:
            s = s.strip()
            for fmt in (
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Cannot parse date/time: '{s}'")

        dt_start = _parse(start)
        if end:
            dt_end = _parse(end)
        else:
            dt_end = datetime.now()

        delta = dt_end - dt_start
        total_sec = delta.total_seconds()
        abs_sec = abs(total_sec)
        sign = "" if total_sec >= 0 else "-"

        days = int(abs_sec // 86400)
        hours = int((abs_sec % 86400) // 3600)
        minutes = int((abs_sec % 3600) // 60)
        seconds = int(abs_sec % 60)

        parts = []
        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if seconds > 0 or not parts:
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

        human = ", ".join(parts)
        if sign:
            human = f"-{human}"

        return (
            f"From: {dt_start.isoformat()}\n"
            f"To: {dt_end.isoformat()}\n"
            f"Duration: {human}\n"
            f"Total: {total_sec:.0f} seconds ({total_sec / 3600:.2f} hours)"
        )

    except Exception as e:
        return f"Error calculating time diff: {type(e).__name__}: {e}"


if __name__ == "__main__":
    print(f"[MCP] datetime: Starting MCP server (tool_datetime.py)", file=sys.stderr)
    print(f"[MCP] datetime: Ready to accept stdio MCP connections", file=sys.stderr)
    mcp.run(transport="stdio")
