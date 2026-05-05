"""
MCP server providing weather information via wttr.in API (no API key required).

Tools:
  - get_weather: current weather for a location
  - get_forecast: 3-day weather forecast for a location

/tools/mcp/tool_weather.py

Version:     0.1.0
Author:      pytraveler
Created:     2026-05-05
"""

import sys
import httpx
from typing import Optional

from fastmcp import FastMCP

mcp = FastMCP("weather")

_WTTR_BASE = "https://wttr.in"
_TIMEOUT = 15.0


def _clean_wttr_json(data: dict, location: str) -> str:
    try:
        current = data.get("current_condition", [{}])[0]
        area = data.get("nearest_area", [{}])[0]

        city = area.get("areaName", [{}])[0].get("value", location)
        country = area.get("country", [{}])[0].get("value", "")
        region = area.get("region", [{}])[0].get("value", "")

        temp_c = current.get("temp_C", "?")
        temp_f = current.get("temp_F", "?")
        feels_c = current.get("FeelsLikeC", "?")
        feels_f = current.get("FeelsLikeF", "?")
        humidity = current.get("humidity", "?")
        wind_speed = current.get("windspeedKmph", "?")
        wind_dir = current.get("winddir16Point", "?")
        pressure = current.get("pressure", "?")
        visibility = current.get("visibility", "?")
        uv = current.get("uvIndex", "?")
        cloud_cover = current.get("cloudcover", "?")
        precip = current.get("precipMM", "0")
        desc_list = current.get("weatherDesc", [{}])
        desc = desc_list[0].get("value", "N/A") if desc_list else "N/A"
        obs_time = current.get("observation_time", "?")

        loc_parts = [p for p in [city, region, country] if p]
        loc_str = ", ".join(loc_parts)

        lines = [
            f"Location: {loc_str}",
            f"Condition: {desc}",
            f"Temperature: {temp_c} C ({temp_f} F)",
            f"Feels like: {feels_c} C ({feels_f} F)",
            f"Humidity: {humidity}%",
            f"Wind: {wind_speed} km/h {wind_dir}",
            f"Pressure: {pressure} hPa",
            f"Visibility: {visibility} km",
            f"Cloud cover: {cloud_cover}%",
            f"Precipitation: {precip} mm",
            f"UV index: {uv}",
            f"Observed at: {obs_time}",
        ]
        return "\n".join(lines)

    except Exception as e:
        return f"Error parsing weather data: {type(e).__name__}: {e}"


def _clean_forecast(data: dict, location: str) -> str:
    try:
        area = data.get("nearest_area", [{}])[0]
        city = area.get("areaName", [{}])[0].get("value", location)
        country = area.get("country", [{}])[0].get("value", "")
        loc_str = f"{city}, {country}" if country else city

        days = data.get("weather", [])
        if not days:
            return f"No forecast data available for '{location}'."

        lines = [f"3-day forecast for {loc_str}:", ""]

        for day in days:
            date = day.get("date", "?")
            max_c = day.get("maxtempC", "?")
            min_c = day.get("mintempC", "?")
            avg_c = day.get("avgtempC", "?")
            sun_hours = day.get("sunHour", "?")
            uv = day.get("uvIndex", "?")
            total_snow = day.get("totalSnow_cm", "0")
            total_precip = day.get("hourly", [{}])
            precip_sum = sum(float(h.get("precipMM", 0)) for h in total_precip)

            hourly = day.get("hourly", [])
            desc_set = set()
            for h in hourly:
                for wd in h.get("weatherDesc", [{}]):
                    v = wd.get("value", "")
                    if v:
                        desc_set.add(v)
            desc_str = ", ".join(sorted(desc_set)) if desc_set else "N/A"

            lines.append(f"--- {date} ---")
            lines.append(f"  Condition: {desc_str}")
            lines.append(f"  Temperature: {min_c}..{max_c} C (avg {avg_c} C)")
            lines.append(f"  Precipitation: {precip_sum:.1f} mm")
            lines.append(f"  Snow: {total_snow} cm")
            lines.append(f"  Sun hours: {sun_hours}")
            lines.append(f"  UV index: {uv}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"Error parsing forecast: {type(e).__name__}: {e}"


@mcp.tool()
async def get_weather(
    location: str,
    lang: str = "en",
) -> str:
    """
    Get current weather conditions for a given location.

    Uses the free wttr.in API — no API key required. Works with city names,
    airport codes, zip codes, or coordinates (e.g. "Moscow", "JFK", "55.75,37.62").

    Returns: temperature, feels-like, humidity, wind, pressure, visibility,
    cloud cover, precipitation, UV index, and weather description.

    Parameters:
    - location: City name, airport code, zip code, or "lat,lon" coordinates.
    - lang: Language code for the description (e.g. "en", "ru", "de"). Default: "en".
    """
    try:
        url = f"{_WTTR_BASE}/{location}"
        params = {"format": "j1", "lang": lang}

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        return _clean_wttr_json(data, location)

    except httpx.HTTPStatusError as e:
        return f"Error: Location '{location}' not found or API error (HTTP {e.response.status_code})."
    except Exception as e:
        return f"Error getting weather: {type(e).__name__}: {e}"


@mcp.tool()
async def get_forecast(
    location: str,
    lang: str = "en",
) -> str:
    """
    Get a 3-day weather forecast for a given location.

    Uses the free wttr.in API. Returns daily high/low temperatures,
    conditions, precipitation, snow, sun hours, and UV index for
    today and the next 2 days.

    Parameters:
    - location: City name, airport code, zip code, or "lat,lon" coordinates.
    - lang: Language code. Default: "en".
    """
    try:
        url = f"{_WTTR_BASE}/{location}"
        params = {"format": "j1", "lang": lang}

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        return _clean_forecast(data, location)

    except httpx.HTTPStatusError as e:
        return f"Error: Location '{location}' not found or API error (HTTP {e.response.status_code})."
    except Exception as e:
        return f"Error getting forecast: {type(e).__name__}: {e}"


if __name__ == "__main__":
    print(f"[MCP] weather: Starting MCP server (tool_weather.py)", file=sys.stderr)
    print(f"[MCP] weather: Ready to accept stdio MCP connections", file=sys.stderr)
    mcp.run(transport="stdio")
