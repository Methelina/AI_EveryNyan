"""
MCP server providing currency conversion tools using free exchange rate APIs.

Primary API:   https://frankfurter.app (European Central Bank data, ~30 currencies)
Fallback API:  https://open.er-api.com (broader coverage, ~150 currencies)

Tools:
  - convert_currency:  convert an amount between two currencies
  - list_currencies:   list all available currency codes and names
  - exchange_rate:     get the current exchange rate for a currency pair

/tools/mcp/tool_currency.py

Version:     0.1.0
Author:      pytraveler
Created:     2026-05-05
"""

import sys
import httpx
from datetime import datetime
from typing import Optional

from fastmcp import FastMCP

mcp = FastMCP("currency")

_TIMEOUT = 15.0

_FRANKFURTER_BASE = "https://frankfurter.app"
_ERAPI_BASE = "https://open.er-api.com/v6"


async def _fetch_frankfurter(path: str, params: dict = None) -> dict:
    url = f"{_FRANKFURTER_BASE}{path}"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


async def _fetch_erapi(path: str) -> dict:
    url = f"{_ERAPI_BASE}{path}"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


async def _get_rates(base: str) -> tuple[dict, str]:
    base = base.upper()
    try:
        data = await _fetch_frankfurter("/latest", {"from": base})
        rates = {base: 1.0}
        for k, v in data.get("rates", {}).items():
            rates[k] = v
        date = data.get("date", "")
        return rates, f"frankfurter.app (ECB {date})"
    except Exception:
        pass

    try:
        data = await _fetch_erapi(f"/latest/{base}")
        rates = {base: 1.0}
        for k, v in data.get("rates", {}).items():
            rates[k] = v
        date = data.get("time_last_update_utc", "")
        return rates, f"open.er-api.com ({date})"
    except Exception:
        pass

    return {}, ""


async def _get_rate_pair(base: str, target: str) -> tuple[Optional[float], str]:
    base = base.upper()
    target = target.upper()
    if base == target:
        return 1.0, "same currency"

    try:
        data = await _fetch_frankfurter("/latest", {"from": base, "to": target})
        rate = data.get("rates", {}).get(target)
        if rate is not None:
            date = data.get("date", "")
            return rate, f"frankfurter.app (ECB {date})"
    except Exception:
        pass

    try:
        data = await _fetch_erapi(f"/latest/{base}")
        rate = data.get("rates", {}).get(target)
        if rate is not None:
            date = data.get("time_last_update_utc", "")
            return rate, f"open.er-api.com ({date})"
    except Exception:
        pass

    return None, ""


async def _get_currencies() -> tuple[dict, str]:
    try:
        data = await _fetch_frankfurter("/currencies")
        return data, "frankfurter.app"
    except Exception:
        pass

    try:
        data = await _fetch_erapi("/latest/USD")
        rates = data.get("rates", {})
        result = {k: "" for k in rates}
        result["USD"] = ""
        return result, "open.er-api.com"
    except Exception:
        pass

    return {}, ""


@mcp.tool()
async def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
) -> str:
    """
    Convert an amount from one currency to another using current exchange rates.

    Uses live rates from free APIs (no API key required). Falls back to a secondary
    source if the primary is unavailable.

    Supports ~150+ currencies: USD, EUR, GBP, JPY, CNY, RUB, KRW, CHF, CAD, AUD,
    INR, BRL, MXN, TRY, PLN, CZK, SEK, NOK, DKK, NZD, ZAR, SGD, HKD, THB, UAH,
    and many more.

    Examples:
      - convert_currency(100, "USD", "EUR")
      - convert_currency(5000, "RUB", "CNY")
      - convert_currency(42.5, "JPY", "GBP")

    Parameters:
    - amount: The amount of money to convert.
    - from_currency: Source currency code (3 letters, e.g. "USD").
    - to_currency: Target currency code (3 letters, e.g. "EUR").
    """
    try:
        from_curr = from_currency.strip().upper()
        to_curr = to_currency.strip().upper()

        if len(from_curr) != 3 or len(to_curr) != 3:
            return "Error: Currency codes must be 3 letters (e.g. USD, EUR, RUB)."

        rate, source = await _get_rate_pair(from_curr, to_curr)

        if rate is None:
            return f"Error: Could not fetch exchange rate for {from_curr} -> {to_curr}. Both APIs unavailable."

        converted = amount * rate

        if converted == int(converted) and abs(converted) < 1e12:
            converted_str = f"{int(converted):,}"
        else:
            converted_str = f"{converted:,.2f}"

        if amount == int(amount):
            amount_str = f"{int(amount):,}"
        else:
            amount_str = f"{amount:,.2f}"

        return (
            f"{amount_str} {from_curr} = {converted_str} {to_curr}\n"
            f"Rate: 1 {from_curr} = {rate:.6f} {to_curr}\n"
            f"Source: {source}"
        )

    except Exception as e:
        return f"Error converting currency: {type(e).__name__}: {e}"


@mcp.tool()
async def exchange_rate(
    base: str = "USD",
    targets: str = "",
) -> str:
    """
    Get current exchange rate(s) for a base currency against one or more targets.

    If targets is empty, returns rates for the most common currencies.

    Examples:
      - exchange_rate("USD")           — USD vs common currencies
      - exchange_rate("EUR", "USD,GBP,CHF")  — EUR vs specific currencies
      - exchange_rate("RUB", "USD,EUR,CNY")

    Parameters:
    - base: Base currency code (3 letters). Default: "USD".
    - targets: Comma-separated target currency codes. Empty = top 15 common currencies.
    """
    try:
        base = base.strip().upper()

        if targets.strip():
            target_list = [t.strip().upper() for t in targets.split(",") if t.strip()]
        else:
            target_list = [
                "USD", "EUR", "GBP", "JPY", "CNY", "CHF", "CAD", "AUD",
                "KRW", "RUB", "INR", "BRL", "TRY", "THB", "UAH",
            ]
            target_list = [t for t in target_list if t != base]

        rates, source = await _get_rates(base)

        if not rates:
            return f"Error: Could not fetch rates for {base}. Both APIs unavailable."

        lines = [f"Exchange rates for 1 {base}:", ""]

        found = []
        missing = []
        for t in target_list:
            if t in rates:
                found.append((t, rates[t]))
            else:
                missing.append(t)

        for code, rate in found:
            if rate == int(rate) and abs(rate) < 1e10:
                rate_str = f"{int(rate):,}"
            elif rate < 0.01:
                rate_str = f"{rate:.8f}"
            else:
                rate_str = f"{rate:.4f}"
            lines.append(f"  1 {base} = {rate_str} {code}")

        if missing:
            lines.append(f"\n  Not available: {', '.join(missing)}")

        lines.append(f"\nSource: {source}")
        return "\n".join(lines)

    except Exception as e:
        return f"Error getting exchange rate: {type(e).__name__}: {e}"


@mcp.tool()
async def list_currencies() -> str:
    """
    List all available currency codes and their names.

    Returns the full list of currencies supported for conversion,
    sourced from the exchange rate API.

    Use this when you need to check if a specific currency is supported
    or to see all available options.
    """
    try:
        currencies, source = await _get_currencies()

        if not currencies:
            return "Error: Could not fetch currency list. Both APIs unavailable."

        lines = [f"Available currencies ({len(currencies)} total, source: {source}):", ""]

        sorted_currs = sorted(currencies.items(), key=lambda x: x[0])

        common = {"USD", "EUR", "GBP", "JPY", "CNY", "CHF", "CAD", "AUD",
                  "KRW", "RUB", "INR", "BRL", "MXN", "TRY", "PLN", "CZK",
                  "SEK", "NOK", "DKK", "NZD", "ZAR", "SGD", "HKD", "THB", "UAH"}

        lines.append("Common currencies:")
        for code, name in sorted_currs:
            if code in common:
                display = f"  {code} — {name}" if name else f"  {code}"
                lines.append(display)

        lines.append("")
        lines.append("All currencies:")
        for code, name in sorted_currs:
            display = f"  {code} — {name}" if name else f"  {code}"
            lines.append(display)

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing currencies: {type(e).__name__}: {e}"


if __name__ == "__main__":
    print(f"[MCP] currency: Starting MCP server (tool_currency.py)", file=sys.stderr)
    print(f"[MCP] currency: Ready to accept stdio MCP connections", file=sys.stderr)
    mcp.run(transport="stdio")
