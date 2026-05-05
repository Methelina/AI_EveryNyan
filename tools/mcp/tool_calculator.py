"""
MCP server providing a safe mathematical expression calculator.

Tools:
  - calculate: evaluate a math expression (arithmetic, trig, log, etc.)
  - convert_units: convert between common units (temperature, length, weight, data)

/tools/mcp/tool_calculator.py

Version:     0.1.0
Author:      pytraveler
Created:     2026-05-05
"""

import sys
import math
import ast
from typing import Optional

from fastmcp import FastMCP

mcp = FastMCP("calculator")

_SAFE_NAMES = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "int": int,
    "float": float,
    "bool": bool,
    "sqrt": math.sqrt,
    "cbrt": math.cbrt,
    "exp": math.exp,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "ln": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "degrees": math.degrees,
    "radians": math.radians,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}


class _SafeEval(ast.NodeVisitor):
    ALLOWED = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare,
        ast.BoolOp, ast.IfExp, ast.Call, ast.Constant, ast.Name,
        ast.Load, ast.Mod, ast.Pow, ast.Lt, ast.Gt, ast.LtE, ast.GtE,
        ast.Eq, ast.NotEq, ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
        ast.Invert, ast.Mult, ast.Div, ast.FloorDiv, ast.Add, ast.Sub,
        ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift,
        ast.List, ast.Tuple,
    )

    def visit(self, node):
        if not isinstance(node, self.ALLOWED):
            raise ValueError(f"Unsupported syntax: {type(node).__name__}")
        return super().visit(node)


def _safe_eval(expr: str) -> float | int:
    tree = ast.parse(expr, mode="eval")
    _SafeEval().visit(tree)

    compiled = compile(tree, "<calc>", "eval")
    result = eval(compiled, {"__builtins__": {}}, _SAFE_NAMES)
    return result


@mcp.tool()
async def calculate(
    expression: str,
    precision: int = -1,
) -> str:
    """
    Evaluate a mathematical expression and return the result.

    Supports: +, -, *, /, //, %, **, parentheses, and math functions.
    Functions: sqrt, cbrt, exp, log, log2, log10, ln, sin, cos, tan,
               asin, acos, atan, sinh, cosh, tanh, degrees, radians,
               ceil, floor, round, abs, min, max, factorial, gcd, pow.
    Constants: pi, e, tau, inf.

    Examples:
      - "2 + 3 * 4" → 14
      - "sqrt(144)" → 12
      - "sin(pi / 2)" → 1.0
      - "log2(1024)" → 10
      - "2 ** 64" → 18446744073709551616
      - "round(347 * 0.18 + 12.5, 2)" → 74.96

    Parameters:
    - expression: The mathematical expression to evaluate.
    - precision: Number of decimal places for rounding. -1 = no rounding (full precision).
    """
    try:
        result = _safe_eval(expression)

        if precision >= 0:
            result = round(result, precision)

        if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
            display = str(int(result))
        elif isinstance(result, float):
            display = f"{result:.15g}"
        else:
            display = str(result)

        return f"{expression} = {display}"

    except Exception as e:
        return f"Error evaluating expression: {type(e).__name__}: {e}"


_UNIT_CONVERSIONS = {
    "temperature": {
        "C": {"F": lambda x: x * 9 / 5 + 32, "K": lambda x: x + 273.15},
        "F": {"C": lambda x: (x - 32) * 5 / 9, "K": lambda x: (x - 32) * 5 / 9 + 273.15},
        "K": {"C": lambda x: x - 273.15, "F": lambda x: (x - 273.15) * 9 / 5 + 32},
    },
    "length": {
        "mm": {"cm": 0.1, "m": 0.001, "km": 1e-6, "in": 0.0393701, "ft": 0.00328084, "mi": 6.21371e-7},
        "cm": {"mm": 10, "m": 0.01, "km": 1e-5, "in": 0.393701, "ft": 0.0328084, "mi": 6.21371e-6},
        "m": {"mm": 1000, "cm": 100, "km": 0.001, "in": 39.3701, "ft": 3.28084, "mi": 0.000621371, "yd": 1.09361},
        "km": {"mm": 1e6, "cm": 1e5, "m": 1000, "in": 39370.1, "ft": 3280.84, "mi": 0.621371, "yd": 1093.61},
        "in": {"mm": 25.4, "cm": 2.54, "m": 0.0254, "ft": 1 / 12, "yd": 1 / 36},
        "ft": {"mm": 304.8, "cm": 30.48, "m": 0.3048, "in": 12, "yd": 1 / 3, "mi": 1 / 5280},
        "yd": {"m": 0.9144, "ft": 3, "in": 36, "cm": 91.44},
        "mi": {"km": 1.60934, "m": 1609.34, "ft": 5280, "yd": 1760},
    },
    "weight": {
        "g": {"kg": 0.001, "mg": 1000, "lb": 0.00220462, "oz": 0.035274},
        "kg": {"g": 1000, "mg": 1e6, "lb": 2.20462, "oz": 35.274, "t": 0.001},
        "mg": {"g": 0.001, "kg": 1e-6, "lb": 2.20462e-6, "oz": 3.5274e-5},
        "lb": {"g": 453.592, "kg": 0.453592, "oz": 16, "mg": 453592},
        "oz": {"g": 28.3495, "kg": 0.0283495, "lb": 0.0625, "mg": 28349.5},
        "t": {"kg": 1000, "g": 1e6, "lb": 2204.62, "oz": 35274},
    },
    "data": {
        "B": {"KB": 1 / 1024, "MB": 1 / (1024 ** 2), "GB": 1 / (1024 ** 3), "TB": 1 / (1024 ** 4)},
        "KB": {"B": 1024, "MB": 1 / 1024, "GB": 1 / (1024 ** 2), "TB": 1 / (1024 ** 3)},
        "MB": {"B": 1024 ** 2, "KB": 1024, "GB": 1 / 1024, "TB": 1 / (1024 ** 2)},
        "GB": {"B": 1024 ** 3, "KB": 1024 ** 2, "MB": 1024, "TB": 1 / 1024},
        "TB": {"B": 1024 ** 4, "KB": 1024 ** 3, "MB": 1024 ** 2, "GB": 1024},
    },
    "time": {
        "s": {"ms": 1000, "min": 1 / 60, "h": 1 / 3600, "d": 1 / 86400},
        "ms": {"s": 0.001, "min": 1 / 60000, "h": 1 / 3.6e6},
        "min": {"s": 60, "ms": 60000, "h": 1 / 60, "d": 1 / 1440},
        "h": {"s": 3600, "min": 60, "d": 1 / 24},
        "d": {"h": 24, "min": 1440, "s": 86400},
    },
}


@mcp.tool()
async def convert_units(
    value: float,
    from_unit: str,
    to_unit: str,
) -> str:
    """
    Convert a value between common units.

    Supported categories and units (case-insensitive):
    - Temperature: C, F, K
    - Length: mm, cm, m, km, in, ft, yd, mi
    - Weight: g, kg, mg, lb, oz, t
    - Data: B, KB, MB, GB, TB
    - Time: s, ms, min, h, d

    Examples:
      - convert_units(100, "C", "F") → 212
      - convert_units(5, "km", "mi") → 3.106855
      - convert_units(1024, "MB", "GB") → 1

    Parameters:
    - value: The numeric value to convert.
    - from_unit: Source unit abbreviation.
    - to_unit: Target unit abbreviation.
    """
    try:
        fu = from_unit.strip().lower()
        tu = to_unit.strip().lower()

        if fu == tu:
            return f"{value} {from_unit} = {value} {to_unit}"

        for category, units in _UNIT_CONVERSIONS.items():
            if fu in units and tu in units.get(fu, {}):
                conv = units[fu][tu]
                if callable(conv):
                    result = conv(value)
                else:
                    result = value * conv

                if isinstance(result, float) and abs(result) < 1e10:
                    if result == int(result):
                        result = int(result)
                    else:
                        result = round(result, 8)
                        if result == int(result):
                            result = int(result)

                return f"{value} {from_unit} = {result} {to_unit}"

        return (
            f"Error: Cannot convert from '{from_unit}' to '{to_unit}'. "
            f"Check that both units belong to the same category."
        )

    except Exception as e:
        return f"Error converting units: {type(e).__name__}: {e}"


if __name__ == "__main__":
    print(f"[MCP] calculator: Starting MCP server (tool_calculator.py)", file=sys.stderr)
    print(f"[MCP] calculator: Ready to accept stdio MCP connections", file=sys.stderr)
    mcp.run(transport="stdio")
