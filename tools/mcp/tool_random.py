"""
MCP server providing random generation tools.

Tools:
  - generate_uuid:        generate UUID v4 or v7
  - generate_random_string: generate random alphanumeric/password string
  - generate_random_number: generate random integer or float in a range
  - pick_random:          pick random items from a list

/tools/mcp/tool_random.py

Version:     0.1.0
Author:      pytraveler
Created:     2026-05-05
"""

import sys
import uuid
import secrets
import random
import string
from typing import Optional

from fastmcp import FastMCP

mcp = FastMCP("random")


@mcp.tool()
async def generate_uuid(
    version: int = 4,
    count: int = 1,
    uppercase: bool = False,
) -> str:
    """
    Generate one or more UUIDs (Universal Unique Identifiers).

    Use this when you need unique identifiers, random IDs, or UUIDs for any purpose.

    Parameters:
    - version: UUID version (4 = random, 7 = time-ordered). Default: 4.
    - count: How many UUIDs to generate (1-100). Default: 1.
    - uppercase: Return uppercase hex. Default: false (lowercase).
    """
    try:
        count = max(1, min(count, 100))
        results = []
        for _ in range(count):
            if version == 7:
                u = uuid.uuid7()
            else:
                u = uuid.uuid4()
            s = str(u).upper() if uppercase else str(u)
            results.append(s)

        if count == 1:
            return f"UUID v{version}: {results[0]}"
        return f"UUID v{version} ({count}):\n" + "\n".join(results)

    except Exception as e:
        return f"Error generating UUID: {type(e).__name__}: {e}"


@mcp.tool()
async def generate_random_string(
    length: int = 16,
    count: int = 1,
    charset: str = "alphanumeric",
    exclude_ambiguous: bool = False,
) -> str:
    """
    Generate random strings suitable for passwords, tokens, or IDs.

    Use this for generating passwords, API keys, random tokens, or any
    random string needs.

    Parameters:
    - length: String length (1-256). Default: 16.
    - count: Number of strings to generate (1-50). Default: 1.
    - charset: Character set to use:
        "alphanumeric" = a-z, A-Z, 0-9 (default)
        "alpha" = a-z, A-Z
        "numeric" = 0-9
        "hex" = 0-9, a-f
        "lowercase" = a-z
        "uppercase" = A-Z
        "password" = a-z, A-Z, 0-9, and special chars (!@#$%^&*_-+=)
        "full" = all printable ASCII
    - exclude_ambiguous: Remove easily confused chars (0/O, 1/l/I). Default: false.
    """
    try:
        length = max(1, min(length, 256))
        count = max(1, min(count, 50))

        charsets = {
            "alphanumeric": string.ascii_letters + string.digits,
            "alpha": string.ascii_letters,
            "numeric": string.digits,
            "hex": string.digits + "abcdef",
            "lowercase": string.ascii_lowercase,
            "uppercase": string.ascii_uppercase,
            "password": string.ascii_letters + string.digits + "!@#$%^&*_-+=",
            "full": string.printable.strip(),
        }

        chars = charsets.get(charset, charsets["alphanumeric"])

        if exclude_ambiguous:
            ambiguous = "0O1lI|`'"
            chars = "".join(c for c in chars if c not in ambiguous)

        if not chars:
            return "Error: Empty charset after excluding ambiguous characters."

        results = []
        for _ in range(count):
            s = "".join(secrets.choice(chars) for _ in range(length))
            results.append(s)

        if count == 1:
            entropy = len(chars) ** length
            return (
                f"Random string ({charset}, length={length}):\n"
                f"{results[0]}\n"
                f"Entropy: ~{entropy.bit_length()} bits ({len(chars)}^{length} possibilities)"
            )

        return f"Random strings ({charset}, length={length}, count={count}):\n" + "\n".join(results)

    except Exception as e:
        return f"Error generating random string: {type(e).__name__}: {e}"


@mcp.tool()
async def generate_random_number(
    min_val: float = 0,
    max_val: float = 100,
    count: int = 1,
    integer: bool = True,
    seed: Optional[int] = None,
) -> str:
    """
    Generate random number(s) within a given range.

    Use this for dice rolls, random picks, simulations, or any random number needs.

    Parameters:
    - min_val: Minimum value (inclusive). Default: 0.
    - max_val: Maximum value (inclusive for integers, exclusive for floats). Default: 100.
    - count: How many numbers to generate (1-1000). Default: 1.
    - integer: Return integers (true) or floats (false). Default: true.
    - seed: Optional seed for reproducibility. If omitted, uses system entropy.
    """
    try:
        count = max(1, min(count, 1000))

        if seed is not None:
            random.seed(seed)

        results = []
        for _ in range(count):
            if integer:
                val = random.randint(int(min_val), int(max_val))
            else:
                val = random.uniform(min_val, max_val)
                val = round(val, 6)
            results.append(str(val))

        rng_type = "integer" if integer else "float"
        if count == 1:
            return f"Random {rng_type} [{min_val}, {max_val}]: {results[0]}"
        return f"Random {rng_type}s [{min_val}, {max_val}] ({count}):\n" + ", ".join(results)

    except Exception as e:
        return f"Error generating random number: {type(e).__name__}: {e}"


@mcp.tool()
async def pick_random(
    items: str,
    count: int = 1,
    allow_duplicates: bool = False,
    delimiter: str = ",",
) -> str:
    """
    Pick random item(s) from a list of choices.

    Use this for random selections, raffles, "choose one from these", or
    decision-making like "pick a random option".

    Parameters:
    - items: Comma-separated (or custom delimiter) list of items to pick from.
             Example: "pizza, sushi, burger, salad, ramen"
    - count: Number of items to pick. Default: 1.
    - allow_duplicates: Allow picking the same item more than once. Default: false.
    - delimiter: Separator between items. Default: ",".
    """
    try:
        choices = [item.strip() for item in items.split(delimiter) if item.strip()]
        if not choices:
            return "Error: No items provided. Use comma-separated values."

        count = max(1, min(count, len(choices) if not allow_duplicates else 1000))

        if allow_duplicates:
            picked = [secrets.choice(choices) for _ in range(count)]
        else:
            if count > len(choices):
                return f"Error: Cannot pick {count} unique items from a list of {len(choices)}."
            picked = random.sample(choices, count)

        if count == 1:
            return f"Random pick from {len(choices)} items: {picked[0]}"
        return f"Random {count} picks from {len(choices)} items:\n" + "\n".join(
            f"  {i}. {item}" for i, item in enumerate(picked, 1)
        )

    except Exception as e:
        return f"Error picking random: {type(e).__name__}: {e}"


if __name__ == "__main__":
    print(f"[MCP] random: Starting MCP server (tool_random.py)", file=sys.stderr)
    print(f"[MCP] random: Ready to accept stdio MCP connections", file=sys.stderr)
    mcp.run(transport="stdio")
