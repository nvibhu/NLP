#!/usr/bin/env python3
"""
LangChain-based agent (Python) that calls an MCP tool for weather forecasts.
It spawns the local MCP server (mcp_server.py) over stdio, calls the
weather_forecast tool, and composes a friendly answer.
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
from pathlib import Path
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage

try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
except Exception as e:
    print("Missing or incompatible 'mcp' package. Install with: pip install mcp", file=sys.stderr)
    raise


def parse_ai_content(msg: AIMessage) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content.strip()
    # LC often returns a list of parts
    try:
        if isinstance(content, list) and content and isinstance(content[0], dict) and "text" in content[0]:
            return str(content[0]["text"]).strip()
    except Exception:
        pass
    try:
        return json.dumps(content)
    except Exception:
        return str(content)


async def call_mcp_weather(location: str, days_ahead: int) -> dict:
    """Spawn the MCP server and call the weather_forecast tool via stdio."""
    here = Path(__file__).resolve().parent
    # Prefer the Node MCP server (robust SDK); it exposes the same weather_forecast tool
    node_server_path = here.parent / "mcp-server" / "server.mjs"
    use_node_server = node_server_path.exists()
    # Spawn the server via stdio_client; provide a minimal object with command/args
    class _Server:
        def __init__(self, command: str, args: list[str], env: dict | None = None, cwd: str | None = None):
            self.command = command
            self.args = args
            # stdio_client may access env and cwd attributes; provide defaults
            self.env = env
            self.cwd = cwd

    if use_node_server:
        server = _Server(command="node", args=[str(node_server_path)])
    else:
        # Fallback to Python MCP server
        server = _Server(command=sys.executable, args=[str(here / "mcp_server.py")])
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "weather_forecast",
                arguments={"location": location, "days_ahead": int(days_ahead)},
            )
            # result.content is an array of parts; we returned dict in server, so it's structured
            # FastMCP returns content already structured for JSON by default; attempt to coerce
            for part in result.content:
                if getattr(part, "type", None) == "json" or getattr(part, "type", None) == "object":
                    try:
                        return part.data if hasattr(part, "data") else part.object
                    except Exception:
                        pass
                if getattr(part, "type", None) == "text":
                    try:
                        return json.loads(part.text)
                    except Exception:
                        continue
            # last resort: return raw
            return {"raw": [getattr(p, "__dict__", str(p)) for p in result.content]}


async def main_async() -> int:
    load_dotenv()
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('Usage: python agent.py "What will the weather be in <city> tomorrow?"', file=sys.stderr)
        return 2

    # Helper: simple deterministic extractor if LLM is unavailable
    def extract_without_llm(q: str) -> tuple[str, int]:
        ql = q.lower()
        days = 0
        if "day after tomorrow" in ql:
            days = 2
        elif "tomorrow" in ql:
            days = 1
        elif "today" in ql:
            days = 0
        # capture location after 'in'
        m = re.search(r"\bin\s+([A-Za-z\s]+?)(?:\?|$|,|\s+today|\s+tomorrow)", q)
        loc = None
        if m:
            loc = m.group(1).strip()
        # fallback: take last capitalized word token
        if not loc:
            caps = re.findall(r"\b([A-Z][a-zA-Z]+)\b", q)
            if caps:
                loc = caps[-1]
        if not loc:
            loc = "Bengaluru"
        return loc, days

    use_llm = bool(os.environ.get("OPENAI_API_KEY")) and os.environ.get("NO_LLM", "0") != "1"
    if use_llm:
        try:
            model = os.environ.get("MODEL", "gpt-4o-mini")
            llm = ChatOpenAI(model=model, temperature=0)
            extract_prompt = (
                "You are a planner that extracts arguments for a weather API.\n"
                "From the user question, produce ONLY a compact JSON object with keys: "
                "location (string), days_ahead (integer 0-7).\n"
                "days_ahead=0 means today, 1=tomorrow. If not specified, default to 0.\n"
                "Examples:\n"
                "Q: What's the weather in Bengaluru tomorrow?\n"
                '{"location":"Bengaluru","days_ahead":1}\n'
                "Q: Weather in Mumbai?\n"
                '{"location":"Mumbai","days_ahead":0}\n'
                "Now for this question, output JSON only, no commentary or code fences:\n"
                f"Q: {question}"
            )
            extract_msg = await llm.ainvoke(extract_prompt)
            extract_text = parse_ai_content(extract_msg)
            args = json.loads(extract_text)
            location = str(args.get("location"))
            days_ahead = int(args.get("days_ahead", 0))
            if not location:
                raise ValueError("Missing location")
            days_ahead = max(0, min(7, days_ahead))
        except Exception as e:
            # Fall back if rate-limited or key/quota issues
            location, days_ahead = extract_without_llm(question)
            use_llm = False
    else:
        location, days_ahead = extract_without_llm(question)

    weather_json = await call_mcp_weather(location, days_ahead)

    # Compose answer
    if use_llm:
        compose_prompt = (
            "Given this JSON weather forecast, answer the user's question clearly and concisely.\n"
            f"JSON:\n{json.dumps(weather_json, indent=2)}\n"
            f"User question: {question}\n"
            "Respond in 2-4 sentences."
        )
        compose_msg = await llm.ainvoke(compose_prompt)
        print(parse_ai_content(compose_msg))
    else:
        loc = weather_json.get("location", {})
        name = loc.get("name") or location
        country = loc.get("country") or ""
        date = weather_json.get("date")
        tmax = weather_json.get("temperature_max_c")
        tmin = weather_json.get("temperature_min_c")
        pp = weather_json.get("precipitation_probability_max")
        parts = []
        if date:
            parts.append(f"On {date}")
        place = f"in {name}{', ' + country if country else ''}"
        parts.append(place)
        if tmax is not None and tmin is not None:
            parts.append(f"expect a high of {tmax}°C and a low of {tmin}°C")
        if pp is not None:
            parts.append(f"with a precipitation chance of {pp}%")
        print(" ".join(parts) + ".")
    return 0


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
