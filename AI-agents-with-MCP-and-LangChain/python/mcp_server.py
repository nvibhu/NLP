#!/usr/bin/env python3
"""
MCP Weather Server (Python) using Open‑Meteo
Exposes a single tool: weather_forecast(location: str, days_ahead: int=0)
"""
from __future__ import annotations
import requests
import anyio
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("mcp-weather-py")


def _geocode(name: str) -> dict:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": name, "count": 1}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data or not data.get("results"):
        raise ValueError(f"Location not found: {name}")
    res = data["results"][0]
    return {
        "latitude": res["latitude"],
        "longitude": res["longitude"],
        "name": res.get("name"),
        "country": res.get("country"),
        "timezone": res.get("timezone"),
    }


def _forecast(lat: float, lon: float, days_ahead: int) -> dict:
    daily = ",".join([
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_probability_max",
        "weathercode",
    ])
    # forecast_days = days_ahead + 1 to make sure the target day index exists
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": daily,
        "forecast_days": max(1, min(7, days_ahead + 1)),
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def weather_forecast(location: str, days_ahead: int = 0) -> dict:
    """Get a daily weather forecast for a location.

    Args:
        location: City name (e.g., Bengaluru)
        days_ahead: 0=today, 1=tomorrow, up to 7
    Returns:
        Dict with basic daily forecast fields.
    """
    days = max(0, min(7, int(days_ahead)))
    geo = _geocode(location)
    fc = _forecast(geo["latitude"], geo["longitude"], days)
    times = (fc.get("daily") or {}).get("time") or []
    idx = min(days, max(0, len(times) - 1))
    daily = fc.get("daily", {})
    return {
        "location": {
            "name": geo.get("name"),
            "country": geo.get("country"),
            "timezone": geo.get("timezone"),
            "lat": geo.get("latitude"),
            "lon": geo.get("longitude"),
        },
        "date": times[idx] if times else None,
        "temperature_max_c": (daily.get("temperature_2m_max") or [None])[idx] if daily.get("temperature_2m_max") else None,
        "temperature_min_c": (daily.get("temperature_2m_min") or [None])[idx] if daily.get("temperature_2m_min") else None,
        "precipitation_probability_max": (daily.get("precipitation_probability_max") or [None])[idx] if daily.get("precipitation_probability_max") else None,
        "weathercode": (daily.get("weathercode") or [None])[idx] if daily.get("weathercode") else None,
    }


# Register tool explicitly for MCP Server (compatible with mcp==1.0.0)
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "location": {"type": "string", "description": "City name, e.g., Bengaluru"},
        "days_ahead": {"type": "integer", "minimum": 0, "maximum": 7, "default": 0},
    },
    "required": ["location"],
    "additionalProperties": False,
}

# Some versions of the SDK expect keyword args; attempt to register with keywords
try:
    server.add_tool(
        name="weather_forecast",
        description="Get a daily weather forecast for a location.",
        input_schema=INPUT_SCHEMA,
        handler=weather_forecast,
    )
except TypeError:
    # Fallback positional signature
    server.add_tool("weather_forecast", "Get a daily weather forecast for a location.", weather_forecast, INPUT_SCHEMA)


async def amain() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write)


if __name__ == "__main__":
    anyio.run(amain)
