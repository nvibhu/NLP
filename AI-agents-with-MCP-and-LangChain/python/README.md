AI Agents with MCP and LangChain — Python (Weather Forecast)

This is a Python implementation of the LangChain agent and MCP server for weather forecasting, using Open‑Meteo. The agent uses an LLM (OpenAI via langchain-openai) to extract tool arguments, calls the MCP tool over stdio, and composes a concise answer.

Prerequisites
- Python 3.10+
- pip
- An OpenAI API key (or adapt the model/provider in agent.py)

Setup
1) Create and activate a virtual environment
   python3 -m venv .venv
   source .venv/bin/activate

2) Install dependencies
   pip install -r requirements.txt

3) Configure environment
   cp .env.example .env
   # edit .env and set OPENAI_API_KEY

Run
- Start the MCP server (usually not necessary to run manually; the agent will spawn it automatically)
  python mcp_server.py

- Ask the agent a question (spawns the MCP server, calls the tool, and prints a friendly answer)
  python agent.py "What will the weather be in Bengaluru tomorrow?"

Notes
- The MCP server exposes a tool named weather_forecast with arguments: {"location": string, "days_ahead": int}.
- The server uses the Open‑Meteo Geocoding API and Forecast API (no API key required).
- If you prefer a different model, set MODEL in .env (defaults to gpt-4o-mini in the code).
