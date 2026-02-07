AI Agents with MCP and LangChain — Weather Forecast

This project demonstrates how to build an AI Agent using LangChain that calls an MCP (Model Context Protocol) server tool to fetch weather forecasts. The MCP server exposes a weather_forecast tool backed by the Open-Meteo APIs (no API key required). The LangChain agent plans and decides when to call the tool.

What’s included
- mcp-server/server.mjs — MCP server exposing weather_forecast
- agent/agent.mjs — LangChain ReAct-style agent that connects to the MCP server and invokes the tool
- package.json — single root package for both server and agent
- .env.example — environment variables template

Prerequisites
- Node.js >= 18
- An LLM provider key for the agent (e.g., OPENAI_API_KEY). You can swap in another provider if preferred.

Quickstart
1) Install dependencies
   npm install

2) Configure env
   cp .env.example .env
   # Edit .env and set your OPENAI_API_KEY

3) In one terminal, run the MCP server
   npm run start:mcp

4) In a second terminal, ask the agent for weather
   npm run agent -- "What will the weather be in Bengaluru tomorrow?"

Troubleshooting
- If you see OpenAI auth errors, ensure OPENAI_API_KEY is set in .env or your shell.
- You can change the model by setting MODEL in .env (e.g., gpt-4o-mini).
- The MCP server uses stdout/stdin transport; do not run it manually unless you also wire the transport.

Notes
- The MCP server uses Open-Meteo Geocoding API and Forecast API.
- The agent wraps the MCP tool as a LangChain tool and uses an LLM (OpenAI by default) to decide when/how to call it.
- You can change the model in agent/agent.mjs (e.g., gpt-4o-mini) or swap providers.

Folder structure
- AI-agents-with-MCP-and-LangChain/
  - mcp-server/server.mjs
  - agent/agent.mjs
  - package.json
  - .env.example

Security & rate limits
- Open-Meteo is free and does not require a key, but please respect their usage policies.
