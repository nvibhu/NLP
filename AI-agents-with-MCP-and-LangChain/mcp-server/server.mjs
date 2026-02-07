#!/usr/bin/env node
// MCP Weather Server using Open-Meteo
import fetch from 'node-fetch';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/transports/stdio.js';

const server = new Server(
  { name: 'mcp-weather', version: '1.0.0' },
  { capabilities: { tools: {} } }
);

// Tool schema
const inputSchema = {
  type: 'object',
  properties: {
    location: { type: 'string', description: 'City name, e.g., Bengaluru' },
    days_ahead: {
      type: 'integer',
      description: 'Days ahead (0=today, 1=tomorrow). Max 7',
      minimum: 0,
      maximum: 7,
      default: 0
    }
  },
  required: ['location'],
  additionalProperties: false
};

async function geocode(name) {
  const url = new URL('https://geocoding-api.open-meteo.com/v1/search');
  url.searchParams.set('name', name);
  url.searchParams.set('count', '1');
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`Geocoding failed: ${res.status}`);
  const data = await res.json();
  if (!data || !data.results || data.results.length === 0) {
    throw new Error(`Location not found: ${name}`);
  }
  const r = data.results[0];
  return { latitude: r.latitude, longitude: r.longitude, name: r.name, country: r.country, timezone: r.timezone };
}

async function forecast(lat, lon, days) {
  // daily params
  const daily = [
    'temperature_2m_max',
    'temperature_2m_min',
    'precipitation_probability_max',
    'weathercode'
  ].join(',');
  const url = new URL('https://api.open-meteo.com/v1/forecast');
  url.searchParams.set('latitude', String(lat));
  url.searchParams.set('longitude', String(lon));
  url.searchParams.set('daily', daily);
  url.searchParams.set('forecast_days', String(Math.min(7, Math.max(1, (days ?? 0) + 1))));
  url.searchParams.set('timezone', 'auto');
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`Forecast failed: ${res.status}`);
  return res.json();
}

server.tool(
  'weather_forecast',
  'Get a daily weather forecast for a location. Input: {"location":"City","days_ahead":0-7}',
  { schema: inputSchema },
  async ({ arguments: args }) => {
    const location = args.location;
    const days = args.days_ahead ?? 0;
    const geo = await geocode(location);
    const fc = await forecast(geo.latitude, geo.longitude, days);
    const idx = Math.min(days, (fc.daily?.time?.length ?? 1) - 1);
    const reply = {
      location: { name: geo.name, country: geo.country, timezone: geo.timezone, lat: geo.latitude, lon: geo.longitude },
      date: fc.daily?.time?.[idx],
      temperature_max_c: fc.daily?.temperature_2m_max?.[idx],
      temperature_min_c: fc.daily?.temperature_2m_min?.[idx],
      precipitation_probability_max: fc.daily?.precipitation_probability_max?.[idx],
      weathercode: fc.daily?.weathercode?.[idx]
    };
    return {
      content: [
        { type: 'text', text: JSON.stringify(reply, null, 2) }
      ]
    };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
