#!/usr/bin/env node
// LangChain-based agent that calls an MCP tool for weather forecasts
import 'dotenv/config';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { ChatOpenAI } from '@langchain/openai';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/transports/stdio.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, '..');

function startMcpServer() {
  const serverPath = path.join(PROJECT_ROOT, 'mcp-server', 'server.mjs');
  const child = spawn(process.execPath, [serverPath], {
    cwd: PROJECT_ROOT,
    stdio: ['pipe', 'pipe', 'inherit']
  });
  return child;
}

async function connectMcp(child) {
  const transport = new StdioClientTransport({ stdin: child.stdin, stdout: child.stdout });
  const client = new Client({ name: 'weather-agent-client', version: '1.0.0' }, { capabilities: { tools: {} } });
  await client.connect(transport);
  return client;
}

function parseAIContent(aiMessage) {
  // Try to normalize LangChain AIMessage content to a string
  const c = aiMessage?.content;
  if (typeof c === 'string') return c.trim();
  if (Array.isArray(c) && c.length && typeof c[0]?.text === 'string') return c[0].text.trim();
  try { return JSON.stringify(c); } catch { return String(c); }
}

async function main() {
  const question = process.argv.slice(2).join(' ').trim();
  if (!question) {
    console.error('Usage: npm run agent -- "What will the weather be in <city> tomorrow?"');
    process.exit(1);
  }

  const model = process.env.MODEL || 'gpt-4o-mini';
  const llm = new ChatOpenAI({ model, temperature: 0 });

  // 1) Extract structured tool inputs from the natural language question
  const extractPrompt = `You are a planner that extracts arguments for a weather API.\n` +
    `From the user question, produce ONLY a compact JSON object with keys: location (string), days_ahead (integer 0-7).\n` +
    `days_ahead=0 means today, 1=tomorrow. If not specified, default to 0.\n` +
    `Examples:\n` +
    `Q: What's the weather in Bengaluru tomorrow?\n` +
    `{"location":"Bengaluru","days_ahead":1}\n` +
    `Q: Weather in Mumbai?\n` +
    `{"location":"Mumbai","days_ahead":0}\n` +
    `Now for this question, output JSON only, no commentary or code fences:\n` +
    `Q: ${question}`;

  const extractRes = await llm.invoke(extractPrompt);
  const extractText = parseAIContent(extractRes);
  let args;
  try {
    args = JSON.parse(extractText);
    if (typeof args.location !== 'string') throw new Error('Missing location');
    if (typeof args.days_ahead !== 'number') args.days_ahead = 0;
    args.days_ahead = Math.max(0, Math.min(7, Math.round(args.days_ahead)));
  } catch (e) {
    console.error('Could not parse tool arguments from LLM output:', extractText);
    process.exit(2);
  }

  // 2) Call MCP tool
  const child = startMcpServer();
  const client = await connectMcp(child);
  let toolResult;
  try {
    toolResult = await client.callTool({ name: 'weather_forecast', arguments: args });
  } catch (e) {
    console.error('MCP tool call failed:', e);
    child.kill();
    process.exit(3);
  }

  // 3) Parse MCP tool response
  let weatherJson;
  try {
    const textPart = (toolResult?.content || []).find((c) => c.type === 'text');
    weatherJson = JSON.parse(textPart?.text || '{}');
  } catch (e) {
    console.error('Failed to parse MCP response:', toolResult);
    child.kill();
    process.exit(4);
  }

  // 4) Compose a friendly answer
  const composePrompt = `Given this JSON weather forecast, answer the user's question clearly and concisely.\n` +
    `JSON:\n${JSON.stringify(weatherJson, null, 2)}\n` +
    `User question: ${question}\n` +
    `Respond in 2-4 sentences.`;
  const composeRes = await llm.invoke(composePrompt);
  const finalText = parseAIContent(composeRes);

  console.log(finalText);

  // Cleanup
  child.kill();
}

main().catch((err) => {
  console.error(err);
  process.exit(99);
});
