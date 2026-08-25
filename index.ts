import { ChatGroq } from "@langchain/groq";

import {
  HumanMessage,
  SystemMessage,
  type AIMessage,
} from "@langchain/core/messages";

import { createEventTool, getEventsTool, deleteEventTool } from "./tools";

import { END, MessagesAnnotation, StateGraph } from "@langchain/langgraph";

import { ToolNode } from "@langchain/langgraph/prebuilt";

import readline from "node:readline";
import { randomUUID } from "node:crypto";

// ============================================================
// COLORS
// ============================================================

const C = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  italic: "\x1b[3m",
  underline: "\x1b[4m",

  black: "\x1b[30m",
  red: "\x1b[31m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
  cyan: "\x1b[36m",
  white: "\x1b[37m",

  brightBlack: "\x1b[90m",
  brightRed: "\x1b[91m",
  brightGreen: "\x1b[92m",
  brightYellow: "\x1b[93m",
  brightBlue: "\x1b[94m",
  brightMagenta: "\x1b[95m",
  brightCyan: "\x1b[96m",
  brightWhite: "\x1b[97m",

  bgBlue: "\x1b[44m",
  bgCyan: "\x1b[46m",
  bgGreen: "\x1b[42m",
  bgMagenta: "\x1b[45m",
};

// ============================================================
// TERMINAL HELPERS
// ============================================================

function termWidth(fallback = 62) {
  return Math.max(50, Math.min(process.stdout.columns || fallback, 100));
}

/**
 * Strip ANSI codes to measure visible length correctly
 */
function visibleLength(str: string) {
  // eslint-disable-next-line no-control-regex
  return str.replace(/\x1b\[[0-9;]*m/g, "").length;
}

function padLine(content: string, width: number) {
  const pad = width - visibleLength(content);

  return content + " ".repeat(Math.max(0, pad));
}

function timestamp() {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

// ============================================================
// TOOLS
// ============================================================

const tools = [createEventTool, getEventsTool, deleteEventTool];

const TOOL_META: Record<
  string,
  {
    icon: string;
    label: string;
    color: string;
  }
> = {
  "get-events": {
    icon: "🔍",
    label: "Searching events",
    color: C.brightCyan,
  },

  "create-event": {
    icon: "🗓️",
    label: "Creating event",
    color: C.brightGreen,
  },

  "delete-event": {
    icon: "🗑️",
    label: "Deleting event",
    color: C.brightRed,
  },
};

// ============================================================
// MODEL
// ============================================================

const model = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0,
}).bindTools(tools);

// ============================================================
// SYSTEM PROMPT
// ============================================================

function getSystemPrompt() {
  const now = new Date();

  return `
You are a Google Calendar assistant.

Current date and time:
${now.toString()}

Current ISO datetime:
${now.toISOString()}

Your responsibilities:

1. Help the user find calendar events.
2. Help the user create calendar events.
3. Help the user delete calendar events.
4. Use the available tools whenever calendar information is required.
5. Never invent calendar events.
6. When the user asks about "today", "tomorrow", "this week", etc.,
   calculate the appropriate date/time range using the current date.
7. Use RFC3339 datetime values when calling calendar tools.
8. When creating an event, ask for missing required information
   such as date, start time, or end time when necessary.
9. If the user does not specify an end time for a meeting, make a
   reasonable assumption only when the context clearly allows it.
10. When an event is successfully created, clearly provide:
    - event title
    - date/time
    - attendees if applicable
    - Google Meet link if available
11. Keep responses concise and conversational.
12. Do not expose OAuth credentials, access tokens, refresh tokens,
    or other secrets.

DELETE EVENT RULES:

When the user wants to delete an event:

1. If the user provides an exact event ID, you may call delete-event.
2. If the user does NOT provide an event ID:
   - Use get-events to find the relevant events.
   - Show the matching events and their event IDs.
   - Ask the user which event ID they want to delete.
   - Do NOT call delete-event yet.
3. Never guess which event should be deleted.
4. Never delete an event based only on its title when multiple
   matching events exist.
5. delete-event is only for deleting the exact event ID selected
   by the user.

Available tools:

- get-events
  Search and retrieve calendar events.
  Returns event IDs.

- create-event
  Create a Google Calendar event.

- delete-event
  Delete a Google Calendar event using its exact event ID.
`.trim();
}

// ============================================================
// ASSISTANT NODE
// ============================================================

async function callModel(state: typeof MessagesAnnotation.State) {
  const response = await model.invoke([
    new SystemMessage(getSystemPrompt()),
    ...state.messages,
  ]);

  return {
    messages: [response],
  };
}

// ============================================================
// TOOL NODE
// ============================================================

const baseToolNode = new ToolNode(tools);

async function toolNode(state: typeof MessagesAnnotation.State) {
  const lastMessage = state.messages[state.messages.length - 1] as AIMessage;

  const calls = lastMessage.tool_calls ?? [];

  // Only show friendly tool activity.
  // Do NOT print tool arguments.
  for (const call of calls) {
    printToolCall(call.name);
  }

  // Execute tools normally.
  // Their raw responses are NOT printed.
  return await baseToolNode.invoke(state);
}

// ============================================================
// TOOL ROUTING
// ============================================================

function shouldContinue(state: typeof MessagesAnnotation.State) {
  const lastMessage = state.messages[state.messages.length - 1] as AIMessage;

  if (lastMessage.tool_calls?.length) {
    return "tools";
  }

  return END;
}

// ============================================================
// GRAPH
// ============================================================

const graph = new StateGraph(MessagesAnnotation)
  .addNode("assistant", callModel)
  .addNode("tools", toolNode)
  .addEdge("__start__", "assistant")
  .addEdge("tools", "assistant")
  .addConditionalEdges("assistant", shouldContinue, {
    tools: "tools",
    [END]: END,
  });

const app = graph.compile();

// ============================================================
// UI HELPERS
// ============================================================

function printLine(char = "─", width = termWidth()) {
  console.log(C.brightBlack + char.repeat(width) + C.reset);
}

function box(
  lines: string[],
  opts?: {
    color?: string;
    title?: string;
  },
) {
  const color = opts?.color ?? C.brightCyan;

  const width = termWidth();
  const inner = width - 4;

  const top = opts?.title
    ? `╭─ ${C.bold}${opts.title}${C.reset}${color} ${"─".repeat(
        Math.max(0, inner - visibleLength(opts.title) - 2),
      )}╮`
    : `╭${"─".repeat(width - 2)}╮`;

  console.log(color + top + C.reset);

  for (const line of lines) {
    console.log(
      `${color}│${C.reset} ${padLine(line, inner)} ${color}│${C.reset}`,
    );
  }

  console.log(color + `╰${"─".repeat(width - 2)}╯` + C.reset);
}

function printHeader() {
  console.clear();

  console.log();

  box(
    [
      "",

      `${C.brightBlue}${C.bold}📅  Google Calendar Assistant${C.reset}`,

      `${C.dim}Your AI assistant for Google Calendar${C.reset}`,

      "",

      `${C.yellow}/help${C.reset}  commands   ${C.yellow}/clear${C.reset}  reset   ${C.yellow}/exit${C.reset}  quit`,

      "",
    ],
    {
      color: C.brightCyan,
    },
  );

  console.log();
}

function printUser(message: string) {
  console.log();

  console.log(
    `${C.brightGreen}${C.bold}You${C.reset} ${C.brightBlack}· ${timestamp()}${C.reset}`,
  );

  console.log(`${C.brightBlack}›${C.reset} ${message}`);
}

function printAssistant(message: string) {
  console.log();

  console.log(
    `${C.brightBlue}${C.bold}Assistant${C.reset} ${C.brightBlack}· ${timestamp()}${C.reset}`,
  );

  printLine("─", Math.min(40, termWidth()));

  for (const line of message.split("\n")) {
    console.log(`${C.white}${line}${C.reset}`);
  }

  console.log();
}

// ============================================================
// FRIENDLY TOOL DISPLAY
// ============================================================

function printToolCall(name: string) {
  const meta = TOOL_META[name] ?? {
    icon: "🛠️",
    label: name,
    color: C.brightMagenta,
  };

  console.log();

  console.log(`${meta.color}${C.bold}${meta.icon} ${meta.label}${C.reset}`);
}

// ============================================================
// HELP
// ============================================================

function printHelp() {
  console.log();

  box(
    [
      `${C.brightYellow}${C.bold}Commands${C.reset}`,

      "",

      `${C.yellow}/help${C.reset}   Show available commands`,

      `${C.yellow}/clear${C.reset}  Clear conversation`,

      `${C.yellow}/exit${C.reset}   Exit application`,
    ],
    {
      color: C.brightYellow,
    },
  );

  console.log();
}

// ============================================================
// SPINNER
// ============================================================

const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

function startThinkingSpinner() {
  let frame = 0;

  process.stdout.write("\n");

  const interval = setInterval(() => {
    const spin = SPINNER_FRAMES[frame % SPINNER_FRAMES.length];

    frame++;

    process.stdout.write(
      `\r${C.brightMagenta}${C.bold}${spin} Assistant${C.reset} ${C.dim}is thinking${C.reset}   `,
    );
  }, 80);

  return () => {
    clearInterval(interval);

    process.stdout.write("\r" + " ".repeat(termWidth()) + "\r");
  };
}

// ============================================================
// ERROR
// ============================================================

function printError(message: string) {
  console.log();

  box(
    [
      `${C.brightRed}${C.bold}✗ Error${C.reset}`,

      "",

      `${C.red}${message}${C.reset}`,
    ],
    {
      color: C.brightRed,
    },
  );

  console.log();
}

// ============================================================
// CHAT
// ============================================================

async function main() {
  printHeader();

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: true,

    prompt: `${C.brightGreen}${C.bold}You${C.reset} ${C.brightBlack}›${C.reset} `,
  });

  let messages: (typeof MessagesAnnotation.State)["messages"] = [];

  const askQuestion = () =>
    new Promise<string>((resolve) => {
      rl.question(
        `\n${C.brightGreen}${C.bold}You${C.reset} ${C.brightBlack}›${C.reset} `,
        resolve,
      );
    });

  while (true) {
    const input = (await askQuestion()).trim();

    // --------------------------------------------------------
    // EMPTY INPUT
    // --------------------------------------------------------

    if (!input) {
      continue;
    }

    // --------------------------------------------------------
    // HELP
    // --------------------------------------------------------

    if (input.toLowerCase() === "/help") {
      printHelp();
      continue;
    }

    // --------------------------------------------------------
    // CLEAR
    // --------------------------------------------------------

    if (input.toLowerCase() === "/clear") {
      messages = [];

      printHeader();

      console.log(`${C.brightGreen}✓${C.reset} Conversation cleared.`);

      continue;
    }

    // --------------------------------------------------------
    // EXIT
    // --------------------------------------------------------

    if (
      input.toLowerCase() === "/exit" ||
      input.toLowerCase() === "/quit" ||
      input.toLowerCase() === "/q"
    ) {
      console.log();

      console.log(`${C.brightCyan}👋 Goodbye!${C.reset}`);

      console.log();

      rl.close();

      break;
    }

    // --------------------------------------------------------
    // USER MESSAGE
    // --------------------------------------------------------

    printUser(input);

    messages.push(
      new HumanMessage({
        id: randomUUID(),
        content: input,
      }),
    );

    // --------------------------------------------------------
    // THINKING
    // --------------------------------------------------------

    const stopSpinner = startThinkingSpinner();

    // --------------------------------------------------------
    // RUN GRAPH
    // --------------------------------------------------------

    try {
      const result = await app.invoke({
        messages,
      });

      stopSpinner();

      // ------------------------------------------------------
      // KEEP CONVERSATION
      // ------------------------------------------------------

      messages = result.messages;

      // ------------------------------------------------------
      // LAST MESSAGE
      // ------------------------------------------------------

      const lastMessage = result.messages[result.messages.length - 1];

      let content: string;

      if (typeof lastMessage?.content === "string") {
        content = lastMessage.content;
      } else {
        content = JSON.stringify(lastMessage?.content, null, 2);
      }

      printAssistant(content);
    } catch (error) {
      stopSpinner();

      printError(error instanceof Error ? error.message : String(error));
    }
  }
}

// ============================================================
// START
// ============================================================

main().catch((error) => {
  console.error(`${C.brightRed}Fatal error:${C.reset}`, error);

  process.exit(1);
});
