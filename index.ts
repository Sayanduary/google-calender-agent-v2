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
// TOOLS
// ============================================================

const tools = [createEventTool, getEventsTool, deleteEventTool];

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
  Delete a calendar event using its exact event ID.
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

const toolNode = new ToolNode(tools);

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
// TERMINAL UI
// ============================================================

function printHeader() {
  console.clear();

  console.log();
  console.log("╭────────────────────────────────────────────────────────────╮");
  console.log("│                                                            │");
  console.log("│              Google Calendar Assistant                    │");
  console.log("│                                                            │");
  console.log("│     Ask about meetings, create events, find events...      │");
  console.log("│                                                            │");
  console.log("│     /help       commands                                  │");
  console.log("│     /clear      clear conversation                        │");
  console.log("│     /exit       quit                                      │");
  console.log("│                                                            │");
  console.log("╰────────────────────────────────────────────────────────────╯");
  console.log();
}

function printUser(message: string) {
  console.log();
  console.log(`You: ${message}`);
}

function printAssistant(message: string) {
  console.log();
  console.log("Assistant");
  console.log("─".repeat(60));

  for (const line of message.split("\n")) {
    console.log(line);
  }

  console.log();
}

function printHelp() {
  console.log();
  console.log("Commands");
  console.log("─".repeat(40));
  console.log("/help   Show available commands");
  console.log("/clear  Clear conversation history");
  console.log("/exit   Exit the application");
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
  });

  /**
   * Persistent conversation history.
   *
   * This allows:
   *
   * User:
   *   Create a meeting with Rahul tomorrow.
   *
   * Assistant:
   *   What time?
   *
   * User:
   *   5 PM.
   *
   * Assistant:
   *   ...
   */
  let messages: (typeof MessagesAnnotation.State)["messages"] = [];

  const askQuestion = () =>
    new Promise<string>((resolve) => {
      rl.question("\nYou: ", resolve);
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

    if (input === "/help") {
      printHelp();
      continue;
    }

    // --------------------------------------------------------
    // CLEAR
    // --------------------------------------------------------

    if (input === "/clear") {
      messages = [];

      console.log();
      console.log("Conversation cleared.");
      console.log();

      continue;
    }

    // --------------------------------------------------------
    // EXIT
    // --------------------------------------------------------

    if (input === "/exit" || input === "/quit" || input === "/q") {
      console.log();
      console.log("Goodbye!");
      console.log();

      rl.close();

      break;
    }

    // --------------------------------------------------------
    // SHOW USER MESSAGE
    // --------------------------------------------------------

    printUser(input);

    // --------------------------------------------------------
    // ADD USER MESSAGE
    // --------------------------------------------------------

    messages.push(
      new HumanMessage({
        id: randomUUID(),
        content: input,
      }),
    );

    // --------------------------------------------------------
    // THINKING INDICATOR
    // --------------------------------------------------------

    process.stdout.write("\nAssistant is thinking");

    let dots = 0;

    const spinner = setInterval(() => {
      dots = (dots + 1) % 4;

      process.stdout.write("\rAssistant is thinking" + ".".repeat(dots));
    }, 350);

    // --------------------------------------------------------
    // RUN GRAPH
    // --------------------------------------------------------

    try {
      const result = await app.invoke({
        messages,
      });

      clearInterval(spinner);

      // Clear thinking indicator
      process.stdout.write("\r" + " ".repeat(40) + "\r");

      // ------------------------------------------------------
      // KEEP FULL CONVERSATION
      // ------------------------------------------------------

      messages = result.messages;

      // ------------------------------------------------------
      // GET LAST AI MESSAGE
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
      clearInterval(spinner);

      process.stdout.write("\r" + " ".repeat(40) + "\r");

      console.log();
      console.log(
        "Error:",
        error instanceof Error ? error.message : String(error),
      );
      console.log();
    }
  }
}

// ============================================================
// START APPLICATION
// ============================================================

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
