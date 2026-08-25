import { tool } from "@langchain/core/tools";
import { google } from "googleapis";
import z from "zod";
import { oauth2Client } from "./server";
import tokens from "./tokens.json";

const calendar = google.calendar({ version: "v3", auth: oauth2Client });

oauth2Client.setCredentials(tokens);

const getEventSchema = z.object({
  q: z
    .string()
    .describe(
      "Free-text search query. Searches event summary, description, location, attendee names/emails, and organizer names/emails.",
    ),
  timeMin: z
    .string()
    .describe("The start datetime in RFC3339 format, preferably UTC."),
  timeMax: z
    .string()
    .describe("The end datetime in RFC3339 format, preferably UTC."),
});

type Params = z.infer<typeof getEventSchema>;

export const getEventsTool = tool(
  async (params) => {
    /**
     * timeMin
     * timeMax
     * q
     */

    const { q, timeMin, timeMax } = params as Params;

    try {
      const response = await calendar.events.list({
        calendarId: "primary",
        q,
        timeMin,
        timeMax,
        singleEvents: true,
        orderBy: "startTime",
      });

      const result = response.data.items?.map((event) => {
        return {
          id: event.id,
          summary: event.summary,
          status: event.status,
          organiser: event.organizer,
          start: event.start,
          end: event.end,
          attendees: event.attendees,
          meetingLink: event.hangoutLink,
          eventType: event.eventType,
        };
      });

      return JSON.stringify(result ?? []);
    } catch (error) {
      console.log("Error: ", error);
      return "Failed To Fetch the Calendar Events";
    }
  },
  {
    name: "get-events",
    description:
      "Get calendar events within a specified time range. The query searches event summary, description, location, attendees, and organizer information.",
    schema: getEventSchema,
  },
);

type Attendee = {
  email: string;
  displayName?: string;
};

// type EventData = {
//   summary: string;
//   start: {
//     dateTime: string;
//     timeZone?: string;
//   };
//   end: {
//     dateTime: string;
//     timeZone?: string;
//   };
//   attendees?: Attendee[];
// };

const createEventSchema = z.object({
  summary: z.string().describe("The title or summary of the calendar event."),
  start: z.object({
    dateTime: z
      .string()
      .describe("The event start datetime in RFC3339 format."),
    timeZone: z
      .string()
      .optional()
      .describe("The timezone of the event start in UTC format."),
  }),
  end: z.object({
    dateTime: z.string().describe("The event end datetime in UTC format."),
    timeZone: z
      .string()
      .optional()
      .describe("The timezone of the event end in UTC format."),
  }),
  attendees: z
    .array(
      z.object({
        email: z.string(),
        displayName: z.string().optional(),
      }),
    )
    .optional()
    .describe("List of attendees to invite to the event."),
});

type EventData = z.infer<typeof createEventSchema>;

export const createEventTool = tool(
  async (eventData) => {
    const { summary, start, end, attendees } = eventData as EventData;

    try {
      const response = await calendar.events.insert({
        calendarId: "primary",
        sendUpdates: "all",
        conferenceDataVersion: 1,
        requestBody: {
          summary,
          start,
          end,
          attendees,
          conferenceData: {
            createRequest: {
              requestId: crypto.randomUUID(),
              conferenceSolutionKey: {
                type: "hangoutsMeet",
              },
            },
          },
        },
      });

      console.log("response: ", response);

      return JSON.stringify({
        success: true,
        id: response.data.id,
        summary: response.data.summary,
        start: response.data.start,
        end: response.data.end,
        attendees: response.data.attendees,
        meetingLink:
          response.data.hangoutLink ??
          response.data.conferenceData?.entryPoints?.find(
            (entry) => entry.entryPointType === "video",
          )?.uri,
      });
    } catch (error) {
      console.log("Error: ", error);

      return JSON.stringify({
        success: false,
        message: "Failed to create calendar event",
      });
    }
  },
  {
    name: "create-event",
    description:
      "Create a Google Calendar event with a summary, start time, end time, and optional attendees.",
    schema: createEventSchema,
  },
);
