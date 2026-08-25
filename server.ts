import express from "express";
import { google } from "googleapis";
import dotenv from "dotenv";
dotenv.config();

const app = express();
const PORT = 3600;

export const oauth2Client = new google.auth.OAuth2(
  process.env.GOOGLE_CLIENT_ID,
  process.env.CLIENT_SECRET,
  process.env.GOOGLE_REDIRECT_URL,
);

app.get("/auth", (req, res) => {
  //generate the link
  const scopes = ["https://www.googleapis.com/auth/calendar"];
  const url = oauth2Client.generateAuthUrl({
    access_type: "offline",
    scope: scopes,
    prompt: "consent",
  });
  res.redirect(url);
});

app.get("/callback", async (req, res) => {
  const code = req.query.code as string;
  // exchange code with access token / refresh token
  const { tokens } = await oauth2Client.getToken(code);
  console.log(tokens);
  res.send("Google Account Connected !.You Can Close this tab now");
});

app.listen(PORT, () => {
  console.log(`Server Running on ${PORT}`);
});
