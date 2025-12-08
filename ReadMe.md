# Setup Guide: API Keys & Credentials

To run the IT Support Agent, you need to configure the `.env` file with credentials for OpenAI and Slack.

## 1. OpenAI API Key
**Variable**: `OPENAI_API_KEY`

1. Go to [OpenAI Platform](https://platform.openai.com/api-keys).
2. Click **"Create new secret key"**.
3. Name it (e.g., "IT Agent").
4. Copy the key (starts with `sk-...`).

---

## 2. Slack Configuration
**Variables**: `SLACK_APP_TOKEN`, `SLACK_BOT_TOKEN`

You need to create a Slack App and install it in your workspace.

### Step A: Create the App
1. Go to [Slack API: Your Apps](https://api.slack.com/apps).
2. Click **"Create New App"**.
3. Select **"From scratch"**.
4. Name it "IT Support Agent" and select your workspace.

### Step B: Enable Socket Mode (For App Token)
*Socket Mode allows the bot to run behind a firewall without public URLs.*

1. In the left sidebar, click **Socket Mode**.
2. Toggle **"Enable Socket Mode"**.
3. Generate an App-Level Token:
   - **Token Name**: `socket-token`
   - **Scopes**: `connections:write` (should be selected automatically).
4. **Copy the `xapp-...` token**. This is your `SLACK_APP_TOKEN`.

### Step C: Configure Permissions (For Bot Token)
1. In the left sidebar, click **OAuth & Permissions**.
2. Scroll down to **Scopes** -> **Bot Token Scopes**.
3. Add the following scopes:
   - `app_mentions:read` (To hear when you @mention it)
   - `chat:write` (To reply to messages)
   - `im:history` (Optional: if you want it to work in DMs)
   - `users:read` (To see user emails for the mock tools)

### Step D: Subscribe to Events
1. In the left sidebar, click **Event Subscriptions**.
2. Toggle **"Enable Events"**.
3. Expand **"Subscribe to bot events"**.
4. Add `app_mention`.
5. **Critical**: Add `message.im` (To allow the bot to reply to Direct Messages without being mentioned).
6. Click **"Save Changes"** (at the bottom).

### Step E: Install & Get Token
1. Go back to **OAuth & Permissions**.
2. Scroll to the top and click **"Install to Workspace"**.
3. Allow the permissions.
4. **Copy the `xoxb-...` token**. This is your `SLACK_BOT_TOKEN`.

---

## 3. Final Step
Paste these values into your `.env` file:

```env
OPENAI_API_KEY=sk-...
SLACK_APP_TOKEN=xapp-...
SLACK_BOT_TOKEN=xoxb-...
```

Save the file and run `python main.py`.
