# Troubleshooting: Slack Connection Issues

If you see **"Bolt app is running!"** but get **no response** and **no logs** when you message the bot, check these common issues.

## 1. Socket Mode is OFF (Most Likely)
If Socket Mode is disabled, Slack tries to send events to a URL (which you don't have).
1. Go to **[Slack API Apps](https://api.slack.com/apps)**.
2. Click your app.
3. Click **Socket Mode** in the sidebar.
4. Ensure the **"Enable Socket Mode"** toggle is **ON**. (It must be green).

## 2. App not Reinstalled
If you added permissions (like `message.im`) but didn't reinstall, they aren't active.
1. Click **OAuth & Permissions** in the sidebar.
2. Look for a yellow banner at the top saying "You've changed scopes...".
3. Click **"Reinstall to Workspace"**.

## 3. Token Mismatch (Common)
You might be using an **App Token (`xapp`)** from one app and a **Bot Token (`xoxb`)** from another.
1. Verify that both tokens in your `.env` file come from the **SAME** app in the Slack dashboard.

## 4. Wrong Bot
1. Ensure you are sending a DM to the exact bot name that matches your app. 
2. In the screenshot, you are DMing **"IT Support"**. Ensure your App Name in the dashboard is "IT Support".

## 5. Event Subscription Disabled
1. Click **Event Subscriptions** in the sidebar.
2. Ensure **"Enable Events"** is **ON**.
3. Ensure `message.im` is listed under **"Subscribe to bot events"**.
