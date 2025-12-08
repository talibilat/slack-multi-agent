import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from slack_bolt.adapter.socket_mode import SocketModeHandler
from src.slack_bot import app

def main():
    bot_token = os.environ.get("SLACK_BOT_TOKEN")
    app_token = os.environ.get("SLACK_APP_TOKEN")
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")

    if not bot_token or not app_token:
        print("ERROR: SLACK_BOT_TOKEN or SLACK_APP_TOKEN not found in .env")
        return

    if not azure_key:
        print("ERROR: AZURE_OPENAI_API_KEY not found in .env")
        return

    print("Starting IT Support Agent...")
    
    # Verify Identity
    try:
        identity = app.client.auth_test()
        bot_user_id = identity["user_id"]
        bot_user_name = identity["user"]
        print(f"✅ Authenticated as: {bot_user_name} (ID: {bot_user_id})")
        print(f"👉 Please make sure you are messaging THIS user.")
    except Exception as e:
        print(f"❌ Auth failed: {e}")
        return

    handler = SocketModeHandler(app, app_token)
    handler.start()

if __name__ == "__main__":
    main()
