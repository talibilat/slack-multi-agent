import os
from slack_bolt import App
from langchain_core.messages import HumanMessage
from src.graph import app as graph_app

# Initialize Slack App
# Note: In production, verify tokens are present.
app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET") 
)

@app.event("message")
def handle_message_events(event, say, logger):
    try:
        user_id = event.get("user")
        text = event.get("text", "")
        channel_id = event.get("channel")
        
        # Skip bot's own messages (loop prevention)
        # Note: Bolt often handles this, but explicit check is good
        if not user_id or user_id == os.environ.get("BOT_USER_ID"):
            return

        # Send placeholder
        response = say("🤖💭 *(Thinking...)*")
        placeholder_ts = response["ts"]
        
        # Prepare state
        input_state = {
            "messages": [HumanMessage(content=text)],
            "user_id": user_id
        }
        
        # Invoke Agent
        output_state = graph_app.invoke(input_state)
        
        # Extract assistant reply
        # The graph appends the answer as the last message
        assistant_reply = None
        if output_state["messages"]:
            last_msg = output_state["messages"][-1]
            if last_msg.type == "ai":
                assistant_reply = last_msg.content
        
        if assistant_reply:
            # Update the placeholder
            app.client.chat_update(
                channel=channel_id,
                ts=placeholder_ts,
                text=assistant_reply
            )
        else:
            # Fallback update
            app.client.chat_update(
                channel=channel_id,
                ts=placeholder_ts,
                text="⚠️ I’m sorry, I couldn’t process that request."
            )
            
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        # Try to clean up placeholder if possible, or just post error
        say("❌ An error occurred while processing your request.")

@app.event("app_mention")
def handle_app_mention_events(event, say, logger):
    # Reuse handling logic
    handle_message_events(event, say, logger)

