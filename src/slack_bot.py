import os
from slack_bolt import App
from langchain_core.messages import HumanMessage
from src.graph import app as graph_app

# Initialize Slack App
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

def process_and_reply(text: str, user_id: str, say):
    """
    Invokes the LangGraph agent and replies to the user.
    """
    print(f"Processing message from {user_id}: {text}")
    
    initial_state = {"messages": [HumanMessage(content=text)]}
    
    # Run the graph
    try:
        result = graph_app.invoke(initial_state)
        last_message = result["messages"][-1]
        response_text = last_message.content
        say(response_text)
    except Exception as e:
        print(f"Error processing message: {e}")
        say(f"Sorry, I encountered an error: {e}")

@app.event("app_mention")
def handle_app_mention(event, say):
    """
    Listens to @mentions (e.g. @ITSupport Help me)
    """
    text = event.get("text")
    user = event.get("user")
    process_and_reply(text, user, say)

@app.message(".*")
def handle_message(message, say):
    """
    Listens to DMs and Channel messages (requires message.im / message.channels scopes)
    """
    text = message.get("text")
    user = message.get("user")
    
    # Avoid bot replying to itself if not handled by Bolt automatically
    if message.get("bot_id") is None:
        process_and_reply(text, user, say)
