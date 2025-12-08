import os
import sys

# Mock env vars to prevent crashes on import
os.environ["OPENAI_API_KEY"] = "sk-dummy"
os.environ["SLACK_BOT_TOKEN"] = "xoxb-dummy"
os.environ["AZURE_OPENAI_API_KEY"] = "dummy-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://dummy.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-05-15"

try:
    from src.graph import app
    print("Graph compiled successfully!")
except Exception as e:
    print(f"Graph compilation failed: {e}")
    sys.exit(1)
