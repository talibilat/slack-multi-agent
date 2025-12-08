import os
import dotenv

# Load environment variables first
dotenv.load_dotenv()

from langchain_core.messages import HumanMessage
from src.graph import app

def run_test(query: str, user_id: str = "U123456"):
    print(f"\n\n>>> Testing Query: '{query}' (User: {user_id})")
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "user_id": user_id
    }
    try:
        result = app.invoke(initial_state)
        last_message = result["messages"][-1]
        print(f"Final Response: {last_message.content}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        # print("ERROR: AZURE_OPENAI_API_KEY not found in environment variables.")
        # print("Please create a .env file with your key.")
        # exit(1)
        pass # Allow running even if keys missing to see code path errors (it will fail at LLM call)

    print("--- Starting Agent Verification ---")
    
    # Test 1: RAG (Knowledge Retrieval) - All users can access
    run_test("What is the guest wifi password?", user_id="U12345")
    
    # Test 2: Employee requesting standard software (Allowed)
    run_test("I need a license for Jira for myself (alice@example.com)", user_id="U12345")
    
    # Test 3: Employee requesting SENSITIVE software (Should Fail)
    run_test("I need access to production_db", user_id="U12345")
    
    # Test 4: IT Support provisioning SENSITIVE software (Allowed)
    run_test("Provision access to production_db for alice@example.com", user_id="U67890")
    
    # Test 5: IT Support resetting OTHER password (Allowed)
    run_test("Reset password for alice@example.com", user_id="U67890")
    
    # Test 6: Employee resetting OTHER password (Should Fail)
    run_test("Reset password for bob@example.com", user_id="U12345")

    print("\n--- Verification Complete ---")
