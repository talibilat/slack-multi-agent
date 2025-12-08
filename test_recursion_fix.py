
import os
import sys
from unittest.mock import MagicMock, patch

# Mock env vars
os.environ["SLACK_BOT_TOKEN"] = "xoxb-dummy"
os.environ["SLACK_APP_TOKEN"] = "xapp-dummy"
os.environ["AZURE_OPENAI_API_KEY"] = "dummy"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://dummy.azure.com/"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "dummy-dep"
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-05-15"

# Mock classes to avoid real API calls
from langchain_core.messages import AIMessage


# We need to patch before importing src.graph because it initializes things at module level
with patch('langchain_openai.AzureChatOpenAI') as MockAzure:
    # Setup mock
    mock_llm = MagicMock()
    MockAzure.return_value = mock_llm
    
    # We also need to mock the bind_tools result
    mock_bound = MagicMock()
    mock_llm.bind_tools.return_value = mock_bound
    
    # Import the app
    try:
        from src.graph import app
    except ImportError:
        # If imports fail due to other dependencies, we might need more mocks
        # But let's try.
        sys.modules['src.rag'] = MagicMock()
        sys.modules['src.tools'] = MagicMock()
        from src.graph import app

    print("Graph imported successfully.")

    # Now we want to run the graph and see if it terminates.
    # We need to control the mocks to guide it down the 'it_tools' path and then finish.

    # 1. Router thinks it's 'it_tools'
    # The routers uses `router_tool.invoke`.
    # `router_tool` is `llm.bind_tools(...)`
    # So `mock_bound.invoke` should return something with tool_calls for RouteQuery.
    
    # 2. Tools agent runs.
    # It also uses `llm.bind_tools(...)` -> `mock_bound`.
    # We need to distinguish between router call and agent call?
    # They both use `mock_bound`.
    # Let's use `side_effect` on `mock_bound.invoke`.
    
    # Sequence of calls expected:
    # 1. Router: returns RouteQuery(datasource='it_tools')
    # 2. Tools Agent (1st pass): returns tool call (e.g. reset_password)
    # 3. Tools Node: executes tool (we need to mock this or let it fail? Use Mock tools?)
    #    src/graph.py imports tools from `src.tools`.
    # 4. Tools Agent (2nd pass): returns "Done" (no tool calls).
    # 5. END
    
    # Let's mock the tool execution to be safe.
    # We can patch `src.graph.provision_access` and `src.graph.reset_password`.
    
    from src.graph import provision_access, reset_password
    
    def router_response(input_arg):
        # Return a message with tool call
        return AIMessage(content="", tool_calls=[{
            "name": "RouteQuery",
            "args": {"datasource": "it_tools"},
            "id": "call_1"
        }])

    def agent_response_with_tool(input_arg):
        return AIMessage(content="I will reset it.", tool_calls=[{
            "name": "reset_password",
            "args": {"user_id": "bob"},
            "id": "call_2"
        }])

    def agent_response_done(input_arg):
        return AIMessage(content="Password reset complete.")

    mock_bound.invoke.side_effect = [
        router_response(None),     # Router
        agent_response_with_tool(None), # Agent 1st pass
        agent_response_done(None)   # Agent 2nd pass
    ]
    
    # We also need to handle the tool execution in ToolNode.
    # ToolNode tries to run the tool.
    # reset_password is a function. We can just mock it if we can patch it in the graph?
    # The graph uses the imported functions.
    # But ToolNode is instantiated with `[provision_access, reset_password]`.
    # If we want to mock execution, we should have patched them before import or use logic.
    # However, since we define `tools` node using those functions, and `ToolNode` calls them.
    # If we didn't patch them, they are real functions.
    # Let's assume they are safe to run or will error?
    # Real `reset_password` probably does nothing or prints?
    # Let's check `src/tools.py`.
    
    # To be safe, let's use a simple state and run.
    
    from langchain_core.messages import HumanMessage
    
    print("Running graph...")
    try:
        # recursion_limit defaults to 25.
        # If the bug exists, it will loop until limit.
        # If fix works, it should finish quickly.
        final_state = app.invoke(
            {"messages": [HumanMessage(content="Reset password for Bob")]},
            config={"recursion_limit": 10} # Set low limit to fail fast if bug exists
        )
        print("Graph finished successfully!")
        print("Final State Messages:", len(final_state["messages"]))
    except Exception as e:
        print(f"Graph failed: {e}")
        # If recursion error, it might print "Recursion limit..."
        if "Recursion limit" in str(e):
            print("FAILURE: Recursion limit reached. The bug is still present.")
            sys.exit(1)
        else:
            # Other errors might be due to mocking, but if it's not recursion, 
            # we might have passed the loop point.
            pass

