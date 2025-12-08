from typing import TypedDict, Annotated, Sequence, Dict, Optional, List, Literal
import operator
import os

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.graph import MessagesState

# Internal imports
from src.tools import provision_access_tool, reset_password_tool, USER_DB
from src.rag import get_retriever

# --- State Definition ---
# Blog uses MessagesState extended with custom keys
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_id: str
    route: Optional[str]
    target_email: Optional[str]
    app_name: Optional[str]
    retrieved_text: Optional[str]
    action_result: Optional[Dict]

# --- Components ---
llm = AzureChatOpenAI(
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# --- Nodes ---

def analyze_query(state: AgentState):
    """
    Analyze user message and decide which path to take.
    Simple rule-based classifier for the demo, as per blog.
    """
    print("--- 1. ANALYZE QUERY ---")
    user_text = state["messages"][-1].content.lower()
    decision = {}
    
    # Get user email for defaults
    user_info = USER_DB.get(state["user_id"], {})
    user_email = user_info.get("email", "")

    if "password" in user_text and "reset" in user_text:
        decision["route"] = "reset_password"
        # Naive extraction for demo: assuming 'for email@domain.com'
        # In reality an LLM extraction is better, but here we strictly follow the blog's idea
        words = user_text.split()
        target_email = user_email # Default to self
        for word in words:
            if "@" in word:
                target_email = word
        decision["target_email"] = target_email
        print(f"Decision: Reset Password for {target_email}")
        
    elif "access" in user_text or "license" in user_text or "provision" in user_text:
        decision["route"] = "provision_access"
        # Naive extraction: assume app name is often the 2nd or last word? 
        # Let's clean up logic slightly better than pure split:
        # "access to production_db" -> production_db
        words = user_text.split()
        app_name = ""
        if "access to" in user_text:
             app_name = user_text.split("access to")[-1].strip().split()[0]
        elif "license for" in user_text:
             app_name = user_text.split("license for")[-1].strip().split()[0]
        else:
             app_name = words[-1] # Fallback
        
        # Remove punctuation
        app_name = app_name.strip(".,?!")
        decision["app_name"] = app_name
        print(f"Decision: Provision Access to {app_name}")
        
    else:
        decision["route"] = "knowledge"
        print("Decision: Knowledge Query")
        
    return decision

def retrieve_info(state: AgentState):
    """Use the retriever to get relevant text for answering."""
    print("--- 2. RETRIEVE INFO ---")
    query = state["messages"][-1].content
    retriever = get_retriever()
    docs = retriever.invoke(query)
    
    if docs:
        state["retrieved_text"] = "\n\n".join([d.page_content for d in docs])
    else:
        state["retrieved_text"] = ""
    return {"retrieved_text": state["retrieved_text"]}

def answer_user(state: AgentState):
    """Use LLM to generate answer, possibly with context."""
    print("--- 3. ANSWER USER ---")
    user_question = state["messages"][-1].content
    context = state.get("retrieved_text", "")
    
    system_prompt = "You are a helpful IT support assistant. Answer the question using the provided context if relevant."
    
    if context:
        prompt = f"{system_prompt}\n\nContext: {context}\n\nUser: {user_question}"
    else:
        prompt = f"{system_prompt}\n\nUser: {user_question}"
        
    response = llm.invoke(prompt)
    # Return as list to append
    return {"messages": [response]}

def perform_action(state: AgentState):
    """Dispatch to the appropriate tool based on route."""
    print("--- 4. PERFORM ACTION ---")
    route = state.get("route")
    result = {}
    
    if route == "reset_password":
        tool_state = { 
            "requester": state["user_id"],
            "target_email": state.get("target_email", "") 
        }
        result = reset_password_tool(tool_state)
        
    elif route == "provision_access":
        tool_state = { 
            "requester": state["user_id"], 
            "app_name": state.get("app_name", "") 
        }
        result = provision_access_tool(tool_state)
        
    return {"action_result": result}

def format_action_result(state: AgentState):
    print("--- 5. FORMAT RESULT ---")
    res = state.get("action_result", {})
    if not res:
        reply = "I'm sorry, I couldn't perform the requested action."
    elif res.get("result") == "success":
        reply = f"✅ {res.get('message', 'Done.')}"
    else:
        reply = f"⚠️ {res.get('message', 'Action failed.')}"
        
    return {"messages": [AIMessage(content=reply)]}

# --- Graph Construction ---
workflow = StateGraph(AgentState)

workflow.add_node("analyze_query", analyze_query)
workflow.add_node("retrieve_info", retrieve_info)
workflow.add_node("answer_user", answer_user)
workflow.add_node("perform_action", perform_action)
workflow.add_node("format_result", format_action_result)

workflow.add_edge(START, "analyze_query")

# Conditional logic for analyze_query
def route_check(state):
    return state["route"]

workflow.add_conditional_edges(
    "analyze_query",
    route_check,
    {
        "knowledge": "retrieve_info",
        "reset_password": "perform_action",
        "provision_access": "perform_action"
    }
)

# Knowledge path
workflow.add_edge("retrieve_info", "answer_user")
workflow.add_edge("answer_user", END)

# Action path
workflow.add_edge("perform_action", "format_result")
workflow.add_edge("format_result", END)

app = workflow.compile()
