from typing import TypedDict, Literal, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
import operator
import os

# Internal imports
from src.tools import provision_access, reset_password
from src.rag import get_retriever

# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    intent: str

# --- Components ---
# Using Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# Router
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "it_tools"] = Field(
        ...,
        description="Given a user question choose to route it to the 'vectorstore' for policy questions or 'it_tools' for actions like provisioning or resets.",
    )

# structured_llm_router = llm.with_structured_output(RouteQuery, method="function_calling")
# Fallback to manual binding for Azure legacy/custom model compatibility
router_tool = llm.bind_tools([RouteQuery], tool_choice="RouteQuery")

# --- Nodes ---

def router_node(state: AgentState):
    """
    Analyzes the user's last message and decides the route.
    """
    system_prompt = """You are an expert IT triage agent.
    You must decide if a user's request is a QUESTION about policy (route to vectorstore) 
    or an ACTION requiring IT tools (route to it_tools).
    
    Examples:
    "How do I reset my password?" -> vectorstore
    "I need a Jira license" -> it_tools
    "What is the wifi password?" -> vectorstore
    "Reset the password for bob@example.com" -> it_tools
    """
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    
    # Invoke with tool choice forced
    response = router_tool.invoke(messages)
    
    # Parse the tool call
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        # RouteQuery arguments
        args = tool_call["args"]
        datasource = args.get("datasource")
        if datasource:
            return {"intent": datasource}
            
    # Fallback default
    return {"intent": "vectorstore"}

def rag_node(state: AgentState):
    """
    Retrieves documents and answers the question.
    """
    last_message = state["messages"][-1]
    question = last_message.content
    
    retriever = get_retriever()
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful IT assistant. Use the following context to answer the user's question. If you don't know, say so.\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])
    
    chain = rag_prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return {"messages": [response]}

def tools_agent_node(state: AgentState):
    """
    The agent responsible for calling tools.
    """
    tools = [provision_access, reset_password]
    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# --- Conditional Edges ---

def route_conditional(state: AgentState):
    if state["intent"] == "vectorstore":
        return "rag_node"
    elif state["intent"] == "it_tools":
        return "tools_agent_node"
    else:
        return "rag_node" # Fallback

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# --- Graph Construction ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("router", router_node)
workflow.add_node("rag_node", rag_node)
workflow.add_node("tools_agent_node", tools_agent_node)
workflow.add_node("tools", ToolNode([provision_access, reset_password]))

# Add Edges
workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    route_conditional,
    {
        "rag_node": "rag_node",
        "tools_agent_node": "tools_agent_node"
    }
)

workflow.add_edge("rag_node", END)

workflow.add_edge("tools_agent_node", "tools") # Execute tool
workflow.add_edge("tools", END) # In a more complex loop, we might go back to agent. 
# But for 'do X' -> 'done', END is fine. 
# Actually, usually after tool execution, the agent should confirm.
# Let's verify: LLM -> ToolCall -> ToolNode -> ToolMessage -> LLM (to say "It's done").
# So I should add an edge back to tools_agent_node?
# Let's adjust: tools -> tools_agent_node.
# And tools_agent_node -> should_continue -> END if no more tools.

workflow.add_edge("tools", "tools_agent_node")
workflow.add_conditional_edges(
    "tools_agent_node",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

app = workflow.compile()
