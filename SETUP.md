# Setup Instructions

## Prerequisites
- Python 3.9+
- Azure OpenAI Service (Deployment for Chat and Embeddings)
- Slack App (with Socket Mode)

## System Overview

This agent is an **Internal IT Concierge** that runs on Slack. It is built using **LangGraph** for orchestration, **Azure OpenAI** for reasoning, and **ChromaDB** for knowledge retrieval (RAG).

The system consists of three main components:
1.  **Slack Bot (`src/slack_bot.py`)**: Handles the user interface, listening for messages and posting responses.
2.  **Agent Graph (`src/graph.py`)**: The brain of the agent. It decides whether to answer a question (using RAG) or perform an action (using Tools).
3.  **Tools (`src/tools.py`)**: Secure functions that perform real-world actions like password resets and access provisioning, with built-in RBAC checks.

## How it Works

The agent follows a structured workflow defined in `src/graph.py`:

1.  **Analyze Query**: The user's message is analyzed to classify the intent:
    *   **Knowledge Query**: General questions (e.g., "How do I setup VPN?").
    *   **Action Request**: Specific tasks (e.g., "Reset my password", "Grant access to Jira").
2.  **Route**:
    *   **Knowledge Path**: The agent uses `src/rag.py` to search the `data/employee_handbook.txt` for relevant policies, then generates an answer using the LLM.
    *   **Action Path**: The agent extracts parameters (like specific apps or emails) and calls the appropriate tool.
3.  **Tools & Security**:
    *   Tools in `src/tools.py` enforce Role-Based Access Control (RBAC).
    *   Example: An employee can only reset their *own* password, but IT Support can reset *anyone's*.
    *   Example: Access to sensitive apps (like `production_db`) is restricted to IT Support roles.
4.  **Response**: The result (answer or action confirmation) is formatted and sent back to Slack.

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Environment Variables:
   Copy `.env.example` to `.env` and fill in the values:
   ```bash
   AZURE_OPENAI_API_KEY=...
   AZURE_OPENAI_ENDPOINT=...
   AZURE_OPENAI_API_VERSION=...
   AZURE_OPENAI_DEPLOYMENT=...
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=...
   SLACK_BOT_TOKEN=xoxb-...
   SLACK_APP_TOKEN=xapp-...
   ```

## Running the Agent

Start the agent:
```bash
python main.py
```

## Running Verification

To test the agent offline (simulated):
```bash
python verify_agent.py
```

## Mock Data
- Users are defined in `data/users.json`.
- Policies are in `data/employee_handbook.txt`.
