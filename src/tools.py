import json
from typing import Dict, List, Optional

# Load users from JSON to simulate a database
def load_users():
    with open("data/users.json", "r") as f:
        users = json.load(f)
        # Convert list to dict keyed by user_id for O(1) access if needed, 
        # but the blog uses a direct dict lookup. Let's make it easy to lookup.
        user_db = {}
        for user in users:
            user_db[user["id"]] = user
        return user_db

# Initialize DB
USER_DB = load_users()

def reset_password_tool(state: Dict) -> Dict[str, str]:
    """
    Resets a password.
    state must contain: "requester" (user_id), "target_email"
    """
    user_id = state.get("requester")
    target_email = state.get("target_email")
    
    if not user_id or not target_email:
        return {"result": "error", "message": "Missing requester or target email."}

    info = USER_DB.get(user_id)
    if not info:
        return {"result": "error", "message": "User not found."}

    # Employee can only reset their own password; IT support can reset anyone's
    # Note: user["roles"] is a list like ["employee", "it_support"]
    user_roles = info.get("roles", [])
    
    # Check if target email belongs to requester
    is_self = info["email"].lower() == target_email.lower()
    
    if not is_self and "it_support" not in user_roles:
        return {"result": "error", "message": "Permission denied: You can only reset your own password."}
    
    return {"result": "success", "message": f"Password reset link sent to {target_email}."}

def provision_access_tool(state: Dict) -> Dict[str, str]:
    """
    Provisions access to an app.
    state must contain: "requester" (user_id), "app_name"
    """
    user_id = state.get("requester")
    app_name = state.get("app_name", "").lower()
    
    if not user_id or not app_name:
        return {"result": "error", "message": "Missing requester or app name."}
        
    info = USER_DB.get(user_id)
    if not info:
        return {"result": "error", "message": "User not found."}

    sensitive = {"production_db", "payroll_system"}
    user_roles = info.get("roles", [])

    if app_name in sensitive and "it_support" not in user_roles:
        return {"result": "error", "message": f"Access to {app_name} requires IT support role."}
    
    return {"result": "success", "message": f"Access to {app_name} granted for {info['email']}"}
