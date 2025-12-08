from langchain_core.tools import tool

@tool
def provision_access(software_name: str, employee_email: str) -> str:
    """
    Use this tool to grant software licenses to employees.
    Useful for requests like 'I need access to Jira' or 'Give me a Zoom license'.
    
    Args:
        software_name: The name of the software (e.g. Jira, Zoom, Notion)
        employee_email: The email of the employee requesting access
    """
    # In a real app, this hits the Okta API. 
    # For the demo, we log it or update a local DB.
    print(f"[MOCK TOOL] Provisioning {software_name} for {employee_email}")
    return f"SUCCESS: Provisioning ticket created for {software_name} for user {employee_email}."

@tool
def reset_password(employee_email: str) -> str:
    """
    Use this to initiate a password reset workflow.
    Useful when a user says 'I forgot my password' or 'reset my password'.
    
    Args:
        employee_email: The email of the employee
    """
    print(f"[MOCK TOOL] Resetting password for {employee_email}")
    return f"SUCCESS: Password reset link sent to {employee_email}. Please check your inbox."
