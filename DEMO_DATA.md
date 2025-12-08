# IT Support Agent - Demo Data

## Users
The following users are pre-configured in `data/users.json` for testing RBAC:

| User ID | Email | Roles | Capabilities |
|Str|Str|Str|Str|
| `U12345` | alice@example.com | `["employee"]` | Can reset own password, request standard software. |
| `U67890` | bob@example.com | `["employee", "it_support"]` | Can reset ANY password, provision sensitive apps (e.g. production_db). |
| `U99999` | eve@contractor.com | `["contractor"]` | Limited access. |

## Policies (RAG Data)
The `data/employee_handbook.txt` contains policies on:
- Password complexity & expiration
- Wi-Fi access (Secure vs Guest)
- Remote Access (VPN)
- Software Requests flow
- Security Incident reporting
