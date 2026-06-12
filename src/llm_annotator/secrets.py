"""Load API keys from a shared private Google Sheet.

The sheet has two columns: key_name, key_value.
Team members with read access never need to copy/paste keys.
Falls back to Colab Secrets if the sheet isn't configured.
"""

import os


def load_api_keys(gc, secrets_sheet_id: str = ""):
    """Load API keys into os.environ from a shared Google Sheet.

    Sheet format (tab "Keys"):
        | key_name          | key_value     |
        | OPENAI_API_KEY    | sk-...        |
        | ANTHROPIC_API_KEY | sk-ant-...    |
        | GEMINI_API_KEY    | AI...         |

    Falls back to Colab userdata if secrets_sheet_id is empty.
    """
    if secrets_sheet_id:
        _load_from_sheet(gc, secrets_sheet_id)
    else:
        _load_from_colab_secrets()


def _load_from_sheet(gc, sheet_id: str):
    ws = gc.open_by_key(sheet_id).sheet1
    rows = ws.get_all_values()
    loaded = 0
    for row in rows[1:]:
        if len(row) >= 2 and row[0].strip() and row[1].strip():
            os.environ[row[0].strip()] = row[1].strip()
            loaded += 1
    print(f"Loaded {loaded} API key(s) from shared sheet.")


def _load_from_colab_secrets():
    try:
        from google.colab import userdata
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]:
            try:
                os.environ[key] = userdata.get(key)
            except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
                pass
        print("Loaded API keys from Colab Secrets.")
    except ImportError:
        print("Not in Colab and no secrets sheet configured — set API keys manually.")
