"""Lightweight run lock using a Google Sheet tab.

Prevents two people from running experiments simultaneously.
Lock state lives in a "Lock" tab of the results sheet.
"""

import datetime
import gspread


_LOCK_TAB = "Lock"
_HEADERS = ["user", "started", "status"]


def _get_lock_ws(gc, sheet_id: str):
    spreadsheet = gc.open_by_key(sheet_id)
    try:
        ws = spreadsheet.worksheet(_LOCK_TAB)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=_LOCK_TAB, rows=5, cols=3)
        ws.update("A1:C1", [_HEADERS])
    return ws


def check_lock(gc, sheet_id: str) -> dict | None:
    """Return current lock info or None if unlocked."""
    ws = _get_lock_ws(gc, sheet_id)
    vals = ws.get_all_values()
    if len(vals) < 2:
        return None
    row = vals[1]
    if len(row) >= 3 and row[2] == "running":
        return {"user": row[0], "started": row[1]}
    return None


def acquire_lock(gc, sheet_id: str, user: str, force: bool = False) -> bool:
    """Try to acquire the run lock. Returns True if acquired."""
    existing = check_lock(gc, sheet_id)
    if existing and not force:
        print(f"⚠️  {existing['user']} is running since {existing['started']}")
        print("   Pass force=True to override, or wait for them to finish.")
        return False
    ws = _get_lock_ws(gc, sheet_id)
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    ws.update("A2:C2", [[user, now, "running"]])
    print(f"🔒 Lock acquired by {user} at {now}")
    return True


def release_lock(gc, sheet_id: str):
    """Release the run lock."""
    ws = _get_lock_ws(gc, sheet_id)
    ws.update("C2", [["done"]])
    print("🔓 Lock released.")
