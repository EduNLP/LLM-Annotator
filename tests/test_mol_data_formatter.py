"""
Unit tests for MOL data formatting: obsid → Tracker row → deidentified transcript.

Usage:
    # Locally (parse-only, no auth):
    python tests/test_mol_data_formatter.py --local

    # In Colab (hits real Tracker sheet):
    python tests/test_mol_data_formatter.py --obsid 241
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TRACKER_SHEET_ID = "1d9iWJzY8GXh1UWJEEYBn9pQeYTuseA0Yqr4WMsTSy-k"
TRACKER_TAB = "Tracker"
EXPECTED_TRANSCRIPT_COLS = [
    "#", "segment", "in_cue", "out_cue", "duration", "speaker", "named_language", "dialogue"
]

_results = []


def report(name, status, detail=""):
    _results.append((name, status, detail))
    tag = {"PASS": "✓", "FAIL": "✗", "SKIP": "–"}[status]
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_tracker_row(ws_data, obsid: int):
    """Find the row in Tracker matching obsid (e.g. 241 matches 25-0241 or 26-0241)."""
    headers = ws_data[0]
    idx_col = next(i for i, h in enumerate(headers) if h.strip().lower() == "index")
    for row in ws_data[1:]:
        if idx_col < len(row):
            val = row[idx_col].strip()
            match = re.search(r"\d{2}-0*(\d+)", val)
            if match and int(match.group(1)) == obsid:
                return dict(zip(headers, row))
    return None


def extract_sheet_id_from_url(url: str) -> str:
    """Pull the sheet ID out of a Google Sheets/Drive URL."""
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    return m.group(1) if m else ""


def filter_dash_segments(rows: list[dict]) -> list[dict]:
    """Remove rows where segment column contains a dash."""
    return [r for r in rows if r.get("segment", "").strip() not in ("-", "—", "–", "")]


# ---------------------------------------------------------------------------
# Local-only tests (no auth needed)
# ---------------------------------------------------------------------------

def test_extract_sheet_id():
    url = "https://docs.google.com/spreadsheets/d/1ABCdef_ghiJKL/edit#gid=0"
    sid = extract_sheet_id_from_url(url)
    if sid == "1ABCdef_ghiJKL":
        report("extract_sheet_id from URL", "PASS")
    else:
        report("extract_sheet_id from URL", "FAIL", f"got {sid}")


def test_filter_dash_segments():
    rows = [
        {"#": "1", "segment": "a", "dialogue": "hello"},
        {"#": "2", "segment": "-", "dialogue": "skip me"},
        {"#": "3", "segment": "b", "dialogue": "world"},
        {"#": "4", "segment": "—", "dialogue": "also skip"},
        {"#": "5", "segment": "", "dialogue": "empty skip"},
    ]
    filtered = filter_dash_segments(rows)
    if len(filtered) == 2 and filtered[0]["#"] == "1" and filtered[1]["#"] == "3":
        report("filter_dash_segments", "PASS")
    else:
        report("filter_dash_segments", "FAIL", f"got {len(filtered)} rows: {filtered}")


def test_find_tracker_row_mock():
    mock_data = [
        ["Index", "Name", "Link to deidentified transcripts"],
        ["25-0241", "Obs A", "https://docs.google.com/spreadsheets/d/FAKE_ID/edit"],
        ["26-0100", "Obs B", "https://example.com"],
    ]
    row = find_tracker_row(mock_data, 241)
    if row and row["Index"] == "25-0241":
        report("find_tracker_row (mock, 241)", "PASS")
    else:
        report("find_tracker_row (mock, 241)", "FAIL", f"got {row}")

    row2 = find_tracker_row(mock_data, 999)
    if row2 is None:
        report("find_tracker_row (mock, missing)", "PASS")
    else:
        report("find_tracker_row (mock, missing)", "FAIL", f"expected None, got {row2}")


# ---------------------------------------------------------------------------
# Integration tests (need gspread auth)
# ---------------------------------------------------------------------------

def test_real_tracker_lookup(gc, obsid: int):
    print(f"\n  Fetching Tracker sheet for obsid {obsid}...")
    spreadsheet = gc.open_by_key(TRACKER_SHEET_ID)
    ws = spreadsheet.worksheet(TRACKER_TAB)
    data = ws.get_all_values()

    row = find_tracker_row(data, obsid)
    if row is None:
        report(f"find obs {obsid} in Tracker", "FAIL", "row not found")
        return
    report(f"find obs {obsid} in Tracker", "PASS", f"Index={row.get('Index', '?')}")

    link_col = None
    for key in row:
        if "deidentified" in key.lower() and "link" in key.lower():
            link_col = key
            break

    if not link_col or not row[link_col]:
        report("has deidentified transcript link", "FAIL", f"col={link_col}, val={row.get(link_col, 'MISSING')}")
        return
    report("has deidentified transcript link", "PASS", row[link_col][:60])

    transcript_sheet_id = extract_sheet_id_from_url(row[link_col])
    if not transcript_sheet_id:
        report("extract transcript sheet ID", "FAIL", f"from: {row[link_col]}")
        return
    report("extract transcript sheet ID", "PASS", transcript_sheet_id)

    print(f"  Fetching transcript sheet {transcript_sheet_id}...")
    try:
        t_ws = gc.open_by_key(transcript_sheet_id).sheet1
        t_data = t_ws.get_all_values()
    except Exception as e:
        report("open transcript sheet", "FAIL", str(e))
        return
    report("open transcript sheet", "PASS", f"{len(t_data)} rows")

    if len(t_data) < 2:
        report("transcript has data rows", "FAIL", "only header or empty")
        return

    headers = [h.strip().lower() for h in t_data[0]]
    expected_lower = [c.lower() for c in EXPECTED_TRANSCRIPT_COLS]
    missing = [c for c in expected_lower if c not in headers]
    if missing:
        report("transcript has expected columns", "FAIL", f"missing: {missing}; got: {headers}")
    else:
        report("transcript has expected columns", "PASS")

    transcript_rows = [dict(zip(t_data[0], r)) for r in t_data[1:]]
    before_count = len(transcript_rows)
    filtered = filter_dash_segments(transcript_rows)
    dash_count = before_count - len(filtered)
    report("filter dash segments", "PASS", f"kept {len(filtered)}/{before_count} rows (removed {dash_count} dash rows)")

    if filtered:
        sample = filtered[0]
        print(f"  Sample row: segment={sample.get('segment')}, speaker={sample.get('speaker')}, dialogue={sample.get('dialogue', '')[:60]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run parse-only tests (no auth)")
    parser.add_argument("--obsid", type=int, default=241, help="Observation ID to test")
    args = parser.parse_args()

    print("\n── MOL Data Formatter Tests ──\n")

    # Always run local tests
    test_extract_sheet_id()
    test_filter_dash_segments()
    test_find_tracker_row_mock()

    if not args.local:
        from google.auth import default
        import gspread
        creds, _ = default()
        gc = gspread.authorize(creds)
        test_real_tracker_lookup(gc, args.obsid)

    # Summary
    print(f"\n{'='*40}")
    passed = sum(1 for _, s, _ in _results if s == "PASS")
    failed = sum(1 for _, s, _ in _results if s == "FAIL")
    print(f"  {passed} passed, {failed} failed, {len(_results) - passed - failed} skipped")
    if failed:
        print("  FAILED:")
        for name, s, detail in _results:
            if s == "FAIL":
                print(f"    ✗ {name}: {detail}")
    print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
