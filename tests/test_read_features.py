"""
Test reading MOL Roles Features from the Google Sheet and building FEATURE_OPTIONS.

Source: https://docs.google.com/spreadsheets/d/1iIzzfXqq2nYMSbzDgu2wQJqnIIwwUY6r
Known tabs: "Conceptual" (gid=2088459072), "Discursives" (gid=2140603446).
New tabs (e.g. "Multimodal") may be added — tests dynamically read all tabs.

Usage:
    python tests/test_read_features.py --local       # mock-only tests
    python tests/test_read_features.py               # hits real sheet (Colab)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

FEATURES_SHEET_ID = "1iIzzfXqq2nYMSbzDgu2wQJqnIIwwUY6r"

# Known tabs — test verifies these exist but also accepts new ones
KNOWN_TABS = ["Conceptual", "Discursives"]

# Known features per tab — used to catch accidental deletions, not to restrict additions
KNOWN_CONCEPTUAL = [
    "Offtask", "Recording", "Directions", "Competent", "Language",
    "Coordinate", "Understanding", "Tool",
]
KNOWN_DISCURSIVES = [
    "claim", "reason", "agree", "disagree", "compare", "addon",
    "question", "revoice", "monitor", "nextstep", "redirect",
    "compliment", "apology",
]
KNOWN_FEATURES = KNOWN_CONCEPTUAL + KNOWN_DISCURSIVES

_results = []


def report(name, status, detail=""):
    _results.append((name, status, detail))
    tag = {"PASS": "✓", "FAIL": "✗", "SKIP": "–"}[status]
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))


def read_all_features(gc, sheet_id: str) -> tuple[dict[str, list[dict]], list[str]]:
    """Read all feature codes from every tab in the features sheet.

    Returns:
        (tab_features, all_tabs) where tab_features maps tab name →
        list of feature dicts, and all_tabs is the list of tab names.
    """
    spreadsheet = gc.open_by_key(sheet_id)
    all_tabs = [ws.title for ws in spreadsheet.worksheets()]

    tab_features = {}
    for tab_name in all_tabs:
        ws = spreadsheet.worksheet(tab_name)
        data = ws.get_all_values()
        if len(data) < 2:
            continue

        headers = data[0]
        code_idx = None
        for i, h in enumerate(headers):
            if h.strip().lower() == "code":
                code_idx = i
                break

        if code_idx is None:
            continue

        features = []
        for row in data[1:]:
            if code_idx < len(row) and row[code_idx].strip():
                row_dict = dict(zip(headers, row))
                features.append({
                    "code": row[code_idx].strip(),
                    "definition": row_dict.get("Definition", "").strip(),
                    "has_examples": bool(row_dict.get("example1", "").strip()),
                    "filter_if": row_dict.get("filter_if", "").strip(),
                    "linked_with": row_dict.get("linked_with", "").strip(),
                    "subcode_of": row_dict.get("subcode_of", "").strip(),
                    "extra_context_type": row_dict.get("extra_context_type", "").strip(),
                })
        if features:
            tab_features[tab_name] = features

    return tab_features, all_tabs


def build_feature_options(tab_features: dict[str, list[dict]]) -> list[str]:
    """Build a flat FEATURE_OPTIONS list from all tabs, preserving tab order."""
    options = []
    for tab_name, features in tab_features.items():
        for f in features:
            if f["code"] not in options:
                options.append(f["code"])
    return options


# ═══════════════════════════════════════════════════════════════════════════
# LOCAL TESTS
# ═══════════════════════════════════════════════════════════════════════════

def run_local_tests():
    print("\n── Local Tests ──\n")

    # ── 1. Known features list sanity ──
    assert len(KNOWN_FEATURES) > 0
    assert len(KNOWN_FEATURES) == len(set(KNOWN_FEATURES)), "duplicate in KNOWN_FEATURES"
    report("KNOWN_FEATURES valid (no dupes)", "PASS", f"{len(KNOWN_FEATURES)} features")

    # ── 2. Mock: parse features from multiple tabs ──
    mock_tabs = {
        "Conceptual": [
            {"code": "Offtask", "definition": "Off task", "has_examples": True,
             "filter_if": "", "linked_with": "", "subcode_of": "", "extra_context_type": ""},
            {"code": "Directions", "definition": "Following directions", "has_examples": True,
             "filter_if": "offtask", "linked_with": "Coordinate", "subcode_of": "", "extra_context_type": "activity_instructions"},
        ],
        "Discursives": [
            {"code": "claim", "definition": "Makes a claim", "has_examples": True,
             "filter_if": "", "linked_with": "", "subcode_of": "", "extra_context_type": ""},
        ],
        "Multimodal": [
            {"code": "gesture", "definition": "Uses gesture", "has_examples": True,
             "filter_if": "", "linked_with": "", "subcode_of": "", "extra_context_type": ""},
        ],
    }

    options = build_feature_options(mock_tabs)
    assert options == ["Offtask", "Directions", "claim", "gesture"]
    report("build_feature_options from mock tabs", "PASS", f"{options}")

    # ── 3. New tabs automatically included ──
    assert "gesture" in options, "new tab features should appear"
    report("new tabs (Multimodal) auto-included", "PASS")

    # ── 4. Feature rules parseable ──
    directions = mock_tabs["Conceptual"][1]
    filter_if = [c.strip() for c in directions["filter_if"].split(",") if c.strip()]
    linked = [c.strip() for c in directions["linked_with"].split(",") if c.strip()]
    assert filter_if == ["offtask"]
    assert linked == ["Coordinate"]
    assert directions["extra_context_type"] == "activity_instructions"
    report("feature rules parseable from mock", "PASS")

    # ── 5. No dupes across tabs ──
    all_codes = [f["code"] for feats in mock_tabs.values() for f in feats]
    assert len(all_codes) == len(set(all_codes)), "cross-tab duplicate"
    report("no cross-tab duplicate codes (mock)", "PASS")


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

def run_integration_tests():
    print("\n── Integration Tests (real features sheet) ──\n")

    from google.auth import default
    import gspread
    creds, _ = default()
    gc = gspread.authorize(creds)

    tab_features, all_tabs = read_all_features(gc, FEATURES_SHEET_ID)

    if not tab_features:
        report("read features from sheet", "FAIL", "no features found")
        return

    # ── 1. Known tabs exist ──
    for tab in KNOWN_TABS:
        if tab in tab_features:
            report(f"tab '{tab}' exists", "PASS", f"{len(tab_features[tab])} features")
        else:
            report(f"tab '{tab}' exists", "FAIL", f"tabs found: {list(tab_features.keys())}")

    # ── 2. Report any new tabs (future Multimodal etc.) ──
    new_tabs = [t for t in tab_features if t not in KNOWN_TABS]
    if new_tabs:
        for t in new_tabs:
            codes = [f["code"] for f in tab_features[t]]
            report(f"new tab '{t}' discovered", "PASS", f"features: {codes}")
    print(f"\n  All tabs with features: {list(tab_features.keys())}")

    # ── 3. Build FEATURE_OPTIONS ──
    options = build_feature_options(tab_features)
    print(f"\n  FEATURE_OPTIONS = {options}\n")
    report("build FEATURE_OPTIONS", "PASS", f"{len(options)} total")

    # ── 4. Known Conceptual features present ──
    conceptual_codes = [f["code"] for f in tab_features.get("Conceptual", [])]
    missing_c = [f for f in KNOWN_CONCEPTUAL if f not in conceptual_codes]
    new_c = [f for f in conceptual_codes if f not in KNOWN_CONCEPTUAL]
    if missing_c:
        report("Conceptual: known features present", "FAIL", f"missing: {missing_c}")
    else:
        report("Conceptual: known features present", "PASS")
    if new_c:
        report("Conceptual: new features added", "PASS", f"{new_c} — update ui.py")

    # ── 5. Known Discursives features present ──
    discursive_codes = [f["code"] for f in tab_features.get("Discursives", [])]
    missing_d = [f for f in KNOWN_DISCURSIVES if f not in discursive_codes]
    new_d = [f for f in discursive_codes if f not in KNOWN_DISCURSIVES]
    if missing_d:
        report("Discursives: known features present", "FAIL", f"missing: {missing_d}")
    else:
        report("Discursives: known features present", "PASS")
    if new_d:
        report("Discursives: new features added", "PASS", f"{new_d} — update ui.py")

    # ── 6. Every feature has a definition ──
    no_def = [f["code"] for feats in tab_features.values() for f in feats if not f["definition"]]
    if no_def:
        report("all features have definitions", "FAIL", f"missing: {no_def}")
    else:
        report("all features have definitions", "PASS")

    # ── 7. Every feature has at least one example ──
    no_ex = [f["code"] for feats in tab_features.values() for f in feats if not f["has_examples"]]
    if no_ex:
        report("all features have examples", "FAIL", f"missing: {no_ex}")
    else:
        report("all features have examples", "PASS")

    # ── 8. No duplicate codes across all tabs ──
    all_codes = [f["code"] for feats in tab_features.values() for f in feats]
    if len(all_codes) == len(set(all_codes)):
        report("no duplicate codes across tabs", "PASS")
    else:
        seen = set()
        dupes = [c for c in all_codes if c in seen or seen.add(c)]
        report("no duplicate codes across tabs", "FAIL", f"dupes: {set(dupes)}")

    # ── 9. Features with rules configured ──
    rule_cols = ["filter_if", "linked_with", "subcode_of", "extra_context_type"]
    has_rules = []
    for tab, feats in tab_features.items():
        for f in feats:
            rules = {c: f[c] for c in rule_cols if f.get(c)}
            if rules:
                has_rules.append((f["code"], tab, rules))
    if has_rules:
        print(f"\n  Features with rules:")
        for code, tab, rules in has_rules:
            print(f"    {code} ({tab}): {rules}")
        report(f"features with rules ({len(has_rules)})", "PASS")
    else:
        report("features with rules", "SKIP", "none configured yet in sheet")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    run_local_tests()

    if not args.local:
        try:
            from google.colab import auth
            auth.authenticate_user()
        except ImportError:
            print("\n  Not in Colab — skipping integration tests.\n")
            _print_summary()
            return
        run_integration_tests()

    _print_summary()
    return 1 if any(s == "FAIL" for _, s, _ in _results) else 0


def _print_summary():
    print(f"\n{'='*50}")
    passed = sum(1 for _, s, _ in _results if s == "PASS")
    failed = sum(1 for _, s, _ in _results if s == "FAIL")
    skipped = sum(1 for _, s, _ in _results if s == "SKIP")
    print(f"  {passed} passed, {failed} failed, {skipped} skipped")
    if failed:
        for name, s, detail in _results:
            if s == "FAIL":
                print(f"    ✗ {name}: {detail}")
    print()


if __name__ == "__main__":
    sys.exit(main() or 0)
