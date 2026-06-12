"""MOL Data Formatter: raw deidentified transcript → pipeline-ready DataFrame.

Takes a raw deidentified transcript (from Tracker "Link to deidentified
transcripts") and produces the formatted CSV the annotation pipeline expects.

Formatting steps:
  1. Filter out dash/empty segment rows
  2. Filter out teacher rows (keep students only)
  3. Add obsid, turn, line, transcript, segment_id_1sd, uttid columns
  4. Normalize segment letters to lowercase
  5. Assign roles from speaker names

Can also pull the raw transcript directly from the Tracker sheet.
"""

import os
import re
import pandas as pd
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Core formatter
# ═══════════════════════════════════════════════════════════════════════════

def format_transcript(raw_df: pd.DataFrame, obsid: int | str) -> pd.DataFrame:
    """Format a raw deidentified transcript into pipeline-ready format.

    Args:
        raw_df: DataFrame with columns: #, segment, in_cue, out_cue,
                duration, speaker, named_language, dialogue, ...
        obsid: Observation ID (e.g. 241)

    Returns:
        Formatted DataFrame with added columns: obsid, turn, line,
        transcript, segment_id_1sd, uttid, role
    """
    obsid = str(int(obsid)) if str(obsid).isdigit() else str(obsid)
    df = raw_df.copy()

    # 1. Filter dash/empty segments
    if "segment" in df.columns:
        df = df[~df["segment"].astype(str).str.strip().isin(["-", "—", "–", ""])]

    # 2. Assign roles
    df["role"] = "Student"
    if "speaker" in df.columns:
        s = df["speaker"].astype(str)
        teacher_like = (
            s.str.contains("teacher", case=False, na=False)
            | s.str.contains(r"^(Mr\.|Ms\.|Mrs\.|Dr\.)", regex=True, na=False)
            | s.str.contains(r"\.", regex=True, na=False)
        )
        teacher_like = teacher_like | df["speaker"].isna()
        df.loc[teacher_like, "role"] = "Teacher"

    # 3. Filter to students only
    df = df[df["role"] == "Student"].copy()

    # 4. Normalize segment letters to lowercase
    if "segment" in df.columns:
        df["segment"] = df["segment"].astype(str).str.strip().str.lower()

    # 5. Add pipeline columns
    df["obsid"] = obsid
    df["transcript"] = obsid

    # turn = sequential row number within this obs
    df["turn"] = range(1, len(df) + 1)
    df["line"] = df["turn"]

    # segment_id_1sd = "241_segment_a"
    if "segment" in df.columns:
        df["segment_id_1sd"] = obsid + "_segment_" + df["segment"].astype(str)
    else:
        df["segment_id_1sd"] = obsid + "_segment_1"

    # uttid = "241_1", "241_2", ...
    df["uttid"] = obsid + "_" + df["turn"].astype(str)

    df = df.reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Tracker integration
# ═══════════════════════════════════════════════════════════════════════════

def format_from_tracker(
    gc,
    obsid: int,
    tracker_sheet_id: str,
    tracker_tab: str = "Tracker",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Pull deidentified transcript from Tracker and format it.

    Args:
        gc: Authorized gspread client.
        obsid: Observation ID (e.g. 241).
        tracker_sheet_id: Google Sheet ID for the Tracker.
        tracker_tab: Tab name in the Tracker sheet.
        save_path: If provided, save formatted CSV here.

    Returns:
        Formatted DataFrame ready for the pipeline.
    """
    # Find the row
    ws = gc.open_by_key(tracker_sheet_id).worksheet(tracker_tab)
    all_data = ws.get_all_values()
    headers = all_data[0]

    idx_col = next(i for i, h in enumerate(headers) if h.strip().lower() == "index")
    transcript_col = next(
        (i for i, h in enumerate(headers) if "deidentified" in h.lower() and "link" in h.lower()),
        None,
    )

    if transcript_col is None:
        raise ValueError("Could not find 'Link to deidentified transcripts' column in Tracker")

    row_idx = None
    for ri, row in enumerate(all_data[1:], 2):
        if idx_col < len(row):
            m = re.search(r"\d{2}-0*(\d+)", row[idx_col].strip())
            if m and int(m.group(1)) == int(obsid):
                row_idx = ri
                break

    if row_idx is None:
        raise ValueError(f"obsid {obsid} not found in Tracker")

    # Get the hyperlink URL
    from gspread.utils import rowcol_to_a1
    cell_addr = rowcol_to_a1(row_idx, transcript_col + 1)
    formula = ws.acell(cell_addr, value_render_option="FORMULA").value

    url = None
    if formula and "HYPERLINK" in str(formula).upper():
        m = re.search(r'HYPERLINK\("([^"]+)"', str(formula))
        if m:
            url = m.group(1)
    if not url:
        url = ws.acell(cell_addr).value

    # Extract sheet ID
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", str(url))
    if not m:
        raise ValueError(f"Could not extract sheet ID from transcript link: {url}")
    sheet_id = m.group(1)

    # Load raw transcript
    t_ws = gc.open_by_key(sheet_id).sheet1
    t_data = t_ws.get_all_values()
    raw_df = pd.DataFrame(t_data[1:], columns=t_data[0])

    # Format
    formatted = format_transcript(raw_df, obsid)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        formatted.to_csv(save_path, index=False)
        print(f"  Saved formatted transcript: {save_path} ({len(formatted)} rows)")

    return formatted


def format_multiple(
    gc,
    obs_ids: list,
    tracker_sheet_id: str,
    tracker_tab: str = "Tracker",
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Format multiple observations and concatenate into one DataFrame."""
    frames = []
    for obsid in obs_ids:
        try:
            save_path = os.path.join(save_dir, f"{obsid}_formatted.csv") if save_dir else None
            df = format_from_tracker(gc, int(obsid), tracker_sheet_id, tracker_tab, save_path)
            frames.append(df)
            print(f"  ✓ obs {obsid}: {len(df)} student utterances")
        except Exception as e:
            print(f"  ✗ obs {obsid}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if save_dir:
        combined_path = os.path.join(save_dir, "mol_formatted_combined.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\n  Combined: {combined_path} ({len(combined)} rows, {len(frames)} obs)")

    return combined


# ═══════════════════════════════════════════════════════════════════════════
# Validation alignment check
# ═══════════════════════════════════════════════════════════════════════════

def verify_alignment(formatted_df: pd.DataFrame, validation_df: pd.DataFrame) -> dict:
    """Check that formatted transcript rows align with validation set.

    Compares uttid-level alignment: same utterances, same order, same dialogue text.
    Flags mismatches so you catch formatting bugs before annotation.

    Returns:
        dict with: aligned (bool), n_matched, n_formatted_only, n_validation_only,
        mismatches (list of dicts with details)
    """
    # Get overlapping obs IDs
    fmt_obs = set(formatted_df["obsid"].astype(str).unique())
    val_obs = set(validation_df["obsid"].astype(str).unique()) if "obsid" in validation_df.columns else set()
    overlap_obs = fmt_obs & val_obs

    if not overlap_obs:
        return {
            "aligned": None,
            "overlap_obs": 0,
            "message": f"No overlapping obs IDs. Formatted: {fmt_obs}, Validation: {val_obs}",
        }

    fmt_sub = formatted_df[formatted_df["obsid"].astype(str).isin(overlap_obs)]
    val_sub = validation_df[validation_df["obsid"].astype(str).isin(overlap_obs)]

    fmt_uttids = set(fmt_sub["uttid"].astype(str)) if "uttid" in fmt_sub.columns else set()
    val_uttids = set(val_sub["uttid"].astype(str)) if "uttid" in val_sub.columns else set()

    matched = fmt_uttids & val_uttids
    fmt_only = fmt_uttids - val_uttids
    val_only = val_uttids - fmt_uttids

    # Check dialogue text for matched uttids
    mismatches = []
    if matched and "dialogue" in fmt_sub.columns and "dialogue" in val_sub.columns:
        fmt_dict = fmt_sub.set_index(fmt_sub["uttid"].astype(str))["dialogue"].to_dict()
        val_dict = val_sub.set_index(val_sub["uttid"].astype(str))["dialogue"].to_dict()
        for uttid in sorted(matched)[:100]:  # check first 100
            fmt_text = str(fmt_dict.get(uttid, "")).strip()
            val_text = str(val_dict.get(uttid, "")).strip()
            if fmt_text != val_text:
                mismatches.append({
                    "uttid": uttid,
                    "formatted": fmt_text[:80],
                    "validation": val_text[:80],
                })

    aligned = len(fmt_only) == 0 and len(val_only) == 0 and len(mismatches) == 0

    return {
        "aligned": aligned,
        "overlap_obs": len(overlap_obs),
        "n_matched": len(matched),
        "n_formatted_only": len(fmt_only),
        "n_validation_only": len(val_only),
        "n_dialogue_mismatches": len(mismatches),
        "mismatches": mismatches[:10],
        "formatted_only_sample": sorted(fmt_only)[:5],
        "validation_only_sample": sorted(val_only)[:5],
    }


def print_alignment_report(result: dict):
    """Pretty-print alignment check results."""
    if result.get("aligned") is None:
        print(f"  ⚠️  {result.get('message', 'No overlap')}")
        return

    status = "✓ ALIGNED" if result["aligned"] else "⚠️  MISALIGNED"
    print(f"\n  {status}")
    print(f"  Overlapping obs: {result['overlap_obs']}")
    print(f"  Matched uttids:  {result['n_matched']}")

    if result["n_formatted_only"]:
        print(f"  Formatted only:  {result['n_formatted_only']} (in new format but not validation)")
        print(f"    sample: {result['formatted_only_sample']}")
    if result["n_validation_only"]:
        print(f"  Validation only: {result['n_validation_only']} (in validation but not new format)")
        print(f"    sample: {result['validation_only_sample']}")
    if result["n_dialogue_mismatches"]:
        print(f"  Text mismatches: {result['n_dialogue_mismatches']}")
        for mm in result["mismatches"][:3]:
            print(f"    {mm['uttid']}: '{mm['formatted'][:40]}' vs '{mm['validation'][:40]}'")
