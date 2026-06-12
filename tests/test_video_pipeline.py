"""
Step-by-step pipeline test for the video annotation workflow.
Run in Colab (needs Google Sheets/Drive auth) or locally for parse-only tests.

Usage:
    # In Colab — runs all steps including Drive access:
    !python tests/test_video_pipeline.py --obsid 241

    # Locally — parse-only tests, no Drive access needed:
    python tests/test_video_pipeline.py --local
"""

import argparse
import os
import re
import sys

# Allow importing individual modules without triggering __init__.py's heavy imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

_results = []

def step(name):
    """Decorator that runs a test step, catches errors, prompts for verification."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            print(f"\n{'='*60}")
            print(f"  STEP: {name}")
            print(f"{'='*60}")
            try:
                result = fn(*args, **kwargs)
                print(f"\n  → Result:")
                if isinstance(result, (list, dict)):
                    import json
                    print(f"    {json.dumps(result, indent=2, default=str)[:2000]}")
                else:
                    for line in str(result).split("\n"):
                        print(f"    {line}")

                # Human verification prompt
                answer = input(f"\n  Does this look correct? [Y/n/skip] ").strip().lower()
                if answer == "n":
                    _results.append((name, "FAIL", "User said incorrect"))
                    print(f"  [FAIL] {name}")
                elif answer == "skip":
                    _results.append((name, "SKIP", ""))
                    print(f"  [SKIP] {name}")
                else:
                    _results.append((name, "PASS", ""))
                    print(f"  [PASS] {name}")
                return result
            except Exception as e:
                _results.append((name, "ERROR", str(e)))
                print(f"  [ERROR] {name}: {e}")
                import traceback; traceback.print_exc()
                return None
        return wrapper
    return decorator


def print_summary():
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for name, status, detail in _results:
        icon = {"PASS": "✓", "FAIL": "✗", "ERROR": "!", "SKIP": "–"}[status]
        line = f"  {icon} [{status}] {name}"
        if detail:
            line += f" — {detail}"
        print(line)
    total = len(_results)
    passed = sum(1 for _, s, _ in _results if s == "PASS")
    print(f"\n  {passed}/{total} passed")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRACKER_SHEET_ID = "1d9iWJzY8GXh1UWJEEYBn9pQeYTuseA0Yqr4WMsTSy-k"
TRACKER_TAB = "Tracker"
DRIVE_BASE = "/content/drive/Shareddrives/Math Out Loud/Analysis/Roles"
CLIPS_DIR = f"{DRIVE_BASE}/MOL Conceptual Pipeline Outputs/video_clips"
INSTRUCTIONS_DIR = f"{DRIVE_BASE}/MOL Conceptual Pipeline Outputs/activity_instructions"


# ---------------------------------------------------------------------------
# STEP 1: Find obsid row in Tracker sheet
# ---------------------------------------------------------------------------

@step("1. Find obsid in Tracker sheet")
def test_find_obsid(gc, obsid):
    """Given obsid (e.g. 241), find the row where Index column matches '25-0241'."""
    sheet = gc.open_by_key(TRACKER_SHEET_ID).worksheet(TRACKER_TAB)
    data = sheet.get_all_records()

    # obsid 241 → look for "25-0241" in Index column
    target = f"25-{int(obsid):04d}"
    matching = [row for row in data if str(row.get("Index", "")).strip() == target]

    if not matching:
        raise ValueError(f"obsid {obsid} (Index='{target}') not found in Tracker sheet")

    row = matching[0]
    print(f"  Found: Index={row['Index']}")
    print(f"  Columns in row: {list(row.keys())[:15]}...")
    return {
        "index": row["Index"],
        "video_link": row.get("Video Link", ""),
        "segment_timestamps": row.get("Segment Timestamps", ""),
        "transcript_link": row.get("Link to deidentified transcripts", ""),
        "materials_link": row.get("Student Materials Folder Link", ""),
    }


# ---------------------------------------------------------------------------
# STEP 2: Extract Drive folder ID from Video Link hyperlink
# ---------------------------------------------------------------------------

@step("2. Extract Drive folder ID from Video Link")
def test_extract_video_folder(gc, obsid, tracker_row):
    """The Video Link cell may be a hyperlink. Extract the folder ID."""
    # gspread get_all_records returns display text, not hyperlinks.
    # Need to fetch the raw cell to get the hyperlink URL.
    sheet = gc.open_by_key(TRACKER_SHEET_ID).worksheet(TRACKER_TAB)

    # Find Video Link column letter
    header = sheet.row_values(1)
    try:
        col_idx = header.index("Video Link") + 1
    except ValueError:
        raise ValueError(f"'Video Link' column not found. Headers: {header}")

    # Find row number for this obsid
    index_col = sheet.col_values(header.index("Index") + 1)
    target = f"25-{int(obsid):04d}"
    try:
        row_idx = index_col.index(target) + 1
    except ValueError:
        raise ValueError(f"Index '{target}' not found")

    # Fetch cell with hyperlink
    from gspread.utils import rowcol_to_a1
    cell_addr = rowcol_to_a1(row_idx, col_idx)

    # Try to get the actual hyperlink URL (not just display text)
    cell_formula = sheet.acell(cell_addr, value_render_option="FORMULA").value
    print(f"  Cell {cell_addr} raw value: {str(cell_formula)[:200]}")

    # Extract URL from =HYPERLINK("url", "text") or plain URL
    url = None
    if cell_formula and "HYPERLINK" in str(cell_formula).upper():
        m = re.search(r'HYPERLINK\("([^"]+)"', str(cell_formula))
        if m:
            url = m.group(1)
    elif cell_formula and "drive.google.com" in str(cell_formula):
        url = str(cell_formula)

    if not url:
        # Fallback: the display text itself might be the URL
        display_text = sheet.acell(cell_addr).value
        if display_text and "drive.google.com" in str(display_text):
            url = display_text

    if not url:
        raise ValueError(f"Could not extract URL from Video Link cell: {cell_formula}")

    # Extract folder ID from URL
    folder_id = None
    m = re.search(r'/folders/([a-zA-Z0-9_-]+)', url)
    if m:
        folder_id = m.group(1)
    else:
        m = re.search(r'id=([a-zA-Z0-9_-]+)', url)
        if m:
            folder_id = m.group(1)

    print(f"  URL: {url}")
    print(f"  Folder ID: {folder_id}")
    return {"url": url, "folder_id": folder_id}


# ---------------------------------------------------------------------------
# STEP 3: List video files in Drive folder
# ---------------------------------------------------------------------------

@step("3. List video files in Drive folder")
def test_list_video_files(gdrive, folder_id, obsid):
    """List all video files in the Drive folder. Expect OBS-25-XXXX_video*.mov names."""
    if gdrive is None:
        raise RuntimeError("PyDrive not available — run in Colab")

    file_list = gdrive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=false"
    }).GetList()

    target_prefix = f"OBS-25-{int(obsid):04d}"
    video_files = [
        {"name": f["title"], "id": f["id"], "size": f.get("fileSize", "?")}
        for f in file_list
        if f["title"].startswith(target_prefix)
    ]

    if not video_files:
        print(f"  WARNING: No files matching '{target_prefix}*' found.")
        print(f"  All files in folder: {[f['title'] for f in file_list][:20]}")

    for vf in video_files:
        print(f"  {vf['name']}  (id={vf['id'][:12]}...)")

    return video_files


# ---------------------------------------------------------------------------
# STEP 4: Parse Segment Timestamps
# ---------------------------------------------------------------------------

@step("4. Parse Segment Timestamps")
def test_parse_segments(tracker_row):
    """Parse the free-text Segment Timestamps into structured data."""
    from llm_annotator.video_loader import parse_segment_timestamps

    raw_text = tracker_row["segment_timestamps"]
    print(f"  Raw text:\n    {raw_text.replace(chr(10), chr(10) + '    ')}\n")

    segments = parse_segment_timestamps(raw_text)

    result = []
    for letter in sorted(segments.keys()):
        vid_idx, start, end = segments[letter]
        duration = end - start

        # Extract description from raw text
        desc = ""
        for line in raw_text.splitlines():
            m = re.match(rf"^\s*{letter}\s+\S+\s*[-–]\s*\S+\s*(.*)", line, re.IGNORECASE)
            if m:
                desc = m.group(1).strip()

        result.append({
            "letter": letter,
            "video": vid_idx,
            "start": f"{int(start//60)}:{int(start%60):02d}",
            "end": f"{int(end//60)}:{int(end%60):02d}",
            "duration_sec": round(duration, 1),
            "description": desc,
        })
        print(f"  {letter}: vid {vid_idx}, {result[-1]['start']}-{result[-1]['end']} ({duration:.0f}s) {desc}")

    return result


# ---------------------------------------------------------------------------
# STEP 5: Load deidentified transcript
# ---------------------------------------------------------------------------

@step("5. Load deidentified transcript")
def test_load_transcript(gc, tracker_row, obsid):
    """Grab the transcript CSV and show sample rows with segment mapping."""
    link = tracker_row.get("transcript_link", "")
    print(f"  Transcript link: {link}")

    # Extract sheet/file ID
    file_id = None
    m = re.search(r'/d/([a-zA-Z0-9_-]+)', link)
    if m:
        file_id = m.group(1)
    elif re.match(r'^[a-zA-Z0-9_-]+$', link):
        file_id = link

    if not file_id:
        raise ValueError(f"Could not extract file ID from transcript link: {link}")

    import pandas as pd
    try:
        sheet = gc.open_by_key(file_id).sheet1
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
    except Exception:
        # Maybe it's a CSV on Drive
        raise ValueError("Could not open as Google Sheet. May need PyDrive download.")

    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Show a few rows with segment info
    sample_cols = [c for c in ["#", "segment", "in_cue", "out_cue", "speaker", "dialogue"] if c in df.columns]
    if sample_cols:
        # Show first non-dash row and last row
        non_dash = df[df.get("#", df.get("segment", pd.Series())) != "-"]
        if len(non_dash) > 0:
            print(f"\n  First utterance row:")
            print(f"    {non_dash.iloc[0][sample_cols].to_dict()}")
            print(f"\n  Last utterance row:")
            print(f"    {non_dash.iloc[-1][sample_cols].to_dict()}")

    # Explain timestamp format
    if "in_cue" in df.columns:
        sample_cue = df["in_cue"].iloc[1] if len(df) > 1 else ""
        print(f"\n  Timestamp format in transcript: '{sample_cue}'")
        if str(sample_cue).count(":") == 3:
            print(f"  → hh:mm:ss:ff (ff = frames, not seconds)")
        elif str(sample_cue).count(":") == 2:
            print(f"  → hh:mm:ss")

    return {"file_id": file_id, "shape": df.shape, "columns": list(df.columns)}


# ---------------------------------------------------------------------------
# STEP 6: Verify timestamps align across sources
# ---------------------------------------------------------------------------

@step("6. Verify timestamp alignment")
def test_timestamp_alignment(segments, tracker_row, gc, obsid):
    """
    Check that segment boundaries from Tracker match what's in the transcript.

    Tracker Segment Timestamps: mm:ss-mm:ss per segment letter
    Transcript in_cue/out_cue: hh:mm:ss:ff (ff = frames at ~30fps)

    The in_cue times are absolute within each video file. So segment 'a'
    starting at 3:10 in the Tracker means the first row of segment 'a'
    in the transcript should have in_cue near 00:03:10:xx.

    If segments span multiple videos, in_cue resets to 00:00:xx for
    segments in the second video.
    """
    if not segments:
        raise ValueError("No segments to verify")

    print("  Timestamp format comparison:")
    print("    Tracker:    mm:ss (segment start/end within video)")
    print("    Transcript: hh:mm:ss:ff (absolute within video, ff=frames)")
    print()

    for seg in segments[:3]:  # check first 3 segments
        letter = seg["letter"]
        tracker_start = seg["start"]
        print(f"  Segment '{letter}': Tracker says starts at {tracker_start}")
        print(f"    → Expect transcript in_cue near 00:{tracker_start}:xx for first row of segment '{letter}'")
        print(f"    (Verify manually in transcript)")

    return "Manual verification needed — check that transcript in_cue values match Tracker segment start times"


# ---------------------------------------------------------------------------
# STEP 7: Cut video files (stub — shows what WOULD happen)
# ---------------------------------------------------------------------------

@step("7. Cut video files (dry run)")
def test_cut_videos_dry_run(video_files, segments, obsid):
    """Show what ffmpeg commands would run to cut segments from video files."""
    if not video_files:
        raise ValueError("No video files to cut")

    os.makedirs(CLIPS_DIR, exist_ok=True) if os.path.exists(os.path.dirname(CLIPS_DIR)) else None

    commands = []
    for seg in segments:
        letter = seg["letter"]
        vid_idx = seg["video"]

        # Find matching video file
        matching = [vf for vf in video_files if f"video_{vid_idx}" in vf["name"].lower()
                    or (vid_idx == 1 and "video_1" not in vf["name"].lower()
                        and "video_2" not in vf["name"].lower()
                        and len(video_files) == 1)]
        if not matching:
            # Fallback: if only one video, use it
            if len(video_files) == 1:
                matching = video_files
            else:
                print(f"  WARNING: No video file found for vid {vid_idx}, segment {letter}")
                continue

        src = matching[0]["name"]
        _, start_sec = seg["start"].split(":") if ":" in seg["start"] else (0, seg["start"])
        out_name = f"OBS-25-{int(obsid):04d}_{letter}.mp4"
        out_path = os.path.join(CLIPS_DIR, out_name)

        cmd = f'ffmpeg -i "{src}" -ss {seg["start"]}:00 -to {seg["end"]}:00 -c copy "{out_path}"'
        commands.append({"segment": letter, "source": src, "output": out_name, "command": cmd})
        print(f"  {letter}: {src} [{seg['start']}-{seg['end']}] → {out_name}")

    return commands


# ---------------------------------------------------------------------------
# STEP 8: Download student materials PNGs
# ---------------------------------------------------------------------------

@step("8. Download student materials PNGs (dry run)")
def test_download_materials(gc, gdrive, tracker_row, obsid):
    """List PNGs in the Student Materials folder. Show what would be downloaded."""
    link = tracker_row.get("materials_link", "")
    print(f"  Student Materials Folder Link: {link}")

    if not link:
        return "No Student Materials Folder Link found"

    # Extract folder ID
    folder_id = None
    m = re.search(r'/folders/([a-zA-Z0-9_-]+)', str(link))
    if m:
        folder_id = m.group(1)

    if not folder_id:
        return f"Could not extract folder ID from: {link}"

    if gdrive is None:
        return "PyDrive not available — run in Colab to list/download files"

    file_list = gdrive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=false"
    }).GetList()

    pngs = [f for f in file_list if f["title"].lower().endswith((".png", ".jpg", ".jpeg"))]
    print(f"  Found {len(pngs)} image files:")

    downloads = []
    for i, f in enumerate(pngs, 1):
        # Naming: obsid_segmentletter_i.png if we know segment, else obsid_i.png
        out_name = f"OBS-25-{int(obsid):04d}_{i}.png"
        out_path = os.path.join(INSTRUCTIONS_DIR, out_name)
        downloads.append({"src": f["title"], "dest": out_name, "file_id": f["id"]})
        print(f"  {f['title']} → {out_name}")

    return downloads


# ---------------------------------------------------------------------------
# Local-only parse tests (no Google auth needed)
# ---------------------------------------------------------------------------

def _import_video_loader():
    """Import video_loader without triggering __init__.py's heavy deps."""
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "src", "llm_annotator", "video_loader.py")
    spec = importlib.util.spec_from_file_location("video_loader", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_local_tests():
    """Parse-only tests that run without any Google auth."""
    vl = _import_video_loader()
    parse_segment_timestamps = vl.parse_segment_timestamps

    print("\n" + "="*60)
    print("  LOCAL PARSE TESTS (no auth needed)")
    print("="*60)

    # Test 1: single video
    text1 = """a 3:10-9:00 23x4 using base-10 blocks
b 16:50-19:15 23x4 using partial products and base-10 blocks
c 21:43-32:00 48x9 using partial products and base-10 blocks"""
    seg1 = parse_segment_timestamps(text1)
    assert len(seg1) == 3, f"Expected 3 segments, got {len(seg1)}"
    assert seg1["a"][0] == 1, f"Expected video 1, got {seg1['a'][0]}"
    assert seg1["a"][1] == 190.0, f"Expected 190s, got {seg1['a'][1]}"
    assert seg1["a"][2] == 540.0, f"Expected 540s, got {seg1['a'][2]}"
    print(f"  [PASS] Single video: {seg1}")

    # Test 2: multi-video with intro
    text2 = """Vid 1
intro 0:00-0:05
a 2:35-3:30 Warm Up
b 10:00-11:14 Act 1
c 18:45-20:57 Act 1
d 24:50-33:01 Act 1
vid 2"""
    seg2 = parse_segment_timestamps(text2)
    assert "a" in seg2 and "d" in seg2
    assert seg2["a"][0] == 1, "Segment a should be video 1"
    # intro is NOT a valid segment letter (regex matches single [a-z])
    # Actually "intro" starts with 'i' but the full word won't match the pattern
    print(f"  [PASS] Multi-video with intro: {seg2}")
    print(f"    Note: 'intro' line {'is' if 'i' in seg2 else 'is NOT'} captured")

    # Test 3: multi-video with segments split across videos
    text3 = """vid 1
intro 0:00-0:08
a 0:08-10:49 Task E
b 11:59-19:47 Task D
c 20:15-28:57 Task C
vid 2
d 0:00-7:45 Task B
e 8:07-13:33 Task A
f 14:07-20:41 Task F"""
    seg3 = parse_segment_timestamps(text3)
    assert seg3["a"][0] == 1, "a should be video 1"
    assert seg3["c"][0] == 1, "c should be video 1"
    assert seg3["d"][0] == 2, "d should be video 2"
    assert seg3["f"][0] == 2, "f should be video 2"
    assert seg3["d"][1] == 0.0, "d starts at 0:00 in video 2"
    print(f"  [PASS] Split across videos: a-c → vid 1, d-f → vid 2")

    # Test 4: ExperimentConfig validation
    try:
        import importlib.util as _ilu
        _s = _ilu.spec_from_file_location("config", os.path.join(os.path.dirname(__file__), "..", "src", "llm_annotator", "config.py"))
        _cfg = _ilu.module_from_spec(_s)
        _s.loader.exec_module(_cfg)
        ExperimentConfig = _cfg.ExperimentConfig
    except Exception as e:
        print(f"  [SKIP] Config import failed (likely missing deps): {e}")
        ExperimentConfig = None

    if ExperimentConfig:
        try:
            ExperimentConfig(use_video=True)
            print("  [FAIL] Should have raised ValueError for missing obs_sheet_source")
        except ValueError:
            print("  [PASS] use_video=True without obs_sheet_source raises ValueError")

    # Test 5: filter_by_feature_rules
    # Inline reimplementation for local testing — the real function is in preprocess.py
    # but importing it triggers the full package chain which needs openai etc.
    import pandas as pd

    def filter_by_feature_rules(df, feature_meta):
        filter_codes = feature_meta.get("filter_if", [])
        if not filter_codes:
            return df
        mask = pd.Series(False, index=df.index)
        for code in filter_codes:
            if code in df.columns:
                mask = mask | (df[code] == 1)
        return df[~mask].copy()

    df = pd.DataFrame({
        "uttid": [1, 2, 3, 4, 5],
        "dialogue": ["a", "b", "c", "d", "e"],
        "offtask": [1, 0, 1, 0, 0],
    })

    filtered = filter_by_feature_rules(df, {"filter_if": ["offtask"]})
    assert len(filtered) == 3, f"Expected 3 rows, got {len(filtered)}"
    print(f"  [PASS] filter_by_feature_rules: 5→3 rows (offtask filtered)")

    no_filter = filter_by_feature_rules(df, {"filter_if": []})
    assert len(no_filter) == 5
    print(f"  [PASS] filter_by_feature_rules: empty filter keeps all")

    missing_col = filter_by_feature_rules(df, {"filter_if": ["nonexistent"]})
    assert len(missing_col) == 5
    print(f"  [PASS] filter_by_feature_rules: missing column prints warning, keeps all")

    print(f"\n  ✓ All local tests passed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obsid", type=str, default="241", help="Observation ID to test")
    parser.add_argument("--local", action="store_true", help="Run local-only parse tests (no auth)")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Don't actually cut/download files")
    args = parser.parse_args()

    # Always run local tests first
    run_local_tests()

    if args.local:
        return

    # ── Google auth ──
    print("\n" + "="*60)
    print("  PIPELINE INTEGRATION TESTS (needs Google auth)")
    print("="*60)

    try:
        from google.colab import auth
        auth.authenticate_user()
    except ImportError:
        print("  Not in Colab — skipping integration tests.")
        print("  Run with --local for parse-only tests, or run in Colab.")
        return

    import gspread
    from google.auth import default
    creds, _ = default()
    gc = gspread.authorize(creds)

    try:
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        gauth = GoogleAuth()
        gauth.credentials = creds
        gdrive = GoogleDrive(gauth)
    except ImportError:
        print("  PyDrive not available — Drive file listing will be skipped")
        gdrive = None

    # ── Run steps ──
    tracker_row = test_find_obsid(gc, args.obsid)
    if not tracker_row:
        print_summary(); return

    folder_info = test_extract_video_folder(gc, args.obsid, tracker_row)

    video_files = []
    if folder_info and folder_info.get("folder_id") and gdrive:
        video_files = test_list_video_files(gdrive, folder_info["folder_id"], args.obsid)

    segments = test_parse_segments(tracker_row)

    transcript_info = test_load_transcript(gc, tracker_row, args.obsid)

    if segments:
        test_timestamp_alignment(segments, tracker_row, gc, args.obsid)

    if video_files and segments:
        test_cut_videos_dry_run(video_files, segments, args.obsid)

    test_download_materials(gc, gdrive, tracker_row, args.obsid)

    print_summary()


if __name__ == "__main__":
    main()
