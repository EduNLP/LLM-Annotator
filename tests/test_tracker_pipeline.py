"""
Comprehensive tests for the Tracker → transcript → video → materials pipeline.

Usage:
    # Local (no auth, pure logic tests):
    python tests/test_tracker_pipeline.py --local

    # Integration (Colab, hits real Tracker sheet):
    python tests/test_tracker_pipeline.py --obsid 241
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TRACKER_SHEET_ID = "1d9iWJzY8GXh1UWJEEYBn9pQeYTuseA0Yqr4WMsTSy-k"
TRACKER_TAB = "Tracker"
DRIVE_BASE = "/content/drive/Shareddrives/Math Out Loud/Analysis/Roles"
INSTRUCTIONS_DIR = f"{DRIVE_BASE}/MOL Conceptual Pipeline Outputs/activity_instructions"

_results = []


def report(name, status, detail=""):
    _results.append((name, status, detail))
    tag = {"PASS": "✓", "FAIL": "✗", "SKIP": "–"}[status]
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))


# ═══════════════════════════════════════════════════════════════════════════
# Helpers (shared between local and integration tests)
# ═══════════════════════════════════════════════════════════════════════════

def find_tracker_row(ws_data, obsid: int):
    headers = ws_data[0]
    idx_col = next(i for i, h in enumerate(headers) if h.strip().lower() == "index")
    for row in ws_data[1:]:
        if idx_col < len(row):
            val = row[idx_col].strip()
            m = re.search(r"\d{2}-0*(\d+)", val)
            if m and int(m.group(1)) == obsid:
                return dict(zip(headers, row))
    return None


def extract_id_from_url(url: str) -> str:
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    return m.group(1) if m else ""


def get_hyperlink_url(sheet, header, headers_list, row_idx):
    """Fetch the actual hyperlink URL from a cell (not just display text)."""
    col_idx = headers_list.index(header) + 1
    from gspread.utils import rowcol_to_a1
    cell_addr = rowcol_to_a1(row_idx, col_idx)
    formula = sheet.acell(cell_addr, value_render_option="FORMULA").value
    if formula and "HYPERLINK" in str(formula).upper():
        m = re.search(r'HYPERLINK\("([^"]+)"', str(formula))
        if m:
            return m.group(1)
    display = sheet.acell(cell_addr).value
    if display and ("google.com" in str(display) or "drive.google" in str(display)):
        return str(display)
    return str(formula) if formula else ""


# ═══════════════════════════════════════════════════════════════════════════
# LOCAL TESTS (no auth)
# ═══════════════════════════════════════════════════════════════════════════

def run_local_tests():
    print("\n── Local Tests (no auth) ──\n")

    # Import without triggering __init__.py
    import importlib.util

    def _load(name):
        path = os.path.join(os.path.dirname(__file__), "..", "src", "llm_annotator", f"{name}.py")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    vl = _load("video_loader")
    ml = _load("materials_loader")

    # ── 1. Tracker row lookup ──
    mock_data = [
        ["Index", "Video Link", "Segment Timestamps", "Link to deidentified transcripts", "Student Materials Folder Link"],
        ["25-0241", "https://drive.google.com/drive/folders/FAKE_VID_FOLDER", "a 3:10-9:00 activity", "https://docs.google.com/spreadsheets/d/FAKE_TRANSCRIPT/edit", "https://drive.google.com/drive/folders/FAKE_MAT_FOLDER"],
        ["26-0100", "link2", "b 1:00-2:00", "link3", "link4"],
    ]
    row = find_tracker_row(mock_data, 241)
    assert row and row["Index"] == "25-0241"
    report("find_tracker_row (obsid=241 → 25-0241)", "PASS")

    row2 = find_tracker_row(mock_data, 100)
    assert row2 and row2["Index"] == "26-0100"
    report("find_tracker_row (obsid=100 → 26-0100)", "PASS")

    assert find_tracker_row(mock_data, 999) is None
    report("find_tracker_row (missing obsid)", "PASS")

    # ── 2. Extract IDs from URLs ──
    assert extract_id_from_url("https://docs.google.com/spreadsheets/d/ABC123/edit") == "ABC123"
    assert extract_id_from_url("https://drive.google.com/drive/folders/XYZ789") == "XYZ789"
    assert ml.extract_folder_id("https://drive.google.com/drive/folders/XYZ789") == "XYZ789"
    assert ml.extract_folder_id("https://drive.google.com/open?id=QWE456") == "QWE456"
    assert ml.extract_folder_id("") is None
    report("extract IDs from various URL formats", "PASS")

    # ── 3. Single-video timestamp parsing ──
    text_single = """a 3:10-9:00 23x4 using base-10 blocks
b 16:50-19:15 23x4 using partial products and base-10 blocks
c 21:43-32:00 48x9 using partial products and base-10 blocks"""
    seg = vl.parse_segment_timestamps(text_single)
    assert len(seg) == 3
    assert seg["a"] == (1, 190.0, 540.0), f"got {seg['a']}"
    assert seg["b"] == (1, 1010.0, 1155.0), f"got {seg['b']}"
    assert seg["c"] == (1, 1303.0, 1920.0), f"got {seg['c']}"
    assert all(v[0] == 1 for v in seg.values())
    report("parse single-video timestamps (3 segments, all vid 1)", "PASS")

    # ── 4. Multi-video with intro, no segments in vid 2 ──
    text_multi_intro = """Vid 1
intro 0:00-0:05
a 2:35-3:30 Warm Up
b 10:00-11:14 Act 1
c 18:45-20:57 Act 1
d 24:50-33:01 Act 1
vid 2"""
    seg2 = vl.parse_segment_timestamps(text_multi_intro)
    assert "a" in seg2 and "d" in seg2
    assert len(seg2) == 4, f"expected 4 segments, got {len(seg2)}: {list(seg2.keys())}"
    assert all(seg2[l][0] == 1 for l in "abcd")
    assert seg2["a"][1] == 155.0  # 2:35
    assert seg2["a"][2] == 210.0  # 3:30
    report("parse multi-video with intro (4 segs in vid 1, intro skipped)", "PASS")

    # ── 5. Multi-video with segments split across vid 1 and vid 2 ──
    text_split = """vid 1
intro 0:00-0:08
a 0:08-10:49 Task E
b 11:59-19:47 Task D
c 20:15-28:57 Task C
vid 2
d 0:00-7:45 Task B
e 8:07-13:33 Task A
f 14:07-20:41 Task F"""
    seg3 = vl.parse_segment_timestamps(text_split)
    assert len(seg3) == 6, f"expected 6, got {len(seg3)}"
    assert seg3["a"][0] == 1 and seg3["b"][0] == 1 and seg3["c"][0] == 1
    assert seg3["d"][0] == 2 and seg3["e"][0] == 2 and seg3["f"][0] == 2
    assert seg3["d"][1] == 0.0, "d starts at 0:00 in vid 2"
    assert seg3["a"][1] == 8.0, "a starts at 0:08"
    assert seg3["f"][2] == 1241.0, f"f ends at 20:41 = 1241s, got {seg3['f'][2]}"
    report("parse split across vid 1 (a-c) and vid 2 (d-f)", "PASS")

    # ── 6. Intro lines are never captured ──
    for text in [text_multi_intro, text_split]:
        seg = vl.parse_segment_timestamps(text)
        for letter in seg:
            assert letter != "i" or seg[letter][1] != 0.0, "intro captured as segment 'i'"
    report("intro lines excluded from segments", "PASS")

    # ── 7. Segment descriptions extractable from raw text ──
    for line in text_single.splitlines():
        m = re.match(r"^\s*([a-z])\s+\S+\s*[-–]\s*\S+\s+(.*)", line)
        assert m, f"couldn't parse description from: {line}"
        assert m.group(2).strip(), f"empty description for segment {m.group(1)}"
    report("segment descriptions parseable from raw text", "PASS")

    # ── 8. Download plan: with segment mapping ──
    mock_images = [
        {"name": "page1.png", "file_id": "id1", "mime_type": "image/png"},
        {"name": "page2.png", "file_id": "id2", "mime_type": "image/png"},
        {"name": "page3.png", "file_id": "id3", "mime_type": "image/png"},
    ]
    plan = ml.build_download_plan(mock_images, 241, segment_map={"a": [0, 1], "b": [2]})
    assert len(plan) == 3
    assert plan[0]["dest_name"] == "241_a_1.png"
    assert plan[1]["dest_name"] == "241_a_2.png"
    assert plan[2]["dest_name"] == "241_b_1.png"
    report("download plan with segment mapping (obsid_letter_i)", "PASS")

    # ── 9. Download plan: without segment mapping ──
    plan2 = ml.build_download_plan(mock_images, 241, segment_map=None)
    assert len(plan2) == 3
    assert plan2[0]["dest_name"] == "241_1.png"
    assert plan2[1]["dest_name"] == "241_2.png"
    assert plan2[2]["dest_name"] == "241_3.png"
    report("download plan without segment mapping (obsid_i)", "PASS")

    # ── 10. Download plan: single image ──
    plan3 = ml.build_download_plan(mock_images[:1], 100, segment_map={"a": [0]})
    assert len(plan3) == 1
    assert plan3[0]["dest_name"] == "100_a_1.png"
    report("download plan single image with segment", "PASS")

    # ── 11. Duration helpers ──
    import pandas as pd
    obs_df = pd.DataFrame([{
        "obsid_raw": "25-0241", "obsid": "241",
        "video_link": "url", "segments": seg3,
    }])
    dur_a = vl.get_segment_duration(obs_df, "241", "a")
    assert dur_a == 10 * 60 + 49 - 8, f"expected 641s, got {dur_a}"
    total = vl.total_video_seconds(obs_df, "241")
    assert total > 0
    report(f"segment duration & total video seconds ({total:.0f}s)", "PASS")

    # ── 12. Dash segment filtering ──
    transcript_rows = [
        {"#": "1", "segment": "a", "dialogue": "hello"},
        {"#": "2", "segment": "-", "dialogue": "skip"},
        {"#": "3", "segment": "b", "dialogue": "world"},
        {"#": "4", "segment": "—", "dialogue": "skip2"},
        {"#": "5", "segment": "", "dialogue": "skip3"},
    ]
    filtered = [r for r in transcript_rows if r["segment"].strip() not in ("-", "—", "–", "")]
    assert len(filtered) == 2
    report("filter dash/empty segments from transcript rows", "PASS")


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS (needs Colab auth)
# ═══════════════════════════════════════════════════════════════════════════

def run_integration_tests(obsid: int):
    print(f"\n── Integration Tests (obsid={obsid}) ──\n")

    from google.auth import default
    import gspread
    creds, _ = default()
    gc = gspread.authorize(creds)

    try:
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        gauth = GoogleAuth()
        gauth.credentials = creds
        gdrive = GoogleDrive(gauth)
    except ImportError:
        print("  PyDrive not available — Drive folder tests will be skipped")
        gdrive = None

    from llm_annotator.video_loader import parse_segment_timestamps
    from llm_annotator.materials_loader import (
        extract_folder_id, list_drive_folder_images, list_drive_folder_videos,
        build_download_plan, download_images,
    )

    # ── 1. Read Tracker sheet ──
    print("  Fetching Tracker sheet...")
    spreadsheet = gc.open_by_key(TRACKER_SHEET_ID)
    ws = spreadsheet.worksheet(TRACKER_TAB)
    all_data = ws.get_all_values()
    headers = all_data[0]

    row = find_tracker_row(all_data, obsid)
    if not row:
        report(f"find obsid {obsid} in Tracker", "FAIL", "row not found")
        return
    report(f"find obsid {obsid} in Tracker", "PASS", f"Index={row.get('Index')}")

    # Find the actual row index for hyperlink fetching
    idx_col_i = next(i for i, h in enumerate(headers) if h.strip().lower() == "index")
    row_idx = None
    for ri, r in enumerate(all_data[1:], 2):
        if ri > 1 and idx_col_i < len(r):
            m = re.search(r"\d{2}-0*(\d+)", r[idx_col_i])
            if m and int(m.group(1)) == obsid:
                row_idx = ri
                break

    # ── 2. Get deidentified transcript link ──
    transcript_col = None
    for h in headers:
        if "deidentified" in h.lower() and "link" in h.lower():
            transcript_col = h
            break

    if transcript_col and row.get(transcript_col):
        transcript_url = get_hyperlink_url(ws, transcript_col, headers, row_idx) if row_idx else row[transcript_col]
        transcript_id = extract_id_from_url(transcript_url)
        if transcript_id:
            report("get deidentified transcript link", "PASS", transcript_url[:80])

            print(f"  Opening transcript sheet {transcript_id}...")
            try:
                t_ws = gc.open_by_key(transcript_id).sheet1
                t_data = t_ws.get_all_values()
                t_headers = [h.strip().lower() for h in t_data[0]]
                expected = ["#", "segment", "in_cue", "out_cue", "duration", "speaker", "named_language", "dialogue"]
                missing = [c for c in expected if c not in t_headers]
                if missing:
                    report("transcript has expected columns", "FAIL", f"missing: {missing}")
                else:
                    report("transcript has expected columns", "PASS", f"{len(t_data)-1} rows")

                # Filter dash segments
                seg_idx = t_headers.index("segment") if "segment" in t_headers else None
                if seg_idx is not None:
                    total = len(t_data) - 1
                    kept = sum(1 for r in t_data[1:] if r[seg_idx].strip() not in ("-", "—", "–", ""))
                    report("filter dash segments in transcript", "PASS", f"{kept}/{total} rows kept")
            except Exception as e:
                report("open transcript sheet", "FAIL", str(e))
        else:
            report("get deidentified transcript link", "FAIL", f"no ID from: {transcript_url}")
    else:
        report("get deidentified transcript link", "SKIP", "column not found")

    # ── 3. Get Video Link + list videos ──
    video_col = "Video Link"
    if video_col in headers and row_idx:
        video_url = get_hyperlink_url(ws, video_col, headers, row_idx)
        vid_folder_id = extract_folder_id(video_url)
        if vid_folder_id:
            report("get Video Link folder ID", "PASS", vid_folder_id[:20])

            if gdrive:
                videos = list_drive_folder_videos(gdrive, vid_folder_id, obsid)
                if len(videos) == 1:
                    report("list videos in folder (single video)", "PASS", videos[0]["name"])
                elif len(videos) > 1:
                    names = [v["name"] for v in videos]
                    report(f"list videos in folder ({len(videos)} videos)", "PASS", ", ".join(names))
                else:
                    report("list videos in folder", "FAIL", "no matching video files found")
            else:
                report("list videos in folder", "SKIP", "no PyDrive")
        else:
            report("get Video Link folder ID", "FAIL", f"from: {video_url}")
    else:
        report("get Video Link", "SKIP", "column not found")

    # ── 4. Parse Segment Timestamps ──
    ts_col = None
    for h in headers:
        if "segment" in h.lower() and "timestamp" in h.lower():
            ts_col = h
            break

    if ts_col and row.get(ts_col):
        raw_ts = row[ts_col]
        segments = parse_segment_timestamps(raw_ts)
        if segments:
            letters = sorted(segments.keys())
            vid_groups = {}
            for l in letters:
                v = segments[l][0]
                vid_groups.setdefault(v, []).append(l)

            detail_parts = []
            for v in sorted(vid_groups):
                detail_parts.append(f"vid {v}: {','.join(vid_groups[v])}")
            detail = "; ".join(detail_parts)

            n_vids = len(vid_groups)
            if n_vids == 1:
                report("parse segment timestamps (single video)", "PASS", detail)
            else:
                report(f"parse segment timestamps ({n_vids} videos)", "PASS", detail)

            for letter in letters[:3]:
                vid_idx, start, end = segments[letter]
                print(f"    {letter}: vid {vid_idx}, {start:.0f}s-{end:.0f}s ({end-start:.0f}s)")
        else:
            report("parse segment timestamps", "FAIL", f"no segments from: {raw_ts[:100]}")
    else:
        report("parse segment timestamps", "SKIP", "column not found")

    # ── 5. Get Student Materials Folder Link + list/plan PNGs ──
    mat_col = None
    for h in headers:
        if "student" in h.lower() and "material" in h.lower() and "link" in h.lower():
            mat_col = h
            break

    if mat_col and row_idx:
        mat_url = get_hyperlink_url(ws, mat_col, headers, row_idx)
        mat_folder_id = extract_folder_id(mat_url)
        if mat_folder_id:
            report("get Student Materials Folder Link", "PASS", mat_folder_id[:20])

            if gdrive:
                images = list_drive_folder_images(gdrive, mat_folder_id)
                report(f"list images in materials folder", "PASS" if images else "FAIL",
                       f"{len(images)} images" + (f": {images[0]['name']}" if images else ""))

                if images:
                    # Plan without segment mapping
                    plan_flat = build_download_plan(images, obsid)
                    report("download plan (obsid_i naming)", "PASS",
                           f"{len(plan_flat)} files, e.g. {plan_flat[0]['dest_name']}")

                    # Plan with segment mapping (map first image to 'a', rest to 'b')
                    if len(images) >= 2:
                        seg_map = {"a": [0], "b": list(range(1, len(images)))}
                        plan_seg = build_download_plan(images, obsid, segment_map=seg_map)
                        report("download plan (obsid_letter_i naming)", "PASS",
                               f"{len(plan_seg)} files, e.g. {plan_seg[0]['dest_name']}")

                    # Dry-run download
                    download_images(gdrive, plan_flat, INSTRUCTIONS_DIR, dry_run=True)
                    report("download PNGs (dry run)", "PASS")
            else:
                report("list images in materials folder", "SKIP", "no PyDrive")
        else:
            report("get Student Materials Folder Link", "SKIP", f"no folder ID from: {mat_url[:60]}")
    else:
        report("get Student Materials Folder Link", "SKIP", "column not found")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--obsid", type=int, default=241)
    args = parser.parse_args()

    run_local_tests()

    if not args.local:
        try:
            from google.colab import auth
            auth.authenticate_user()
        except ImportError:
            print("\n  Not in Colab — skipping integration tests.")
            print("  Run with --local for parse-only tests.\n")
            _print_summary()
            return

        run_integration_tests(args.obsid)

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
