"""
Test video clipping and cross-source timestamp verification.

Three timestamp sources must agree:
  1. Tracker "Segment Timestamps"  →  m:ss or mm:ss
  2. Deidentified transcript CSV   →  hh:mm:ss:ff (ff = frames @ ~30fps)
  3. Clipped video duration        →  ffprobe seconds

Transcript in_cue is absolute within each video — restarts at 00:00:00:00
when the segment switches to a different video. Dash segments are ignored.

Usage:
    python tests/test_video_clipping.py --local           # parse/conversion tests only
    python tests/test_video_clipping.py --obsid 241       # full integration (Colab)
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TRACKER_SHEET_ID = "1d9iWJzY8GXh1UWJEEYBn9pQeYTuseA0Yqr4WMsTSy-k"
TRACKER_TAB = "Tracker"
DRIVE_BASE = "/content/drive/Shareddrives/Math Out Loud/Analysis/Roles"
CLIPS_DIR = f"{DRIVE_BASE}/MOL Conceptual Pipeline Outputs/video_clips"

_results = []


def report(name, status, detail=""):
    _results.append((name, status, detail))
    tag = {"PASS": "✓", "FAIL": "✗", "SKIP": "–"}[status]
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))


def ask(question: str) -> str:
    """Human verification prompt. Returns 'y', 'n', or 'skip'."""
    try:
        ans = input(f"\n  {question} [Y/n/skip] ").strip().lower()
    except EOFError:
        ans = "y"
    return ans if ans in ("n", "skip") else "y"


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def find_tracker_row(ws_data, obsid: int):
    headers = ws_data[0]
    idx_col = next(i for i, h in enumerate(headers) if h.strip().lower() == "index")
    for row in ws_data[1:]:
        if idx_col < len(row):
            m = re.search(r"\d{2}-0*(\d+)", row[idx_col].strip())
            if m and int(m.group(1)) == obsid:
                return dict(zip(headers, row)), ws_data[0]
    return None, headers


def get_hyperlink_url(sheet, col_name, headers, row_idx):
    col_idx = headers.index(col_name) + 1
    from gspread.utils import rowcol_to_a1
    cell_addr = rowcol_to_a1(row_idx, col_idx)
    formula = sheet.acell(cell_addr, value_render_option="FORMULA").value
    if formula and "HYPERLINK" in str(formula).upper():
        m = re.search(r'HYPERLINK\("([^"]+)"', str(formula))
        if m:
            return m.group(1)
    display = sheet.acell(cell_addr).value
    if display and "google.com" in str(display):
        return str(display)
    return str(formula or "")


def extract_id_from_url(url: str) -> str:
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    return m.group(1) if m else ""


# ═══════════════════════════════════════════════════════════════════════════
# LOCAL TESTS
# ═══════════════════════════════════════════════════════════════════════════

def run_local_tests():
    print("\n── Local Tests (time conversion & alignment) ──\n")

    import importlib.util

    def _load(name):
        path = os.path.join(os.path.dirname(__file__), "..", "src", "llm_annotator", f"{name}.py")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    vc = _load("video_clipper")
    vl = _load("video_loader")

    # ── 1. Tracker time conversion (m:ss, mm:ss, h:mm:ss) ──
    assert vc.tracker_time_to_sec("3:10") == 190.0
    assert vc.tracker_time_to_sec("16:50") == 1010.0
    assert vc.tracker_time_to_sec("1:05:30") == 3930.0
    assert vc.tracker_time_to_sec("0:00") == 0.0
    report("tracker_time_to_sec (m:ss, mm:ss, h:mm:ss)", "PASS")

    # ── 2. Transcript time conversion (hh:mm:ss:ff) ──
    assert vc.transcript_time_to_sec("00:03:10:00") == 190.0
    assert vc.transcript_time_to_sec("00:03:10:15") == 190.5  # 15 frames at 30fps
    assert vc.transcript_time_to_sec("00:16:50:00") == 1010.0
    assert vc.transcript_time_to_sec("01:05:30:00") == 3930.0
    assert vc.transcript_time_to_sec("00:00:00:00") == 0.0
    assert vc.transcript_time_to_sec("00:00:01:29") == 1 + 29 / 30.0  # last frame
    report("transcript_time_to_sec (hh:mm:ss:ff at 30fps)", "PASS")

    # ── 3. Transcript time with only hh:mm:ss (no frames) ──
    assert vc.transcript_time_to_sec("00:03:10") == 190.0
    assert vc.transcript_time_to_sec("1:05:30") == 3930.0
    report("transcript_time_to_sec fallback (hh:mm:ss, no frames)", "PASS")

    # ── 4. sec_to_ffmpeg ──
    assert vc.sec_to_ffmpeg(0) == "00:00:00.000"
    assert vc.sec_to_ffmpeg(190.5) == "00:03:10.500"
    assert vc.sec_to_ffmpeg(3930.0) == "01:05:30.000"
    report("sec_to_ffmpeg conversion", "PASS")

    # ── 5. Verify segment alignment (matching) ──
    mock_rows = [
        {"segment": "a", "in_cue": "00:03:10:00", "out_cue": "00:03:45:00"},
        {"segment": "a", "in_cue": "00:03:45:01", "out_cue": "00:08:58:00"},
        {"segment": "-", "in_cue": "00:08:59:00", "out_cue": "00:09:00:00"},
        {"segment": "b", "in_cue": "00:16:50:00", "out_cue": "00:19:13:00"},
    ]
    result_a = vc.verify_segment_alignment("a", 190.0, 540.0, mock_rows, tolerance_sec=5.0)
    assert result_a["aligned"] is True, f"expected aligned, got {result_a}"
    assert result_a["n_rows"] == 2, "dash row should be excluded"
    assert result_a["drift_start_sec"] < 1.0
    report("verify_segment_alignment (matching, dash rows filtered)", "PASS")

    # ── 6. Verify segment alignment (drifted) ──
    result_drift = vc.verify_segment_alignment("a", 200.0, 540.0, mock_rows, tolerance_sec=5.0)
    assert result_drift["aligned"] is False, f"10s drift should fail at 5s tolerance"
    assert result_drift["drift_start_sec"] == 10.0
    report("verify_segment_alignment (drift detected)", "PASS")

    # ── 7. Verify segment alignment (missing segment) ──
    result_miss = vc.verify_segment_alignment("z", 0, 100, mock_rows)
    assert result_miss["aligned"] is None
    report("verify_segment_alignment (missing segment)", "PASS")

    # ── 8. Verify alignment with multi-video (in_cue restarts) ──
    # Video 2 segments have in_cue restarting from 00:00:00:00
    multi_rows = [
        {"segment": "c", "in_cue": "00:20:15:00", "out_cue": "00:28:57:00"},  # vid 1
        {"segment": "d", "in_cue": "00:00:00:00", "out_cue": "00:07:45:00"},  # vid 2 — restarts!
        {"segment": "d", "in_cue": "00:07:45:01", "out_cue": "00:07:45:15"},  # vid 2 continued
    ]
    # Tracker says d is 0:00-7:45 in vid 2
    result_d = vc.verify_segment_alignment("d", 0.0, 465.0, multi_rows, tolerance_sec=2.0)
    assert result_d["aligned"] is True, f"vid 2 segment d should align, got {result_d}"
    assert result_d["transcript_first_in_cue"] == 0.0, "in_cue should restart at 0 for vid 2"
    report("verify_segment_alignment (multi-video, in_cue restarts)", "PASS")

    # ── 9. Clip command generation (dry run) ──
    clip_result = vc.clip_segment(
        "/fake/video.mov", 190.0, 540.0, "/fake/out/clip_a.mp4", dry_run=True
    )
    assert clip_result["dry_run"] is True
    assert "00:03:10.000" in clip_result["command"]
    assert "00:09:00.000" in clip_result["command"]
    assert clip_result["duration_sec"] == 350.0
    report("clip_segment dry run (correct ffmpeg times)", "PASS")

    # ── 10. Dash segment filtering from transcript ──
    all_rows = [
        {"segment": "a", "in_cue": "00:01:00:00", "out_cue": "00:02:00:00", "dialogue": "hello"},
        {"segment": "-", "in_cue": "00:02:01:00", "out_cue": "00:02:05:00", "dialogue": ""},
        {"segment": "—", "in_cue": "00:02:06:00", "out_cue": "00:02:07:00", "dialogue": ""},
        {"segment": "b", "in_cue": "00:05:00:00", "out_cue": "00:06:00:00", "dialogue": "world"},
        {"segment": "", "in_cue": "", "out_cue": "", "dialogue": ""},
    ]
    clean = [r for r in all_rows if str(r["segment"]).strip() not in ("-", "—", "–", "")]
    assert len(clean) == 2
    assert clean[0]["segment"] == "a" and clean[1]["segment"] == "b"
    report("filter dash/empty rows from transcript", "PASS")

    # ── 11. Cross-format consistency: tracker_time ≈ transcript_time for same moment ──
    tracker_sec = vc.tracker_time_to_sec("3:10")
    transcript_sec = vc.transcript_time_to_sec("00:03:10:00")
    assert tracker_sec == transcript_sec == 190.0
    # With frames
    transcript_sec_frames = vc.transcript_time_to_sec("00:03:10:15")
    assert abs(transcript_sec_frames - tracker_sec) < 1.0
    report("cross-format: tracker m:ss == transcript hh:mm:ss:ff (same moment)", "PASS")

    # ── 12. End-to-end: parse tracker → verify against transcript rows ──
    tracker_text = """a 3:10-9:00 23x4 using base-10 blocks
b 16:50-19:15 23x4 partial products"""
    segments = vl.parse_segment_timestamps(tracker_text)

    transcript_data = [
        {"segment": "a", "in_cue": "00:03:11:00", "out_cue": "00:03:30:00"},
        {"segment": "a", "in_cue": "00:03:30:01", "out_cue": "00:08:58:15"},
        {"segment": "-", "in_cue": "00:09:00:00", "out_cue": "00:09:05:00"},
        {"segment": "b", "in_cue": "00:16:49:20", "out_cue": "00:17:00:00"},
        {"segment": "b", "in_cue": "00:17:00:01", "out_cue": "00:19:14:10"},
    ]

    for letter in sorted(segments.keys()):
        vid_idx, start, end = segments[letter]
        result = vc.verify_segment_alignment(letter, start, end, transcript_data, tolerance_sec=5.0)
        assert result["aligned"] is True, f"segment {letter} misaligned: {result}"
    report("end-to-end: parse tracker → verify against transcript (within 5s)", "PASS")

    # ── 13. Multi-video end-to-end ──
    tracker_multi = """vid 1
intro 0:00-0:08
a 0:08-10:49 Task E
b 11:59-19:47 Task D
vid 2
c 0:00-7:45 Task B
d 8:07-13:33 Task A"""
    segments_m = vl.parse_segment_timestamps(tracker_multi)

    transcript_multi = [
        # vid 1
        {"segment": "a", "in_cue": "00:00:09:00", "out_cue": "00:10:48:00"},
        {"segment": "b", "in_cue": "00:12:00:00", "out_cue": "00:19:46:00"},
        # vid 2 — in_cue restarts
        {"segment": "c", "in_cue": "00:00:01:00", "out_cue": "00:07:44:00"},
        {"segment": "d", "in_cue": "00:08:08:00", "out_cue": "00:13:32:00"},
    ]

    for letter in sorted(segments_m.keys()):
        vid_idx, start, end = segments_m[letter]
        result = vc.verify_segment_alignment(letter, start, end, transcript_multi, tolerance_sec=5.0)
        assert result["aligned"] is True, \
            f"segment {letter} (vid {vid_idx}) misaligned: drift_start={result.get('drift_start_sec')}, drift_end={result.get('drift_end_sec')}"
    report("multi-video end-to-end: vid 1 (a,b) + vid 2 (c,d) alignment", "PASS")


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS (Colab, real data, human verification)
# ═══════════════════════════════════════════════════════════════════════════

def run_integration_tests(obsid: int):
    print(f"\n── Integration Tests (obsid={obsid}, human verification) ──\n")

    from google.auth import default
    import gspread
    import pandas as pd
    creds, _ = default()
    gc = gspread.authorize(creds)

    try:
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        gauth = GoogleAuth()
        gauth.credentials = creds
        gdrive = GoogleDrive(gauth)
    except ImportError:
        gdrive = None

    from llm_annotator.video_loader import parse_segment_timestamps
    from llm_annotator.video_clipper import (
        verify_segment_alignment, clip_segment, verify_clip_duration,
        tracker_time_to_sec, transcript_time_to_sec,
    )
    from llm_annotator.materials_loader import extract_folder_id, list_drive_folder_videos

    # ── 1. Fetch Tracker row ──
    print("  Fetching Tracker...")
    spreadsheet = gc.open_by_key(TRACKER_SHEET_ID)
    ws = spreadsheet.worksheet(TRACKER_TAB)
    all_data = ws.get_all_values()
    headers = all_data[0]

    row, _ = find_tracker_row(all_data, obsid)
    if not row:
        report(f"find obsid {obsid}", "FAIL", "not found")
        return
    report(f"find obsid {obsid}", "PASS", row.get("Index"))

    # Find row index for hyperlinks
    idx_col_i = next(i for i, h in enumerate(headers) if h.strip().lower() == "index")
    row_idx = None
    for ri, r in enumerate(all_data[1:], 2):
        m = re.search(r"\d{2}-0*(\d+)", r[idx_col_i].strip())
        if m and int(m.group(1)) == obsid:
            row_idx = ri
            break

    # ── 2. Parse Tracker segment timestamps ──
    ts_col = None
    for h in headers:
        if "segment" in h.lower() and "timestamp" in h.lower():
            ts_col = h
            break

    if not ts_col or not row.get(ts_col):
        report("parse segment timestamps", "SKIP", "column empty")
        return

    raw_ts = row[ts_col]
    segments = parse_segment_timestamps(raw_ts)
    letters = sorted(segments.keys())
    vid_groups = {}
    for l in letters:
        vid_groups.setdefault(segments[l][0], []).append(l)
    n_vids = len(vid_groups)

    print(f"\n  Tracker timestamps ({n_vids} video(s)):")
    for letter in letters:
        vid_idx, start, end = segments[letter]
        print(f"    {letter}: vid {vid_idx}, {start:.0f}s-{end:.0f}s ({end-start:.0f}s)")
    report(f"parse segment timestamps ({len(segments)} segments, {n_vids} videos)", "PASS")

    # ── 3. Load deidentified transcript ──
    transcript_col = None
    for h in headers:
        if "deidentified" in h.lower() and "link" in h.lower():
            transcript_col = h
            break

    transcript_rows = []
    if transcript_col and row_idx:
        t_url = get_hyperlink_url(ws, transcript_col, headers, row_idx)
        t_id = extract_id_from_url(t_url)
        if t_id:
            print(f"\n  Fetching transcript {t_id}...")
            try:
                t_ws = gc.open_by_key(t_id).sheet1
                t_data = t_ws.get_all_values()
                t_headers = t_data[0]
                transcript_rows = [dict(zip(t_headers, r)) for r in t_data[1:]]
                # Filter dash segments
                before = len(transcript_rows)
                transcript_rows = [
                    r for r in transcript_rows
                    if str(r.get("segment", "")).strip() not in ("-", "—", "–", "")
                ]
                report("load transcript", "PASS",
                       f"{len(transcript_rows)} rows (filtered {before - len(transcript_rows)} dash rows)")
            except Exception as e:
                report("load transcript", "FAIL", str(e))

    # ── 4. Verify timestamp alignment per segment ──
    if transcript_rows and segments:
        print(f"\n  Verifying alignment (tolerance: 5s)...")
        all_aligned = True
        for letter in letters:
            vid_idx, start, end = segments[letter]
            result = verify_segment_alignment(letter, start, end, transcript_rows, tolerance_sec=5.0)

            if result["aligned"] is True:
                print(f"    {letter}: ✓ drift start={result['drift_start_sec']}s, end={result['drift_end_sec']}s ({result['n_rows']} rows)")
            elif result["aligned"] is False:
                print(f"    {letter}: ✗ drift start={result['drift_start_sec']}s, end={result['drift_end_sec']}s")
                print(f"         tracker: {result['tracker_start']:.0f}s-{result['tracker_end']:.0f}s")
                print(f"         transcript: {result['transcript_first_in_cue']:.1f}s-{result['transcript_last_out_cue']:.1f}s")
                all_aligned = False
            else:
                print(f"    {letter}: – {result.get('error', 'unknown')}")

        if all_aligned:
            report("timestamp alignment (all segments)", "PASS")
        else:
            ans = ask("Some segments have >5s drift. Is this expected for this observation?")
            if ans == "y":
                report("timestamp alignment", "PASS", "drift accepted by user")
            else:
                report("timestamp alignment", "FAIL", "drift rejected")

    # ── 5. Get video files from Drive ──
    video_col = "Video Link"
    video_files = []
    if video_col in headers and row_idx and gdrive:
        v_url = get_hyperlink_url(ws, video_col, headers, row_idx)
        v_folder_id = extract_folder_id(v_url)
        if v_folder_id:
            video_files = list_drive_folder_videos(gdrive, v_folder_id, obsid)
            if video_files:
                print(f"\n  Video files found:")
                for vf in video_files:
                    print(f"    vid {vf['video_index']}: {vf['name']}")
                report(f"list video files ({len(video_files)} found)", "PASS")
            else:
                report("list video files", "FAIL", "none found")

    # ── 6. Clip videos per segment (with human verification) ──
    if video_files and segments:
        print(f"\n  Clipping segments...")
        os.makedirs(CLIPS_DIR, exist_ok=True)

        for letter in letters:
            vid_idx, start, end = segments[letter]
            duration = end - start

            matching_videos = [v for v in video_files if v["video_index"] == vid_idx]
            if not matching_videos:
                if len(video_files) == 1:
                    matching_videos = video_files
                else:
                    report(f"clip segment {letter}", "SKIP", f"no video for vid {vid_idx}")
                    continue

            vid = matching_videos[0]

            # Download video to local temp if not already there
            local_vid = os.path.join(CLIPS_DIR, vid["name"])
            if not os.path.exists(local_vid):
                print(f"    Downloading {vid['name']}...")
                f = gdrive.CreateFile({"id": vid["file_id"]})
                f.GetContentFile(local_vid)

            out_name = f"OBS-{row['Index']}_{letter}.mp4"
            out_path = os.path.join(CLIPS_DIR, out_name)

            print(f"\n    Clipping segment {letter} (vid {vid_idx}, {start:.0f}s-{end:.0f}s, {duration:.0f}s)...")
            clip_result = clip_segment(local_vid, start, end, out_path)

            if clip_result["success"]:
                dur_check = verify_clip_duration(out_path, duration)
                print(f"    → {out_name}: expected {duration:.0f}s, got {dur_check.get('actual_sec', '?')}s")

                if dur_check["match"]:
                    report(f"clip segment {letter} duration", "PASS",
                           f"{dur_check['actual_sec']}s (expected {dur_check['expected_sec']}s)")
                else:
                    report(f"clip segment {letter} duration", "FAIL",
                           f"drift {dur_check['drift_sec']}s")

                ans = ask(f"Open {out_path} and verify segment '{letter}' content. Does it look correct?")
                if ans == "y":
                    report(f"clip segment {letter} (human verified)", "PASS")
                elif ans == "skip":
                    report(f"clip segment {letter} (human verified)", "SKIP")
                else:
                    report(f"clip segment {letter} (human verified)", "FAIL", "rejected by user")
            else:
                report(f"clip segment {letter}", "FAIL", clip_result.get("stderr", "")[:100])
    elif not video_files:
        # Dry-run with fake paths to verify command generation
        print(f"\n  No video files — running dry-run clip commands...")
        for letter in letters[:3]:
            vid_idx, start, end = segments[letter]
            clip_result = clip_segment(
                f"/fake/OBS-25-{obsid:04d}_video_{vid_idx}.mov",
                start, end,
                f"/fake/clips/OBS-25-{obsid:04d}_{letter}.mp4",
                dry_run=True,
            )
            print(f"    {letter}: {clip_result['command']}")
        report("clip commands (dry run)", "PASS")


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
            print("\n  Not in Colab — skipping integration tests.\n")
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
