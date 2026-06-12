"""Video metadata loader for whole-segment annotation.

Reads an observation metadata Google Sheet with columns:
  - Index:              obsid in "25-0195" format (e.g. obsid 195 → "25-0195")
  - Video Link:         hyperlink to a Drive folder containing OBS-25-XXXX_Video* files
  - Segment Timestamps: free-text block mapping segment letters to time ranges

Provides three public functions:
  load_obs_sheet()         → DataFrame with parsed segment dicts
  parse_segment_timestamps() → {letter: (video_index, start_sec, end_sec)}
  get_segment_duration()   → seconds for a given obsid + segment letter

# TODO (future sprint): add download_segment(obsid, letter, dest_path) that
# trims the Drive video to the correct time range and returns a local path
# suitable for sending to Gemini as a multimodal input.
"""

import re
import pandas as pd
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
SegmentMap = Dict[str, Tuple[int, float, float]]
# segment_letter → (video_index, start_seconds, end_seconds)


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def _time_to_sec(t: str) -> float:
    """Convert 'M:SS' or 'MM:SS' or 'H:MM:SS' to total seconds."""
    parts = t.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Unrecognised time format: '{t}'")


# Matches lines like:  a 3:51-5:35 warm-up
#                      b 10:57-11:49
_INTRO_LINE = re.compile(r"^\s*intro\s+", re.IGNORECASE)

_SEGMENT_LINE = re.compile(
    r"^\s*([a-z])\s+"           # segment letter
    r"(\d{1,2}:\d{2}(?::\d{2})?)"  # start time
    r"\s*[-–]\s*"
    r"(\d{1,2}:\d{2}(?::\d{2})?)"  # end time
    r"(?:\s+.*)?$",             # optional label — ignored
    re.IGNORECASE,
)

# Matches "vid 1", "vid 2", "video 1", etc.
_VID_HEADER = re.compile(r"^\s*vid(?:eo)?\s*(\d+)", re.IGNORECASE)


def parse_segment_timestamps(text: str) -> SegmentMap:
    """Parse a free-text Segment Timestamps block into a SegmentMap.

    Handles both single-video and multi-video formats:

    Single video:
        a 3:51-5:35 warm-up
        b 10:57-11:49 warm-up
        c 14:50-29:26 activity 1

    Multiple videos:
        vid 1
        a 1:37-2:00 warm-up
        b 3:07-4:16 warm-up
        vid 2
        f 0:06-2:09 activity 1
        g 9:30-20:48 activity 2

    Returns:
        Dict mapping segment letter (lowercase) to
        (video_index: int, start_sec: float, end_sec: float).
        Lines that don't match the segment pattern are silently skipped.
    """
    result: SegmentMap = {}
    current_video = 1

    for line in text.splitlines():
        vid_match = _VID_HEADER.match(line)
        if vid_match:
            current_video = int(vid_match.group(1))
            continue

        if _INTRO_LINE.match(line):
            continue

        seg_match = _SEGMENT_LINE.match(line)
        if seg_match:
            letter = seg_match.group(1).lower()
            start  = _time_to_sec(seg_match.group(2))
            end    = _time_to_sec(seg_match.group(3))
            result[letter] = (current_video, start, end)

    return result


# ---------------------------------------------------------------------------
# Sheet loader
# ---------------------------------------------------------------------------

def load_obs_sheet(tracker_sheet_id: str, gc=None) -> pd.DataFrame:
    """Load the Tracker sheet and parse segment timestamps.

    Args:
        tracker_sheet_id: Google Sheet ID for the Tracker, containing Index,
            Video Link, and Segment Timestamps columns.
        gc: Authorised gspread client. If None, attempts to import from
            the dataloader module (works when running in Colab where
            dataloader.py already authenticated).

    Returns:
        DataFrame with columns:
            obsid_raw   – raw value from Index column (e.g. "25-0195")
            obsid       – numeric suffix as string (e.g. "195")
            video_link  – Drive folder URL string (hyperlink text stripped)
            segments    – SegmentMap dict parsed from Segment Timestamps
    """
    if gc is None:
        try:
            from llm_annotator.dataloader import gc as _gc
            gc = _gc
        except Exception:
            pass

    if gc is None:
        raise RuntimeError(
            "No gspread client available. Pass gc= or run in Colab after auth."
        )

    sheet = gc.open_by_key(tracker_sheet_id).sheet1
    data = sheet.get_all_records()
    raw_df = pd.DataFrame(data)

    required = {"Index", "Video Link", "Segment Timestamps"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(
            f"obs_sheet is missing expected columns: {missing}. "
            f"Found: {list(raw_df.columns)}"
        )

    rows = []
    for _, row in raw_df.iterrows():
        obsid_raw = str(row["Index"]).strip()
        # Extract numeric suffix: "25-0195" → "195" (strip leading zeros then re-str)
        m = re.search(r"-0*(\d+)$", obsid_raw)
        obsid = str(int(m.group(1))) if m else obsid_raw

        video_link = str(row["Video Link"]).strip()

        ts_text = str(row.get("Segment Timestamps", "")).strip()
        segments = parse_segment_timestamps(ts_text) if ts_text else {}

        rows.append({
            "obsid_raw":  obsid_raw,
            "obsid":      obsid,
            "video_link": video_link,
            "segments":   segments,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Duration helper (used by cost estimator)
# ---------------------------------------------------------------------------

def get_segment_duration(obs_df: pd.DataFrame,
                         obsid: str,
                         segment_letter: str) -> Optional[float]:
    """Return duration in seconds for an obsid + segment letter.

    Args:
        obs_df: DataFrame returned by load_obs_sheet().
        obsid:  Observation ID as string (e.g. "195").
        segment_letter: Single letter (e.g. "b").

    Returns:
        Duration in seconds, or None if obsid/letter not found.
    """
    row = obs_df[obs_df["obsid"] == str(obsid)]
    if row.empty:
        return None
    segments: SegmentMap = row.iloc[0]["segments"]
    entry = segments.get(segment_letter.lower())
    if entry is None:
        return None
    _, start, end = entry
    return max(0.0, end - start)


def total_video_seconds(obs_df: pd.DataFrame, obsid: str) -> float:
    """Sum of all segment durations for an obsid. Used for cost estimation."""
    row = obs_df[obs_df["obsid"] == str(obsid)]
    if row.empty:
        return 0.0
    segments: SegmentMap = row.iloc[0]["segments"]
    return sum(max(0.0, e - s) for (_, s, e) in segments.values())
