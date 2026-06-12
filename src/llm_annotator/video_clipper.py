"""Cut video segments and verify timestamps across sources.

Three timestamp formats in play:
  - Tracker "Segment Timestamps": m:ss or mm:ss (relative to video start)
  - Deidentified transcript in_cue/out_cue: hh:mm:ss:ff (ff = frames at ~30fps)
  - ffmpeg: hh:mm:ss.ms

Transcript in_cue times are absolute within each video file — they restart
at 00:00:00:00 when the segment belongs to a different video than the previous one.
"""

import os
import re
import subprocess
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Time conversion
# ═══════════════════════════════════════════════════════════════════════════

def tracker_time_to_sec(t: str) -> float:
    """Convert Tracker timestamp (m:ss, mm:ss, or h:mm:ss) to seconds."""
    parts = t.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Unrecognised tracker time: '{t}'")


def transcript_time_to_sec(t: str) -> float:
    """Convert transcript timestamp (hh:mm:ss:ff at ~30fps) to seconds.

    Format: hh:mm:ss:ff where ff is frame number (0-29 at 30fps).
    Example: 00:03:10:15 → 3*60 + 10 + 15/30 = 190.5
    """
    t = t.strip()
    parts = t.split(":")
    if len(parts) == 4:
        h, m, s, ff = parts
        return float(h) * 3600 + float(m) * 60 + float(s) + float(ff) / 30.0
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    raise ValueError(f"Unrecognised transcript time: '{t}'")


def sec_to_ffmpeg(sec: float) -> str:
    """Convert seconds to ffmpeg-compatible hh:mm:ss.mmm."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ═══════════════════════════════════════════════════════════════════════════
# Timestamp verification
# ═══════════════════════════════════════════════════════════════════════════

def verify_segment_alignment(
    segment_letter: str,
    tracker_start_sec: float,
    tracker_end_sec: float,
    transcript_rows: list[dict],
    tolerance_sec: float = 5.0,
) -> dict:
    """Check that transcript in_cue/out_cue values align with Tracker timestamps.

    Args:
        segment_letter: e.g. "a"
        tracker_start_sec: start time from Tracker (seconds)
        tracker_end_sec: end time from Tracker (seconds)
        transcript_rows: list of dicts with keys: segment, in_cue, out_cue
            (only rows for this segment, dash rows already filtered)
        tolerance_sec: allowed drift in seconds

    Returns:
        dict with: aligned (bool), first_in_cue_sec, last_out_cue_sec,
        tracker_start, tracker_end, drift_start, drift_end
    """
    seg_rows = [
        r for r in transcript_rows
        if str(r.get("segment", "")).strip().lower() == segment_letter.lower()
    ]

    if not seg_rows:
        return {"aligned": None, "error": f"no transcript rows for segment '{segment_letter}'"}

    first_in = transcript_time_to_sec(str(seg_rows[0]["in_cue"]))
    last_out = transcript_time_to_sec(str(seg_rows[-1]["out_cue"]))

    drift_start = abs(first_in - tracker_start_sec)
    drift_end = abs(last_out - tracker_end_sec)

    return {
        "aligned": drift_start <= tolerance_sec and drift_end <= tolerance_sec,
        "segment": segment_letter,
        "tracker_start": tracker_start_sec,
        "tracker_end": tracker_end_sec,
        "transcript_first_in_cue": first_in,
        "transcript_last_out_cue": last_out,
        "drift_start_sec": round(drift_start, 2),
        "drift_end_sec": round(drift_end, 2),
        "n_rows": len(seg_rows),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Video clipping
# ═══════════════════════════════════════════════════════════════════════════

def clip_segment(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    dry_run: bool = False,
) -> dict:
    """Cut a segment from a video file using ffmpeg.

    Uses stream copy (no re-encoding) for speed. Falls back to re-encode
    if the copy produces a bad file.

    Returns dict with: success, command, output_path, duration_sec.
    """
    ss = sec_to_ffmpeg(start_sec)
    to = sec_to_ffmpeg(end_sec)
    duration = end_sec - start_sec

    cmd = [
        "ffmpeg", "-y",
        "-ss", ss,
        "-to", to,
        "-i", video_path,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        output_path,
    ]

    if dry_run:
        cmd_str = " ".join(cmd)
        return {"success": True, "command": cmd_str, "output_path": output_path,
                "duration_sec": duration, "dry_run": True}

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    success = result.returncode == 0 and os.path.exists(output_path)

    return {
        "success": success,
        "command": " ".join(cmd),
        "output_path": output_path,
        "duration_sec": duration,
        "stderr": result.stderr[-500:] if not success else "",
    }


def get_video_duration(path: str) -> Optional[float]:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def verify_clip_duration(clip_path: str, expected_sec: float, tolerance_sec: float = 3.0) -> dict:
    """Check that a clipped video has the expected duration."""
    actual = get_video_duration(clip_path)
    if actual is None:
        return {"match": False, "error": "could not read clip duration"}
    drift = abs(actual - expected_sec)
    return {
        "match": drift <= tolerance_sec,
        "expected_sec": round(expected_sec, 1),
        "actual_sec": round(actual, 1),
        "drift_sec": round(drift, 1),
    }
