"""Dry-run preview of the annotation pipeline.

Shows every step that will happen for the selected config without
making any API calls or spending credits. Includes:
  - Data summary (obs IDs, row counts per feature)
  - Feature rules that will be applied
  - Video segment alignment verification (if use_video)
  - Cost estimate
  - Step-by-step plan

Usage:
    from llm_annotator.preview import preview_pipeline
    preview_pipeline(config, gc=gc, validation_path=..., tracker_sheet_id=...)
"""

import os
import re
import math
import pandas as pd
from typing import Optional

from llm_annotator.config import ExperimentConfig
from llm_annotator.cost import estimate_cost


def preview_pipeline(
    config: ExperimentConfig,
    gc=None,
    validation_path: str = "",
    tracker_sheet_id: str = "",
    tracker_tab: str = "Tracker",
):
    """Print a complete dry-run preview of what the pipeline will do."""

    print("\n" + "=" * 70)
    print("  DRY-RUN PREVIEW — no API calls, no credits spent")
    print("=" * 70)

    # ── 1. Load transcript and count rows ──
    print("\n── 1. Data Summary ──\n")
    transcript_df = _load_transcript(config)
    if transcript_df is None:
        print("  ⚠️  Could not load transcript — check transcript_source path")
        return

    obs_ids = transcript_df["obsid"].astype(str).unique().tolist() if "obsid" in transcript_df.columns else []
    n_student = len(transcript_df[transcript_df["role"] == "Student"]) if "role" in transcript_df.columns else len(transcript_df)

    print(f"  Transcript:       {len(transcript_df)} total rows, {n_student} student utterances")
    print(f"  Observations:     {len(obs_ids)} obs → {', '.join(obs_ids[:10])}{'...' if len(obs_ids) > 10 else ''}")
    print(f"  Test mode:        {'YES (~20 rows)' if config.if_test else 'NO (all rows)'}")
    if config.if_test:
        n_student = min(n_student, 20)

    n_requests = math.ceil(n_student / config.n_uttr)
    print(f"  Requests:         ~{n_requests} ({config.n_uttr} utterances each, bwd={config.bwd_context_count} fwd={config.fwd_context_count})")

    # ── 2. Features and rules ──
    print("\n── 2. Features & Rules ──\n")
    feature_dict = _load_feature_dict(config)

    for feat in config.feature_list:
        rules = config.get_feature_rules(feat, feature_dict.get(feat))
        print(f"  {feat}:")
        if rules["filter_if"]:
            print(f"    filter_if:          {rules['filter_if']}")
            # Check if filter columns exist in transcript
            for col in rules["filter_if"]:
                if col in transcript_df.columns:
                    n_filtered = (transcript_df[col] == 1).sum()
                    print(f"      → '{col}' column found: {n_filtered} rows would be filtered out")
                else:
                    print(f"      → '{col}' column NOT in transcript (filter will be skipped)")
        if rules["linked_with"]:
            print(f"    linked_with:        {rules['linked_with']}")
        if rules["subcode_of"]:
            print(f"    subcode_of:         {rules['subcode_of']}")
        if rules["extra_context_type"]:
            ctx_key = rules["extra_context_type"]
            has_ctx = ctx_key in config.extra_context
            print(f"    extra_context_type: {ctx_key} {'✓ provided' if has_ctx else '⚠️  NOT provided in config'}")
        if not any([rules["filter_if"], rules["linked_with"], rules["subcode_of"], rules["extra_context_type"]]):
            print(f"    (no rules)")

    # ── 3. Models ──
    print(f"\n── 3. Models ──\n")
    for model in config.model_list:
        video_note = ""
        if config.use_video:
            from llm_annotator.cost import MODEL_PRICING
            pricing = MODEL_PRICING.get(model)
            if pricing and pricing[2]:
                video_note = " (video supported ✓)"
            else:
                video_note = " (video NOT supported ⚠️)"
        print(f"  • {model}{video_note}")

    # ── 4. Video verification ──
    if config.use_video and tracker_sheet_id and gc:
        print(f"\n── 4. Video Segment Verification ──\n")
        _verify_video_segments(config, gc, transcript_df, obs_ids, tracker_sheet_id, tracker_tab)
    elif config.use_video:
        print(f"\n── 4. Video ── (skipped, no tracker_sheet_id provided)\n")
    else:
        print(f"\n── 4. Video ── (disabled)\n")

    # ── 5. Cost estimate ──
    print(f"\n── 5. Cost Estimate ──")
    estimate_cost(config, transcript_df, feature_dict)

    # ── 6. Validation set check ──
    if validation_path and os.path.exists(validation_path):
        print(f"\n── 6. Validation Set ──\n")
        val_df = pd.read_csv(validation_path)
        val_obsids = val_df["obsid"].astype(str).unique().tolist() if "obsid" in val_df.columns else []
        overlap = set(obs_ids) & set(val_obsids)
        print(f"  Validation CSV:   {os.path.basename(validation_path)} ({len(val_df)} rows, {len(val_obsids)} obs)")
        print(f"  Overlap with run: {len(overlap)} obs → {', '.join(sorted(overlap)[:10])}")
        val_features = [c for c in val_df.columns if c.lower() in [f.lower() for f in config.feature_list]]
        print(f"  Evaluable features: {val_features if val_features else '(none match)'}")
    elif validation_path:
        print(f"\n── 6. Validation ── file not found: {validation_path}\n")

    # ── 7. Step plan ──
    print(f"\n── 7. Pipeline Steps ──\n")
    steps = [
        ("Load data", f"transcript ({len(transcript_df)} rows) + feature sheet"),
        ("Pre-process", f"filter to {len(obs_ids)} obs, assign roles"),
    ]

    for feat in config.feature_list:
        rules = config.get_feature_rules(feat, feature_dict.get(feat))
        if rules["filter_if"]:
            steps.append(("Filter rows", f"drop rows where {rules['filter_if']} == 1 for {feat}"))

    steps.append(("Build prompts", f"{n_requests} requests × {len(config.feature_list)} features × {len(config.model_list)} models"))

    if config.resume_batch_ids:
        steps.append(("Resume batches", f"fetch results for {list(config.resume_batch_ids.keys())}"))
    else:
        steps.append(("Submit batches", f"{len(config.model_list)} model(s) × {len(config.feature_list)} feature(s)"))

    if config.if_wait:
        steps.append(("Wait & fetch", "poll until all batches complete"))
    else:
        steps.append(("Return batch IDs", "save IDs to resume later"))

    if validation_path:
        steps.append(("Evaluate", f"F1/precision/recall against {os.path.basename(validation_path)}"))
    steps.append(("Log results", "append metrics to results sheet"))

    for i, (name, detail) in enumerate(steps, 1):
        print(f"  {i:2d}. {name:<20s} → {detail}")

    print(f"\n{'=' * 70}")
    print(f"  Ready to run. Use run_pipeline() to execute.")
    print(f"{'=' * 70}\n")


def _load_transcript(config) -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(config.transcript_source):
            df = pd.read_csv(config.transcript_source)
        else:
            return None

        if "role" not in df.columns and "speaker" in df.columns:
            s = df["speaker"].astype(str)
            teacher_like = (
                s.str.contains("teacher", case=False, na=False)
                | s.str.contains(r"\.", regex=True, na=False)
            )
            teacher_like = teacher_like | df["speaker"].isna()
            df = df.copy()
            df["role"] = "Student"
            df.loc[teacher_like, "role"] = "Teacher"

        if isinstance(config.obs_list, str) and config.obs_list == "all":
            pass
        elif isinstance(config.obs_list, list):
            df["obsid"] = df["obsid"].astype(str)
            df = df[df["obsid"].isin(config.obs_list)]

        return df
    except Exception as e:
        print(f"  Error loading transcript: {e}")
        return None


def _load_feature_dict(config) -> dict:
    try:
        from llm_annotator.dataloader import DataLoader, generate_features
        dl = DataLoader(sheet_source=config.sheet_source, transcript_source=config.transcript_source)
        feature_dict = {}
        for feat in config.feature_list:
            _, fd = generate_features(dl, feature=feat)
            feature_dict.update(fd)
        return feature_dict
    except Exception as e:
        print(f"  [preview] Could not load feature dict: {e}")
        return {}


def _verify_video_segments(config, gc, transcript_df, obs_ids, tracker_sheet_id, tracker_tab):
    try:
        from llm_annotator.video_loader import parse_segment_timestamps
        from llm_annotator.video_clipper import verify_segment_alignment, tracker_time_to_sec
    except ImportError as e:
        print(f"  Could not import video modules: {e}")
        return

    spreadsheet = gc.open_by_key(tracker_sheet_id)
    ws = spreadsheet.worksheet(tracker_tab)
    all_data = ws.get_all_values()
    headers = all_data[0]

    # Find relevant columns
    idx_col = next((i for i, h in enumerate(headers) if h.strip().lower() == "index"), None)
    ts_col = next((i for i, h in enumerate(headers) if "segment" in h.lower() and "timestamp" in h.lower()), None)
    transcript_col = next((i for i, h in enumerate(headers) if "deidentified" in h.lower() and "link" in h.lower()), None)

    if idx_col is None or ts_col is None:
        print("  ⚠️  Could not find Index or Segment Timestamps column in Tracker")
        return

    for obsid in obs_ids[:5]:  # check first 5 obs
        # Find tracker row
        tracker_row = None
        for row in all_data[1:]:
            if idx_col < len(row):
                m = re.search(r"\d{2}-0*(\d+)", row[idx_col].strip())
                if m and str(int(m.group(1))) == str(obsid):
                    tracker_row = row
                    break

        if not tracker_row:
            print(f"  obs {obsid}: not found in Tracker")
            continue

        ts_text = tracker_row[ts_col] if ts_col < len(tracker_row) else ""
        if not ts_text.strip():
            print(f"  obs {obsid}: no segment timestamps")
            continue

        segments = parse_segment_timestamps(ts_text)
        if not segments:
            print(f"  obs {obsid}: could not parse timestamps")
            continue

        # Get transcript rows for this obs
        obs_transcript = transcript_df[transcript_df["obsid"].astype(str) == str(obsid)]
        if obs_transcript.empty:
            print(f"  obs {obsid}: no transcript rows")
            continue

        # Filter dash segments
        if "segment" in obs_transcript.columns:
            obs_transcript = obs_transcript[
                ~obs_transcript["segment"].astype(str).str.strip().isin(["-", "—", "–", ""])
            ]

        if "in_cue" not in obs_transcript.columns:
            print(f"  obs {obsid}: no in_cue column — skipping alignment check")
            continue

        # Build transcript row dicts for verification
        t_rows = obs_transcript.to_dict("records")

        n_segments = len(segments)
        n_aligned = 0
        issues = []

        for letter in sorted(segments.keys()):
            vid_idx, start, end = segments[letter]
            result = verify_segment_alignment(letter, start, end, t_rows, tolerance_sec=5.0)
            if result["aligned"] is True:
                n_aligned += 1
            elif result["aligned"] is False:
                issues.append(f"{letter}(drift: start={result['drift_start_sec']}s end={result['drift_end_sec']}s)")
            # None = no transcript rows for segment

        status = "✓" if n_aligned == n_segments else "⚠️"
        detail = f"{n_aligned}/{n_segments} segments aligned"
        if issues:
            detail += f", issues: {', '.join(issues[:3])}"
        print(f"  obs {obsid}: {status} {detail} ({len(obs_transcript)} transcript rows)")
