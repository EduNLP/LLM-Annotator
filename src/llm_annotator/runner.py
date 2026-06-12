"""High-level experiment runner.

Single entry point that handles: cost estimation → annotation → evaluation → sheet logging.
Called from the Colab with one line: `run_pipeline(config, ...)`.
"""

import os
import pandas as pd
from datetime import datetime
from typing import Optional

from llm_annotator.config import ExperimentConfig
from llm_annotator.dataloader import DataLoader, generate_features
from llm_annotator.preprocess import pre_process_transcript, filter_by_feature_rules
from llm_annotator.cost import estimate_cost
from llm_annotator.main import annotate, resume, fetch
from llm_annotator.session_lock import acquire_lock, release_lock


def run_pipeline(
    config: ExperimentConfig,
    results_sheet_id: str = "",
    validation_path: str = "",
    gc=None,
    verbose: bool = True,
    user: str = "",
    force_lock: bool = False,
):
    """Run the full annotation pipeline from config.

    Steps:
        1. Load data & estimate cost
        2. Annotate (or resume from existing batch)
        3. Evaluate against validation set (if validation_path provided)
        4. Log metrics to Google Sheets (if results_sheet_id provided)

    Args:
        config: ExperimentConfig with all params.
        results_sheet_id: Google Sheet ID for logging metrics. Skip if empty.
        validation_path: Path to validation CSV. Skip eval if empty.
        gc: Authorised gspread client (needed for sheet logging).
        verbose: Print detailed logs. If False, only prints cost table and final metrics.
    """
    _log = print if verbose else lambda *a, **k: None

    # ── 0. Acquire lock ──
    if results_sheet_id and gc:
        if not acquire_lock(gc, results_sheet_id, user=user or "anonymous", force=force_lock):
            return pd.DataFrame()

    try:
        return _run_inner(config, results_sheet_id, validation_path, gc, verbose, _log)
    finally:
        if results_sheet_id and gc:
            release_lock(gc, results_sheet_id)


def _run_inner(config, results_sheet_id, validation_path, gc, verbose, _log):
    # ── 1. Cost estimate ──
    _log("Loading data...")
    dl = DataLoader(sheet_source=config.sheet_source, transcript_source=config.transcript_source)
    _, transcript_df = pre_process_transcript(dl.transcript_df, config.obs_list)

    feature_dict = {}
    for feat in config.feature_list:
        _, fd = generate_features(dl, feature=feat)
        feature_dict.update(fd)

    print("\n── Cost Estimate ──")
    cost_estimates = estimate_cost(config, transcript_df, feature_dict)

    # ── 2. Annotate ──
    start_ts = datetime.utcnow().isoformat()
    _log(f"\nStarted: {start_ts}")

    for feature in config.feature_list:
        _log(f"\n── {feature} ──")

        # Apply feature rule overrides
        rules = config.get_feature_rules(feature, feature_dict.get(feature))
        if rules["filter_if"]:
            _log(f"  Will filter rows where {rules['filter_if']} == 1")

        if config.resume_batch_ids:
            resume(
                feature=feature,
                resume_batch_ids=config.resume_batch_ids,
                transcript_source=config.transcript_source,
                sheet_source=config.sheet_source,
                save_dir=config.save_dir,
            )
        else:
            annotate(
                model_list=config.model_list,
                obs_list=config.obs_list,
                feature=feature,
                transcript_source=config.transcript_source,
                sheet_source=config.sheet_source,
                n_uttr=config.n_uttr,
                if_wait=config.if_wait,
                if_test=config.if_test,
                save_dir=config.save_dir,
                bwd_context_count=config.bwd_context_count,
                fwd_context_count=config.fwd_context_count,
                filter_if_override=rules["filter_if"] if rules["filter_if"] else None,
                use_video=config.use_video,
            )

    finish_ts = datetime.utcnow().isoformat()
    _log(f"\nFinished: {finish_ts}")

    # ── 3. Evaluate ──
    results_df = pd.DataFrame()
    if validation_path and os.path.exists(validation_path):
        results_df = _evaluate(config, validation_path, verbose)
    elif validation_path:
        print(f"[skip] Validation path not found: {validation_path}")

    # ── 4. Log to sheets ──
    if results_sheet_id and gc and not results_df.empty:
        _log_to_sheets(results_df, results_sheet_id, gc, verbose)

    return results_df


def _evaluate(config, validation_path, verbose):
    from sklearn.metrics import f1_score, precision_score, recall_score
    from llm_annotator.utils import find_latest_dir

    _log = print if verbose else lambda *a, **k: None
    val_df = pd.read_csv(validation_path)
    rows = []

    for feature in config.feature_list:
        feat_dir = os.path.join(config.save_dir, feature)
        ts = find_latest_dir(feat_dir)
        if not ts:
            _log(f"[skip] no results for {feature}")
            continue

        pred_path = os.path.join(feat_dir, ts, "atn_df.csv")
        if not os.path.exists(pred_path):
            _log(f"[skip] {pred_path} not found")
            continue

        pred_df = pd.read_csv(pred_path)
        true_col = feature.lower()

        for model in config.model_list:
            if model not in pred_df.columns or true_col not in val_df.columns:
                continue
            merged = pred_df[["uttid", model]].merge(
                val_df[["uttid", true_col]], on="uttid"
            ).dropna()
            y_pred = merged[model].astype(int)
            y_true = merged[true_col].astype(int)

            rows.append({
                "Feature": feature, "Model": model, "Timestamp": ts,
                "N": len(merged),
                "F1": round(f1_score(y_true, y_pred, zero_division=0), 3),
                "Precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
                "Recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            })

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        try:
            from tabulate import tabulate
            print("\n── Results ──")
            print(tabulate(results_df, headers="keys", tablefmt="github", showindex=False))
        except ImportError:
            print("\n── Results ──")
            print(results_df.to_string(index=False))

    return results_df


def _log_to_sheets(results_df, results_sheet_id, gc, verbose):
    _log = print if verbose else lambda *a, **k: None
    spreadsheet = gc.open_by_key(results_sheet_id)
    METRIC_ROWS = ["Feature", "Model", "Timestamp", "N", "F1", "Precision", "Recall"]

    for _, row in results_df.iterrows():
        import gspread
        try:
            ws = spreadsheet.worksheet(row["Feature"])
        except gspread.WorksheetNotFound:
            ws = spreadsheet.add_worksheet(title=row["Feature"], rows=20, cols=50)

        all_vals = ws.get_all_values()
        next_col = len(all_vals[0]) + 1 if all_vals else 1
        if next_col == 1:
            for i, label in enumerate(METRIC_ROWS, 1):
                ws.update_cell(i, 1, label)
            next_col = 2

        for i, label in enumerate(METRIC_ROWS, 1):
            ws.update_cell(i, next_col, str(row.get(label, "")))

        _log(f"{row['Feature']} / {row['Model']}  F1={row['F1']}  → col {next_col}")
