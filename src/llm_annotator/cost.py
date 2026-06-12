"""Cost estimation for annotation runs.

Provides estimate_cost() which prints a per-model cost table before
any API calls are made. All prices are approximate and should be
verified against current provider pricing pages.

Usage in Colab:
    from llm_annotator.cost import estimate_cost
    estimate_cost(config, transcript_df, feature_dict)
"""

import math
import pandas as pd
from typing import Dict

# ---------------------------------------------------------------------------
# Pricing table  (USD per 1M tokens, as of June 2025 — verify before relying)
# Format: model_key -> (input_$/1M, output_$/1M, video_$/1M_tokens or None)
# Video pricing is expressed as $/1M tokens (Gemini charges video as tokens).
# Roughly: 1 sec video ≈ 263 tokens (Gemini's published rate).
# ---------------------------------------------------------------------------
GEMINI_VIDEO_TOKENS_PER_SEC = 263  # Gemini's published rate

MODEL_PRICING: Dict[str, tuple] = {
    # (input $/1M, output $/1M, supports_video)
    "gpt-4o":           (2.50,   10.00,  False),
    "gpt-5-nano":       (0.15,    0.60,  False),
    "gpt-5-mini":       (0.40,    1.60,  False),
    "gpt-5.1":          (2.00,    8.00,  False),
    "gpt-5.2":          (3.00,   12.00,  False),
    "claude-3-5":       (3.00,   15.00,  False),
    "gemini-2.5-flash-lite": (0.10,  0.40, True),
    "gemini-2.5-flash": (0.30,   2.50,  True),
    "gemini-2.5-pro":   (1.25,  10.00,  True),
    "gemini-3-flash-preview": (0.30, 2.50, True),
    "gemini-3-pro-preview":   (1.25, 10.00, True),
    "mistral":          (2.00,   6.00,  False),
    "deepseek-v3":      (0.27,   1.10,  False),
    "llama-3.2-3b":     (0.06,   0.06,  False),  # Together.ai approx
    "llama-3.3-70b":    (0.88,   0.88,  False),
    "llama-3b-local":   (0.00,   0.00,  False),
    "llama-70b-local":  (0.00,   0.00,  False),
}

CHARS_PER_TOKEN = 4  # rough approximation


def _count_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def estimate_cost(config, transcript_df: pd.DataFrame, feature_dict: Dict,
                  avg_segment_duration_sec: float = 120.0) -> Dict:
    """Estimate and print annotation cost per model.

    Args:
        config: ExperimentConfig instance.
        transcript_df: Loaded transcript DataFrame (after pre-processing).
        feature_dict: Feature metadata dict from generate_features().
        avg_segment_duration_sec: Used for video cost estimate when
            obs_sheet_source is not yet resolved. Default 2 min/segment.

    Returns:
        Dict mapping model name -> estimated USD cost.
    """
    # --- Estimate prompt size ---
    # System prompt: ~500 tokens typical
    system_tokens = 500

    # Feature definition + examples per feature
    def_tokens = 0
    for feat, meta in feature_dict.items():
        def_tokens += _count_tokens(meta.get("definition", ""))
        for key in ["example1", "example2", "example3",
                    "nonexample1", "nonexample2", "nonexample3"]:
            def_tokens += _count_tokens(meta.get(key, ""))

    # Utterance tokens: average dialogue length × rows / n_uttr batches
    if "dialogue" in transcript_df.columns:
        avg_utt_chars = transcript_df["dialogue"].dropna().astype(str).str.len().mean()
    else:
        avg_utt_chars = 60  # fallback
    utt_tokens_per_request = math.ceil(avg_utt_chars * config.n_uttr / CHARS_PER_TOKEN)

    # Context window tokens
    context_tokens = math.ceil(avg_utt_chars * (config.bwd_context_count + config.fwd_context_count) / CHARS_PER_TOKEN)

    input_tokens_per_request = system_tokens + def_tokens + utt_tokens_per_request + context_tokens
    output_tokens_per_request = config.n_uttr * 10  # ~10 tokens per JSON label

    n_student_rows = len(transcript_df[transcript_df.get("role", pd.Series(["Student"] * len(transcript_df))) == "Student"]) \
        if "role" in transcript_df.columns else len(transcript_df)

    if config.if_test:
        n_student_rows = min(n_student_rows, 20)

    n_requests = math.ceil(n_student_rows / config.n_uttr)

    total_input_tokens = input_tokens_per_request * n_requests
    total_output_tokens = output_tokens_per_request * n_requests

    # --- Video tokens ---
    video_tokens = 0
    if config.use_video:
        if config.tracker_sheet_id:
            try:
                from llm_annotator.video_loader import load_obs_sheet, total_video_seconds
                obs_df = load_obs_sheet(config.tracker_sheet_id)
                obs_ids = transcript_df["obsid"].astype(str).unique() \
                    if "obsid" in transcript_df.columns else []
                total_secs = sum(total_video_seconds(obs_df, oid) for oid in obs_ids)
                video_tokens = total_secs * GEMINI_VIDEO_TOKENS_PER_SEC
                print(f"  [video] {len(obs_ids)} obs, {total_secs:.0f}s total → {video_tokens:.0f} tokens")
            except Exception as e:
                print(f"  [video] Could not load obs sheet ({e}), falling back to avg estimate.")
                n_obs = len(transcript_df["obsid"].unique()) if "obsid" in transcript_df.columns else 1
                video_tokens = n_obs * avg_segment_duration_sec * GEMINI_VIDEO_TOKENS_PER_SEC
        else:
            n_obs = len(transcript_df["obsid"].unique()) if "obsid" in transcript_df.columns else 1
            video_tokens = n_obs * avg_segment_duration_sec * GEMINI_VIDEO_TOKENS_PER_SEC

    # --- Compute costs ---
    results = {}
    lines = []
    lines.append(f"\n{'='*62}")
    lines.append(f"  Cost Estimate  |  {n_student_rows} utterances  |  {n_requests} requests")
    lines.append(f"{'='*62}")
    lines.append(f"  {'Model':<28} {'Text $':>8}  {'Video $':>8}  {'Total $':>8}")
    lines.append(f"  {'-'*56}")

    for model in config.model_list:
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            lines.append(f"  {model:<28} {'(no pricing data)':>26}")
            results[model] = None
            continue

        in_price, out_price, supports_video = pricing
        text_cost = (total_input_tokens / 1_000_000 * in_price +
                     total_output_tokens / 1_000_000 * out_price)

        vid_cost = 0.0
        if config.use_video and supports_video:
            vid_cost = video_tokens / 1_000_000 * in_price
        elif config.use_video and not supports_video:
            vid_cost = float("nan")

        total = text_cost + (vid_cost if not math.isnan(vid_cost) else 0)
        results[model] = total

        vid_str = f"${vid_cost:.4f}" if not math.isnan(vid_cost) else "N/A"
        lines.append(f"  {model:<28} ${text_cost:>7.4f}  {vid_str:>8}  ${total:>7.4f}")

    lines.append(f"{'='*62}\n")
    lines.append(f"  Token breakdown per request:")
    lines.append(f"    system prompt:  ~{system_tokens} tokens")
    lines.append(f"    feature defs:   ~{def_tokens} tokens")
    lines.append(f"    utterances:     ~{utt_tokens_per_request} tokens ({config.n_uttr} uttr × ~{int(avg_utt_chars)} chars)")
    lines.append(f"    context:        ~{context_tokens} tokens (bwd={config.bwd_context_count}, fwd={config.fwd_context_count})")
    if config.use_video:
        lines.append(f"    video:          ~{int(avg_segment_duration_sec)}s/segment avg × {GEMINI_VIDEO_TOKENS_PER_SEC} tokens/sec")
    lines.append(f"{'='*62}\n")

    print("\n".join(lines))
    return results
