import llm_annotator.utils as utils
import pandas as pd

from typing import List, Dict


@utils.component("pre-process")
def pre_process_transcript(transcript_df: pd.DataFrame, obs_list: List[str] | str):
    # Ensure role column is in the transcript
    if "role" not in transcript_df.columns and "speaker" in transcript_df.columns:
        s = transcript_df["speaker"].astype(str)
        teacher_like = (
            s.str.contains("teacher", case=False, na=False)
            | s.str.contains(r"\.", regex=True, na=False)  # e.g. Ms., Mr., Teacher 1
        )
        teacher_like = teacher_like | transcript_df["speaker"].isna()
        transcript_df = transcript_df.copy()
        transcript_df["role"] = "Student"
        transcript_df.loc[teacher_like, "role"] = "Teacher"
    
    if isinstance(obs_list, str) and obs_list == "all":
        obs_list = transcript_df["obsid"].unique().tolist()
    elif not isinstance(obs_list, list):
        raise ValueError(f"obs_list should be a list, got {type(obs_list)} instead.")

    transcript_df["obsid"] = transcript_df["obsid"].astype(str)
    transcript_df = transcript_df[transcript_df["obsid"].isin(obs_list)]
    return "transcript_df", transcript_df


def filter_by_feature_rules(df: pd.DataFrame, feature_meta: Dict) -> pd.DataFrame:
    """Drop rows that should be excluded based on the feature's filter_if rules.

    For each code name listed in feature_meta["filter_if"], if a column with
    that name exists in df and has value 1, the row is excluded. This lets
    sheet authors control filtering (e.g. remove offtask-labeled utterances
    before annotating Directions) without hardcoding any feature names here.

    Args:
        df: Transcript DataFrame, may contain pre-existing annotation columns.
        feature_meta: Feature dict produced by generate_features(), must have
            a "filter_if" key (list of str, may be empty).

    Returns:
        Filtered copy of df.
    """
    filter_codes = feature_meta.get("filter_if", [])
    if not filter_codes:
        return df

    mask = pd.Series(False, index=df.index)
    for code in filter_codes:
        if code in df.columns:
            mask = mask | (df[code] == 1)
        else:
            print(f"[filter_by_feature_rules] Column '{code}' not found in df — skipping filter for this code.")

    n_dropped = mask.sum()
    if n_dropped:
        print(f"[filter_by_feature_rules] Dropped {n_dropped} rows where {filter_codes} == 1.")
    return df[~mask].copy()


@utils.component("filter-by-rules")
def apply_feature_filter(transcript_df: pd.DataFrame, feature_dict: Dict, feature: str,
                         filter_if_override: list = None) -> tuple:
    """Pipeline component: filter transcript_df using the feature's filter_if rules."""
    meta = feature_dict.get(feature, {})
    if filter_if_override is not None:
        meta = dict(meta)
        meta["filter_if"] = filter_if_override
    return "transcript_df", filter_by_feature_rules(transcript_df, meta)
