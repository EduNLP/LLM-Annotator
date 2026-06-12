# Author: Arthur Pan(The University of Edinburgh)
# Date: 2025/03/16
# Version: 0.1
import os

from argparse import ArgumentParser
from typing import List, Dict
from openai import OpenAI

import llm_annotator.prompt_parser
import llm_annotator.annotator
import llm_annotator.postprocess
from llm_annotator.registry import fetch_pipe
from llm_annotator.pipeline import Pipeline
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

from llm_annotator.dataloader import DataLoader
from llm_annotator.registry import simple_llm_pipe
from llm_annotator import utils, preprocess
from llm_annotator.llm import openai_annotate, anthropic_annotate, batch_anthropic_annotate


def annotate(
        model_list: List[str],
        obs_list: List[str] | str,
        feature: str,
        transcript_source: str,
        sheet_source: str,
        n_uttr: int,
        if_wait=False,
        system_prompt_path: str = "data/prompts/system_prompt.txt",
        prompt_path: str = "",
        annotation_prompt_path: str = "",
        if_test: bool = False,
        save_dir="",
        mode: str = "",
        fwd_context_count: int = 0,  # Defaults to 0 (no forward context)
        bwd_context_count: int = 0,  # Defaults to 0 (no backward context)
        clip_base_dir: str = "",  # Drive folder with per-line clips: {clip_base_dir}/{obsid}/{obsid}_{seg}_{line}.mp4
        use_video: bool = False,  # If True and clip_base_dir set, attach video clips for Gemini
        filter_if_override: list = None,
        extra_context_text: str = "",
):
    pipe = simple_llm_pipe(model_list=model_list,
                           obs_list=obs_list,
                           feature=feature,
                           transcript_source=transcript_source,
                           system_prompt_path=system_prompt_path,
                           prompt_path=prompt_path,
                           annotation_prompt_path=annotation_prompt_path,
                           sheet_source=sheet_source,
                           if_wait=if_wait,
                           if_test=if_test,
                           save_dir=save_dir,
                           n_uttr=n_uttr,
                           fwd_context_count=fwd_context_count,
                           bwd_context_count=bwd_context_count,
                           clip_base_dir=clip_base_dir,
                           use_video=use_video,
                           filter_if_override=filter_if_override,
                           extra_context_text=extra_context_text)
    pipe()


def fetch(timestamp: str = None,
          feature: str = None,
          save_dir: str = None):

    pipe = fetch_pipe(timestamp=timestamp,
                      feature=feature,
                      save_dir=save_dir)
    pipe()


def resume(feature: str,
           resume_batch_ids: dict,
           transcript_source: str,
           sheet_source: str,
           save_dir: str = "",
           timestamp: str = None):
    """Fetch results for a batch that was already submitted but not yet retrieved.

    Use this when Colab crashed after the API batch was submitted but before
    fetch() was called. Provide the batch IDs printed in the logs.

    Args:
        feature: Feature name (must match what was annotated).
        resume_batch_ids: Dict mapping model name to batch ID,
            e.g. {"gpt-5-mini": "batch_abc123"}.
        transcript_source: Same transcript_source used in the original run.
        sheet_source: Same sheet_source used in the original run.
        save_dir: Save directory. Uses "result/" if empty.
        timestamp: Timestamp string from the original run. If None, a new
            timestamp is created and stub metadata is written so fetch_pipe
            can locate the batch files.
    """
    import json
    from datetime import datetime
    from llm_annotator.utils import create_batch_dir, Batch, BatchRequestCounts

    if not resume_batch_ids:
        raise ValueError("resume_batch_ids is empty — nothing to resume.")

    ts = timestamp or datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    effective_save_dir = save_dir or "result"

    # Write stub metadata.json so fetch_pipe can reconstruct the DataLoader
    batch_dir = create_batch_dir(effective_save_dir, feature, ts)
    metadata = {
        "transcript_source": transcript_source,
        "sheet_source": sheet_source,
        "feature": feature,
        "timestamp": ts,
        "annotation_prompt_path": "",
    }
    meta_path = os.path.join(batch_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[resume] Wrote stub metadata to {meta_path}")

    # Write one stub batch JSON per model so load_batch_files can find them
    for model, batch_id in resume_batch_ids.items():
        stub = Batch(
            id=batch_id,
            status="completed",
            created_at="",
            expires_at="",
            request_counts=BatchRequestCounts(),
            stored_at=ts,
        ).to_dict()
        batch_path = os.path.join(batch_dir, f"{model}.json")
        with open(batch_path, "w") as f:
            json.dump(stub, f, indent=2)
        print(f"[resume] Wrote stub batch file for {model} (id={batch_id})")

    print(f"[resume] Fetching results for feature='{feature}' timestamp='{ts}'")
    fetch(timestamp=ts, feature=feature, save_dir=effective_save_dir)


def set_working_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
    os.chdir(repo_root)
    print(f"Working directory set to: {os.getcwd()}")


def main():
    # Example with OpenAI model
    annotate(model_list=["gpt-4o"],
               obs_list=["17"],
               feature="Mathcompetent",
               transcript_source="./data/mol.csv",
               sheet_source="1miMC8M_UkfY_3XhglTAdt9sC9H8D6d9slzC7_LgcLOc",
               prompt_path="data/prompts/base.txt",
               if_wait=True,
               n_uttr=10,
               mode="CoT",
             if_test=True)

    # Example with local Llama model
    # annotate(model_list=["llama-3b-local"],
    #            obs_list=["17"],
    #            feature="Mathcompetent",
    #            transcript_source="./data/mol.csv",
    #            sheet_source="1miMC8M_UkfY_3XhglTAdt9sC9H8D6d9slzC7_LgcLOc",
    #            prompt_path="data/prompts/base.txt",
    #            if_wait=True,
    #            n_uttr=10,
    #            if_test=True)

    #fetch(feature="Mathcompetent")


set_working_dir()

if __name__ == "__main__":
    main()
