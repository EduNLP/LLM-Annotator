import pandas as pd
import time
import os
import json


from typing import Dict, List, Union
from datetime import datetime

import anthropic
import openai

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from llm_annotator import utils
from llm_annotator.llm import batch_anthropic_annotate, batch_openai_annotate, store_batch, store_meta
from llm_annotator.utils import load_batch_files


def mark_ineligible_rows(model_list: List[str],
                         feature_dict: Dict,
                         transcript_df: pd.DataFrame,
                         min_len):
    # Create separate dfs for individual features
    atn_feature_dfs = {feature_name: transcript_df.copy() for feature_name in feature_dict.keys()}

    # Filter out ineligible rows
    eligible_rows = transcript_df[
    (transcript_df['role'].str.lower() == 'student') & 
    (
        ~transcript_df['dialogue'].str.contains("unintelligible", case=False, na=False) | 
        (transcript_df['dialogue'].str.split().str.len() >= 7)
    )
]
    ineligible_rows = transcript_df.index.difference(eligible_rows.index)

    # Mark ineligible rows with Nones
    for model_name in model_list:
        for feature_name in feature_dict:
            for idx in ineligible_rows:
                atn_feature_dfs[feature_name].at[idx, model_name] = None

    return eligible_rows, atn_feature_dfs


def group_obs(transcript_df: pd.DataFrame,
              if_context: bool,
              obs_list: List[str]):
    obs_groups = {}
    if if_context:
        for obs_id in obs_list:
            obs_groups[obs_id] = transcript_df[transcript_df['obsid'] == obs_id].index.tolist()
    return obs_groups


import re
from typing import Dict, List
import json
from typing import Dict, Any
import uuid
from typing import Optional


# Note: Assuming 'Request' and 'MessageCreateParamsNonStreaming' are defined/imported
# in the global scope of annotator.py, as they are used by this function.

def create_request(model: str, prompt: str, system_prompt: str, idx: int, feature: Optional[str] = None) -> Dict[str, Any]:
    """
    Creates a request descriptor for batching using the old, faulty structure
    for OpenAI models.
    """
    
    def clean_text(text: str) -> str:
        """Removes problematic invisible characters without stripping intentional whitespace."""
        text = text.lstrip('\ufeff')
        text = text.replace('\xa0', ' ')
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text

    # Apply cleaning to both prompts
    prompt = clean_text(prompt)
    system_prompt = clean_text(system_prompt)
    
    # Standard batch metadata required for all external API calls
    feature_tag = f"{feature}_" if feature else ""
    unique_suffix = uuid.uuid4().hex[:8]
    batch_metadata: Dict[str, Any] = {
        "custom_id": f"{feature_tag}request_{idx}_{unique_suffix}",
        "method": "POST",
    }
    # --------------------------------------------------------------------------

    # 1) Anthropic (Claude) - (Untouched, uses custom object)
    if model == "claude-3-7":
        # Assuming Request and MessageCreateParamsNonStreaming are available in global scope
        return Request(
            custom_id=f"request_{idx}",
            params=MessageCreateParamsNonStreaming(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                system=[{"type": "text", "text": system_prompt}],
                messages=[{"role": "user", "content": prompt}],
            )
        )
    
    # 2) GPT-4o
    if model == "gpt-4o":
        # The Responses API requires flattened input, but this older logic 
        # still incorrectly uses the chat/completions 'body' structure.
        
        # We need the Annotation class for the response_format key
        AnnotationCls = globals().get("Annotation")
        response_format = AnnotationCls.model_json_schema() if AnnotationCls and hasattr(AnnotationCls, "model_json_schema") else None
        
        body_data = {
            "model": "gpt-4o",
            # NOTE: This uses the flattened input, but puts it inside a 'body'
            # which is incorrect for the Batch API.
            "input": "\n\n".join([p for p in [system_prompt, prompt] if p]),
            "max_output_tokens": 1000,
         #   "temperature": 0.0,
        }
        


        return {
            **batch_metadata, 
            "method":"POST",
            "url": "/v1/responses",
            "body": body_data
        }
        
    # 3) GPT-5-nano and GPT-5-mini and gpt-5.1 and GPT-5.2
    if model in ["gpt-5-nano", "gpt-5-mini", "gpt-5.1", "gpt-5.2"]:
        # We need the Annotation class for the response_format key
        AnnotationCls = globals().get("Annotation")
        response_format = AnnotationCls.model_json_schema() if AnnotationCls and hasattr(AnnotationCls, "model_json_schema") else None
        
        if model in ("gpt-5.1", "gpt-5.2"):
            reasoning_setting = {"effort": "none"}
        else:
            reasoning_setting = {"effort": "minimal"}

        body_data = {
            "model": model,
            "input": "\n\n".join([p for p in [system_prompt, prompt] if p]),
            "max_output_tokens": 1000,
            "reasoning": reasoning_setting,
           # RQ "temperature": 0.0,
        }
        

        return {
            **batch_metadata,
            "method":"POST",
            "url": "/v1/responses",
            "body": body_data,
        }

    # 4) Local Llama models - (Untouched: uses chat/completions body structure)
    if model in ["llama-7b-local", "llama-13b-local"]:
        # This structure is correct for the local models.
        return {
            **batch_metadata,
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 512,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
        }
        
    # Fallback for unsupported models
    raise ValueError(f"Unsupported model: {model}")


def format_dialogue_as_json(df: pd.DataFrame) -> str:
    """
    Formats a DataFrame of utterances into a JSON string (list of dictionaries).
    """
    if df.empty:
        return "[]"
    
    data_list = df[['uttid', 'speaker', 'dialogue']].rename(
        columns={'uttid': 'ID', 'speaker': 'Speaker', 'dialogue': 'Text'}
    ).to_dict('records')
    
    return json.dumps(data_list, indent=None)

@utils.component("process_observations") 
def process_observations(transcript_df: pd.DataFrame,
                         model_list: List[str],
                         feature_dict: Dict,
                         prompt_template: str,
                         system_prompt: str,
                         n_uttr: int,
                         obs_list: List[str] = None,
                         if_context: bool = False,
                         fwd_context_count: int = 0, 
                         bwd_context_count: int = 0,
                         min_len: int = 6,
                         **kwargs) -> Dict[str, pd.DataFrame]:

    # Initial steps remain
    eligible_rows, atn_feature_dfs = mark_ineligible_rows(model_list=model_list,
                                                         feature_dict=feature_dict,
                                                         transcript_df=transcript_df,
                                                         min_len=min_len)

    obs_groups = group_obs(if_context=if_context, obs_list=obs_list, transcript_df=transcript_df)

    model_reqs = {model: [] for model in model_list}

    i = 0
    while i < len(eligible_rows):
        
        # Initialize context variables for safe formatting
        bwd_context = ""
        fwd_context = ""
        
        # -----------------------------------------------------------
        # CONTEXT WINDOW LOGIC (n_uttr=1)
        # -----------------------------------------------------------
        if n_uttr == 1 and (fwd_context_count > 0 or bwd_context_count > 0):
            
            # 1. Identify Target Utterance details
            batch_uttr = eligible_rows.iloc[[i]]
            current_segment_id = batch_uttr.iloc[0]['segment_id_1sd']
            current_uttid = batch_uttr.iloc[0]['uttid']
            
            # 2. Extract the FULL sequential segment
            full_segment = transcript_df[transcript_df['segment_id_1sd'] == current_segment_id].reset_index(drop=True)
            
            # 3. Find the starting index (position)
            start_idx_in_segment = full_segment[full_segment['uttid'] == current_uttid].index[0]
            
            # 4. Extract Context Windows (bwd_context_count for BWD, fwd_context_count for FWD)
            bwd_start = max(0, start_idx_in_segment - bwd_context_count) 
            bwd_end = start_idx_in_segment
            bwd_context_rows = full_segment.iloc[bwd_start:bwd_end]
            
            fwd_start = start_idx_in_segment + 1
            fwd_end = min(len(full_segment), start_idx_in_segment + 1 + fwd_context_count)
            fwd_context_rows = full_segment.iloc[fwd_start:fwd_end]
            
            # 5. Format Dialogue and Contexts as JSON
            combined_dialogue = format_dialogue_as_json(batch_uttr)
            bwd_context = format_dialogue_as_json(bwd_context_rows)
            fwd_context = format_dialogue_as_json(fwd_context_rows)
            
            # Update index for next iteration
            i += 1
        
        # -----------------------------------------------------------
        # ORIGINAL BATCHING LOGIC (n_uttr > 1 or no context windows)
        # -----------------------------------------------------------
        else:
            # Get current segment ID
            current_segment_id = eligible_rows.iloc[i]['segment_id_1sd']

            # Find the end index - either after n_uttr rows or when segment changes
            max_idx = min(i + n_uttr, len(eligible_rows))
            end_idx = i

            for j in range(i, max_idx):
                if eligible_rows.iloc[j]['segment_id_1sd'] != current_segment_id:
                    break
                end_idx = j + 1

            # Extract the batch of utterances
            batch_uttr = eligible_rows.iloc[i:end_idx]

            # Create combined dialogue using the NEW JSON format
            # NOTE: format_dialogue_with_speaker is replaced with format_dialogue_as_json
            combined_dialogue = format_dialogue_as_json(batch_uttr)
            
            # Context variables remain empty strings ("")
            bwd_context = ""
            fwd_context = ""

            # Update index for next iteration
            i = end_idx if end_idx > i else i + n_uttr
        
        # -----------------------------------------------------------
        # COMMON STEP: CREATE PROMPT AND MODEL REQUESTS (RESTORED)
        # -----------------------------------------------------------
        
        # Create prompt by passing all context variables (empty if not used)
        prompt = prompt_template.format(dialogue=combined_dialogue,
                                       bwd_context=bwd_context,
                                       fwd_context=fwd_context)



        for model in model_list:
           request = create_request(model=model, prompt=prompt, system_prompt=system_prompt, idx=i)
           model_reqs[model].append(request)

    return "model_requests", model_reqs


@utils.component("process_requests")
def process_requests(model_requests: Dict,
                     feature: str,
                     model_list: List[str],
                     obs_list: List[str],
                     transcript_source: str,
                     sheet_source: str,
                     if_wait: bool,
                     n_uttr: int,
                     annotation_prompt_path: str,
                     prompt_template: str,
                     system_prompt: str,
                     timestamp: str,
                     save_dir: str,
                     if_test: bool = False
                     ) -> Dict:
    batches = {}
    for model, req_list in model_requests.items():
        req_list = req_list[:100] if if_test else req_list

        if model in ["gpt-4o", "gpt-5-nano", "gpt-5-mini", "gpt-5.1", "gpt-5.2"]:
            batch = batch_openai_annotate(requests=req_list)

        elif model == "claude-3-7":
            batch = batch_anthropic_annotate(requests=req_list)
            
        elif model in ["llama-3b-local", "llama-70b-local"]:
            from llm_annotator.llm import batch_local_llm_annotate
            model_name = "meta-llama/Llama-3.2-3B-Instruct" if model == "llama-3b-local" else "meta-llama/Llama-3.3-70B-Instruct"
            batch = batch_local_llm_annotate(requests=req_list, model_name=model_name)
            
        batches[model] = batch

    store_batch(batches=batches, feature=feature, timestamp=timestamp, save_dir=save_dir)
    store_meta(feature=feature, model_list=model_list, obs_list=obs_list, transcript_source=transcript_source,
               sheet_source=sheet_source, if_wait=if_wait, n_uttr=n_uttr, annotation_prompt_path=annotation_prompt_path,
               timestamp=timestamp, prompt_template=prompt_template, system_prompt=system_prompt, save_dir=save_dir)

    return "batches", batches


@utils.component("fetch_batch")
def fetch_batch(save_dir: str,
                batches: Dict = None,
                timestamp: str = None,
                feature: str = "",
                if_wait: bool = True):
    results = {}
    print("Fetching results...")

    if not batches:
        batches = load_batch_files(timestamp=timestamp, feature=feature, save_dir=save_dir)
    
    if_gpt_finished = False if any(m in batches.keys() for m in ["gpt-4o", "gpt-5-nano","gpt-5-mini", "gpt-5.1", "gpt-5.2"]) else True

    if_claude_finished = False if "claude-3-7" in batches.keys() else True
    if_local_finished = False if any(model in ["llama-3b-local", "llama-70b-local"] for model in batches.keys()) else True

    # Define the function that processes batches and updates results
    def process_batches(if_gpt_finished: bool, if_claude_finished: bool, if_local_finished: bool):
        for model, batch in batches.items():
            if model in ["llama-3b-local", "llama-70b-local"]:
                # Local models don't have batch.id
                pass
            else:
                batch_id = batch.id
            

            if model in ["gpt-4o", "gpt-5-nano", "gpt-5-mini", "gpt-5.1", "gpt-5.2"]:
                client = openai.OpenAI(timeout=180.0)


                response = client.batches.retrieve(batch_id)

                print("OpenAI Response")
                print(response)


                status = response.status

                # Retrieve completed results
                if status == "completed" and response.output_file_id and not if_gpt_finished:
                    result = client.files.content(response.output_file_id).read().decode("utf-8")
                    print(f"{model} has completed batching.")
                    results[model] = result
                    if_gpt_finished = True
                elif status == "completed" and not response.output_file_id and not if_gpt_finished:
                    print(f"WARNING: {model} batch marked as completed, but no output file ID found. (Possible failure or empty results.)")
                    # Treat as finished to break the loop for this model, but with a warning.
                    if_gpt_finished = True
                elif status == "failed" and not if_gpt_finished:
                    print(f"ERROR: {model} batch failed. Reason: {response.errors}")
                    if_gpt_finished = True
                elif status == "in_progress":
                    print(f"{model}: Batch {batch_id} is still in progress.")

            elif model == "claude-3-7":
                client = anthropic.Anthropic()
                
                try:
                    batch_info = client.messages.batches.retrieve(batch_id)
                    status = batch_info.processing_status
                    
                    # Initialize results list if not already done
                    if model not in results:
                        results[model] = []

                    if status == "ended" and not if_claude_finished:
                        print(f"{model} has completed batching.")
                        
                        result_count = 0
                        error_count = 0
                        
                        try:
                            for result in client.messages.batches.results(message_batch_id=batch_id):
                                if result.result.type == "succeeded":
                                    results[model].append(result)
                                    result_count += 1
                                elif result.result.type == "error":
                                    error_count += 1
                        except Exception as e:
                            print(f"Error retrieving Claude-3-7 results: {e}")
                        
                        if result_count > 0:
                            print(f"Retrieved {result_count} Claude-3-7 results")
                        if error_count > 0:
                            print(f"Found {error_count} Claude-3-7 errors")
                        if_claude_finished = True
                        
                    elif status == "expired":
                        print(f"{model}: Batch has expired.")
                        if_claude_finished = True
                    elif status == "canceled":
                        print(f"{model}: Batch was canceled.")
                        if_claude_finished = True
                    elif status == "ended":
                        if_claude_finished = True
                    elif status in ["in_progress", "validating", "finalizing"]:
                        print(f"{model}: Batch is still in progress.")
                        
                except Exception as e:
                    print(f"Error retrieving Claude-3-7 batch: {e}")
                    if_claude_finished = True

            elif model in ["llama-3b-local", "llama-70b-local"]:
                # Local models process immediately, so results are already available
                if not if_local_finished:
                    print(f"{model}: Local batch completed.")
                    # For local models, batch is already the results list
                    results[model] = batch
                    if_local_finished = True

        return if_gpt_finished, if_claude_finished, if_local_finished

    if if_wait:
        # Use the loop to keep checking until all batches are done
        while True:
            if_gpt_finished, if_claude_finished, if_local_finished = process_batches(if_gpt_finished, if_claude_finished, if_local_finished)

            if if_gpt_finished and if_claude_finished and if_local_finished:
                print("All annotation tasks are finished.")
                break  # Exit loop if all batches are done
            
            print("Waiting 60s for batch completion...")
            time.sleep(60)
    else:
        # Execute the batch processing just once without waiting
        process_batches(if_gpt_finished, if_claude_finished, if_local_finished)

    return "batch_results", results
