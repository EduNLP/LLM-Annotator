import pandas as pd
import os
import json

import llm_annotator.utils as utils
from typing import Dict

import pandas as pd
import os
import json
import re # <-- NECESSARY IMPORT

import llm_annotator.utils as utils
from typing import Dict, Any, Union


@utils.component("save_results")
def save_results(batch_results: Dict, transcript_df: pd.DataFrame, feature: str, timestamp: str = None, save_dir: str = None):
    """
    Saves batch annotation results to CSV/DataFrame.
    Handles OpenAI Batch API (nested), Anthropic Batch (object), and Local LLMs.
    """
    
    # --- HELPER: Robust JSON Extraction ---
    def _extract_json_from_text_block(text: Union[str, Dict]) -> Union[Dict, None]:
        if isinstance(text, dict): return text
        if not text: return None
        
        # If input is not string/dict (e.g., Pydantic object), try to stringify
        if not isinstance(text, str):
            try: text = str(text)
            except: return None

        t = text.strip()
        
        # 1. Try regex for markdown blocks (Most reliable)
        m = re.search(r"```json\s*(.*?)\s*```", t, flags=re.DOTALL)
        if m: 
            try: return json.loads(m.group(1).strip())
            except: pass
            
        # 2. Try raw JSON cleanup
        t_cleaned = t.removeprefix("```json").removesuffix("```").strip()
        start, end = t_cleaned.find("{"), t_cleaned.rfind("}")
        if start != -1 and end != -1:
            try: return json.loads(t_cleaned[start:end+1])
            except: pass
            
        return None

    # --- SETUP ---
    if timestamp is None:
        if not save_dir:
            if not os.path.exists(f"result/{feature}"):
                raise FileNotFoundError("The result folder doesn't exist.")
            batch_dir = f"result/{feature}"
        else:
            batch_dir = save_dir + f"/result/{feature}"
        timestamp = os.path.join(batch_dir, utils.find_latest_dir(batch_dir))
        batch_dir = os.path.join(batch_dir, timestamp)
    else:
        batch_dir = utils.create_batch_dir(save_dir=save_dir, feature=feature, timestamp=timestamp)
        
    transcript_df = transcript_df.copy()
    
    # CRITICAL: Cast IDs to string to ensure JSON keys match DataFrame index
    transcript_df['uttid'] = transcript_df['uttid'].astype(str)

    # --- PROCESSING LOOP ---
    for model, batch_content in batch_results.items():
        print(f"Processing {model} results...")
        annotations_processed = 0
        
        # =====================================================
        # 1. OPENAI MODELS (gpt-4o, gpt-5-nano, gpt-5-mini, gpt-5.1)
        # =====================================================
        if model in ("gpt-4o", "gpt-5-nano", "gpt-5-mini", "gpt-5.1"):
            try:
                if not batch_content: continue
                
                # OpenAI Batch API returns a JSONL string
                raw_lines = batch_content.strip().splitlines()
                print(f"{model}: Reading {len(raw_lines)} lines...")

                for line_idx, raw_line in enumerate(raw_lines):
                    if not raw_line.strip(): continue
                    try: obj = json.loads(raw_line)
                    except: continue

                    # Drill down: response -> body
                    resp_body = obj.get("response", {}).get("body", {})
                    text_payload = None
                    
                    # Extraction Strategy: output (list) -> message -> content -> text
                    outputs = resp_body.get("output", [])
                    if isinstance(outputs, list):
                        for item in outputs:
                            if item.get("type") == "message":
                                for part in item.get("content", []):
                                    if part.get("type") == "output_text":
                                        text_payload = part.get("text")
                                        break
                    
                    # Fallback Extraction
                    if not text_payload:
                        if "parsed" in resp_body: text_payload = resp_body["parsed"]
                        elif "text" in resp_body: text_payload = resp_body["text"]

                    if not text_payload: continue

                    # Parse JSON content
                    parsed_content = _extract_json_from_text_block(text_payload)
                    if not parsed_content: continue

                    # Extract LogProbs (Best Effort)
                    result_log_probs = []
                    try:
                        for item in outputs:
                            if item.get("type") == "message":
                                for part in item.get("content", []):
                                    for prob in part.get("logprobs", []):
                                        if prob.get('token') in ("0", "1"):
                                            result_log_probs.append(round(prob.get('logprob', 0.0), 3))
                    except: pass

                    # Update DataFrame
                    index = 0
                    for utt_id, value in parsed_content.items():
                        match = transcript_df['uttid'] == str(utt_id)
                        if match.any():
                            transcript_df.loc[match, feature] = value
                            if index < len(result_log_probs):
                                transcript_df.loc[match, f"{model}_logprob"] = result_log_probs[index]
                            annotations_processed += 1
                        index += 1

            except Exception as e:
                print(f"Error processing {model}: {e}")

        # =====================================================
        # 2. ANTHROPIC (claude-3-7)
        # =====================================================
        elif model == "claude-3-7":
            try:
                if not batch_content: continue
                # batch_content is a list of BatchResult objects from fetch_batch
                print(f"{model}: Reading {len(batch_content)} results...")
                
                for result_obj in batch_content:
                    try:
                        # Depending on SDK version, access via attribute or dict
                        # Standard Anthropic Batch Result access:
                        if hasattr(result_obj, 'result') and hasattr(result_obj.result, 'message'):
                             # Accessing object attributes
                            content_block = result_obj.result.message.content[0]
                            text_payload = content_block.text
                        else:
                            # Fallback if it's a dict/json
                            continue 

                        parsed_content = _extract_json_from_text_block(text_payload)
                        if not parsed_content: continue

                        for utt_id, value in parsed_content.items():
                            match = transcript_df['uttid'] == str(utt_id)
                            if match.any():
                                transcript_df.loc[match, feature] = value
                                annotations_processed += 1

                    except Exception as inner_e:
                        # print(f"Error parsing specific Claude result: {inner_e}")
                        continue
            except Exception as e:
                print(f"Error processing {model}: {e}")

        # =====================================================
        # 3. LOCAL MODELS (llama-*-local)
        # =====================================================
        elif "local" in model:
            try:
                if not batch_content: continue
                # batch_content is typically a list of response strings or dicts
                print(f"{model}: Reading {len(batch_content)} results...")

                for item in batch_content:
                    text_payload = item
                    
                    # Handle if item is a dict (e.g., {'choices': ...}) or just string
                    if isinstance(item, dict):
                        # Try standard OpenAI-compatible format often used by local servers
                        if 'choices' in item:
                            text_payload = item['choices'][0]['message']['content']
                        else:
                            text_payload = str(item)

                    parsed_content = _extract_json_from_text_block(text_payload)
                    if not parsed_content: continue

                    for utt_id, value in parsed_content.items():
                        match = transcript_df['uttid'] == str(utt_id)
                        if match.any():
                            transcript_df.loc[match, feature] = value
                            annotations_processed += 1
            except Exception as e:
                print(f"Error processing {model}: {e}")
        
        print(f"-> Finished {model}: {annotations_processed} annotations applied.")

    # --- SAVE FINAL OUTPUT ---
    if hasattr(utils, 'save_final_df'):
        utils.save_final_df(transcript_df, batch_dir, feature)
    else:
        # Fallback if utility missing
        out_path = os.path.join(batch_dir, "atn_df.csv")
        transcript_df.to_csv(out_path, index=False)
        print(f"Results saved to: {out_path}")
        
    return ('transcript_df', transcript_df)