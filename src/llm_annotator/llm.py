import instructor
import os
import json
import re

import anthropic

from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Optional
import uuid # Add this to top of llm.py
from pydantic import BaseModel
from openai import OpenAI
from anthropic import Anthropic
from anthropic.types.messages.batch_create_params import Request
from datetime import datetime, timezone


from llm_annotator.utils import create_batch_dir

# Local LLM imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

_ = load_dotenv(find_dotenv())


class Annotation(BaseModel):
    feature_name: str
    annotation: int

def openai_annotate(prompt: str, system_prompt: str = None):
    """
    Single-request call using Responses API (flattened text input).
    Uses Pydantic Annotation.model_json_schema() as response_format (keeps structured parsing).
    Returns the parsed structured output (dict) when present, otherwise None.
    """
    client = OpenAI()

    # Flattened input: system prompt then user prompt, separated by double newline
    flattened_input = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt


    #Debug

    # Use responses.create for Responses API
    completion = client.responses.create(
        model="gpt-5-nano",
        input=flattened_input,
        response_format=Annotation.model_json_schema(),
        max_output_tokens=1000
    )

    # Extract structured parsed output, handling multiple SDK shapes:
    parsed = None
    try:
        # Some SDKs expose a top-level 'output_parsed' or similar attribute
        if hasattr(completion, "output_parsed") and completion.output_parsed:
            parsed = completion.output_parsed
            return parsed

        outputs = getattr(completion, "output", None) or (completion.get("output") if isinstance(completion, dict) else None)
        if outputs and len(outputs) > 0:
            first = outputs[0]
            # If the first output item is a dict, try dict lookup
            if isinstance(first, dict):
                parsed = first.get("parsed")
                # Some SDKs nest parsed deeper (e.g., first.get('content', [{}])[0].get('parsed'))
                if parsed is None:
                    # Common fallback structure: {'content': [{'type': 'output_text', 'text': '...'}], 'parsed': {...}}
                    # Try nested possibilities conservatively
                    content = first.get("content") or first.get("outputs") or first.get("data")
                    if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
                        parsed = content[0].get("parsed") or content[0].get("data", {}).get("parsed")
            else:
                # object-like SDK item; prefer attribute access
                parsed = getattr(first, "parsed", None)
    except Exception:
        parsed = None

    return parsed


def anthropic_annotate(prompt: str, system_prompt: str):
    client = instructor.from_anthropic(Anthropic())
    completion = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        messages=[
            {"role": "system",
             "content": system_prompt},
            {"role": "user", "content": f"{prompt}"}
        ],
        max_tokens=300,
        response_model=Annotation.model_json_schema(),
    )

    return completion


def batch_openai_annotate(requests: List[Dict]):
    """
    Writes a JSONL file for batch processing using the Responses API, then creates a batch.
    Each line is of the form:
      {"custom_id":..., "body": {"model": "...", "input": "...", "response_format": <schema>, ...}}
    and the batch is created with endpoint "/v1/responses".
    """
    os.makedirs("temp", exist_ok=True)

    ts = datetime.now().astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%f")

    random_id = uuid.uuid4().hex
    file_path = os.path.join("temp", f"batch_input_{ts}_{random_id}.jsonl")


    # Build JSONL with Responses API body structure
    with open(file_path, "w", encoding="utf-8") as f:
        for item in requests:
            # item is expected to be what create_request() returned.
            # Ensure `body` contains flattened 'input' field instead of 'messages'
            body = item.get("body", {})
            # If the request still contains `messages`, flatten them into a single text field:
            if "messages" in body and not "input" in body:
                parts = []
                for m in body.get("messages", []):
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    # include role label as markup to preserve role if desired
                    parts.append(f"{role.upper()}: {content}")
                body["input"] = "\n\n".join(parts)
                body.pop("messages", None)

            # Ensure response_format is present; prefer schema from Annotation
           # RQ if "response_format" not in body:
                # Keep using model_json_schema as-is (SDK should accept it)
           #     body["response_format"] = Annotation.model_json_schema()

            line = {"custom_id": item.get("custom_id"), "body": body, "method": "POST", "url": '/v1/responses'} # Method


                    # --- DEBUG START: CHECK FINAL LINE STRUCTURE ---
            # Print the custom ID to see which request is being processed
            print(f"\n[DEBUG] Processing Custom ID: {item.get('custom_id')}")
            
            # Check the final 'input' field length and boundary
            input_content = body.get("input", "")
            print(f"[DEBUG] Input Content Length: {len(input_content)}")
            # Print the start and end of the prompt to catch early/late errors
            print(f"[DEBUG] Input Content Start: {input_content[:150].replace('\n', ' ')}...")
            print(f"[DEBUG] Input Content End: ...{input_content[-150:].replace('\n', ' ')}")
            
            # Print the entire JSON line, but ONLY the first 300 characters
            # The output of json.dumps(line) should be valid JSON.
            try:
                full_json_line = json.dumps(line, ensure_ascii=False)
                print(f"[DEBUG] Final JSON Line Sample: {full_json_line[:300]}...")


            except Exception as e:
                print(f"[CRITICAL DEBUG] Failed to serialize final JSON line: {e}")
            #Debug end


            json.dump(line, f, ensure_ascii=False)


            # line_dictionary_file = json.loads(line)
            # print(f"[DEBUG] Final JSON Line Sample: {line_dictionary_file.keys()}")


            f.write("\n")
    # --- END OF CORRECTION ---

    client = OpenAI()

    # Upload the batch file
    with open(file_path, "rb") as fp:
        batch_input_file = client.files.create(file=fp, purpose="batch")



    batch_input_file_id = batch_input_file.id

    print("BATCH INPUT DATA")
    print(batch_input_file_id)
    print(batch_input_file)

    # Create a batch that targets the Responses endpoint
    batch_file = client.batches.create(
        input_file_id=batch_input_file_id,

        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"description": "Annotation job (responses API)."}
    )

    return batch_file


def batch_anthropic_annotate(requests: List[Request]):
    client = anthropic.Anthropic()

    message_batch = client.messages.batches.create(
        requests=requests
    )
    return message_batch


# Global cache for local models to avoid reloading
_local_model_cache = {}


def _get_local_model(model_name: str):
    """Get or load a local model with caching."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for local models. Install with: pip install transformers torch accelerate")
    
    if model_name not in _local_model_cache:
        print(f"Loading local model: {model_name}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        _local_model_cache[model_name] = {
            "tokenizer": tokenizer,
            "model": model
        }
        print(f"Model {model_name} loaded successfully")
    
    return _local_model_cache[model_name]


def _format_llama_prompt(prompt: str, system_prompt: str = None) -> str:
    """Format prompt for Llama models using the chat template."""
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [
            {"role": "user", "content": prompt}
        ]
    
    return messages


def _extract_json_from_response(response: str) -> Optional[Dict]:
    """Extract JSON from model response."""
    # Try to find JSON in the response
    json_pattern = r'\{[^{}]*"feature_name"[^{}]*"annotation"[^{}]*\}'
    match = re.search(json_pattern, response)
    
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to extract values manually
    feature_match = re.search(r'"feature_name":\s*"([^"]*)"', response)
    annotation_match = re.search(r'"annotation":\s*(\d+)', response)
    
    if feature_match and annotation_match:
        return {
            "feature_name": feature_match.group(1),
            "annotation": int(annotation_match.group(1))
        }
    
    return None


def local_llm_annotate(prompt: str, model_name: str, system_prompt: str = None) -> Annotation:
    """Annotate using a local LLM model."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for local models")
    
    # Get the cached model
    model_cache = _get_local_model(model_name)
    tokenizer = model_cache["tokenizer"]
    model = model_cache["model"]
    
    # Format the prompt
    messages = _format_llama_prompt(prompt, system_prompt)
    
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Move to device if using GPU
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Extract JSON from response
    json_data = _extract_json_from_response(response)
    
    if json_data:
        return Annotation(**json_data)
    else:
        # Fallback annotation if parsing fails
        return Annotation(feature_name="unknown", annotation=0)


def batch_local_llm_annotate(requests: List[Dict], model_name: str) -> List[Dict]:
    """Batch annotate using a local LLM model."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for local models")
    
    results = []
    
    for i, request in enumerate(requests):
        try:
            # Extract prompt and system prompt from request
            messages = request.get("body", {}).get("messages", [])
            system_prompt = None
            user_prompt = None
            
            for message in messages:
                if message.get("role") == "system":
                    system_prompt = message.get("content", "")
                elif message.get("role") == "user":
                    user_prompt = message.get("content", "")
            
            if user_prompt:
                annotation = local_llm_annotate(user_prompt, model_name, system_prompt)
                result = {
                    "custom_id": request.get("custom_id", f"request_{i}"),
                    "response": {
                        "body": {
                            "choices": [{
                                "message": {
                                    "content": json.dumps({
                                        "feature_name": annotation.feature_name,
                                        "annotation": annotation.annotation
                                    })
                                }
                            }]
                        }
                    }
                }
                results.append(result)
            
        except Exception as e:
            print(f"Error processing request {i}: {str(e)}")
            # Add error result
            result = {
                "custom_id": request.get("custom_id", f"request_{i}"),
                "response": {
                    "body": {
                        "choices": [{
                            "message": {
                                "content": json.dumps({
                                    "feature_name": "error",
                                    "annotation": 0
                                })
                            }
                        }]
                    }
                }
            }
            results.append(result)
    
    return results


def store_meta(model_list: List[str],
               feature: str,
               obs_list: List[str],
               transcript_source: str,
               sheet_source: str,
               if_wait: bool,
               n_uttr: int,
               annotation_prompt_path: str,
               timestamp: str,
               prompt_template: str,
               system_prompt: str,
               save_dir: str):
    timestamp_dir = create_batch_dir(save_dir=save_dir, feature=feature, timestamp=timestamp)

    # Create metadata dictionary
    metadata = {
        "model_list": model_list,
        "feature": feature,
        "obs_list": obs_list,
        "transcript_source": transcript_source,
        "sheet_source": sheet_source,
        "save_dir": timestamp_dir,
        "if_wait": if_wait,
        "n_uttr": n_uttr,
        "annotation_prompt_path": annotation_prompt_path,
        "timestamp": timestamp,
        "prompt": f"{prompt_template}",
        "system prompt": system_prompt
    }

    # Save metadata as JSON
    meta_file_path = os.path.join(timestamp_dir, "metadata.json")
    with open(meta_file_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {meta_file_path}")


def store_batch(batches: Dict,
                feature: str,
                save_dir: str,
                timestamp: str):
    batch_dir = create_batch_dir(save_dir=save_dir, feature=feature, timestamp=timestamp)
    for model, batch_file in batches.items():
        batch_filename = f"{model}.json"
        file_path = os.path.join(batch_dir, batch_filename)

        if model in ("gpt-4o", "gpt-5-nano", "gpt-5-mini", "gpt-5.1"):

            batch_metadata = {
                "id": batch_file.id,
                "object": batch_file.object,
                "endpoint": batch_file.endpoint,
                "errors": batch_file.errors,
                "input_file_id": batch_file.input_file_id,
                "completion_window": batch_file.completion_window,
                "status": batch_file.status,
                "output_file_id": batch_file.output_file_id,
                "error_file_id": batch_file.error_file_id,
                "created_at": batch_file.created_at,
                "in_progress_at": batch_file.in_progress_at,
                "expires_at": batch_file.expires_at,
                "completed_at": batch_file.completed_at,
                "failed_at": batch_file.failed_at,
                "expired_at": batch_file.expired_at,
                "stored_at": datetime.now().isoformat()
            }
        elif model == "claude-3-7":
            # Convert MessageBatchRequestCounts to a dictionary
            batch_metadata = {
                "id": batch_file.id,
                "type": batch_file.type,
                "processing_status": batch_file.processing_status,
                "request_counts": {
                    "processing": batch_file.request_counts.processing,
                    "succeeded": batch_file.request_counts.succeeded,
                    "errored": batch_file.request_counts.errored,
                    "canceled": batch_file.request_counts.canceled,
                    "expired": batch_file.request_counts.expired
                },
                "ended_at": batch_file.ended_at,
                "created_at": batch_file.created_at.isoformat(),
                "expires_at": batch_file.expires_at.isoformat(),
                "cancel_initiated_at": batch_file.cancel_initiated_at,
                "results_url": batch_file.results_url,
                "stored_at": datetime.now().isoformat()
            }
        elif model in ["llama-3b-local", "llama-70b-local"]:
            # For local models, batch_file is a list of results
            batch_metadata = {
                "model": model,
                "type": "local_batch",
                "processing_status": "completed",
                "request_counts": {
                    "processing": 0,
                    "succeeded": len(batch_file),
                    "errored": 0,
                    "canceled": 0,
                    "expired": 0
                },
                "total_requests": len(batch_file),
                "created_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
                "stored_at": datetime.now().isoformat()
            }
        
        with open(file_path, "w") as f:
            json.dump(batch_metadata, f, indent=2)
    return "batch_dir", batch_dir
