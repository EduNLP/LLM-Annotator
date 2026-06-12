from llm_annotator.llm import batch_anthropic_annotate, batch_local_llm_annotate, batch_openai_annotate
from llm_annotator.constants import GEMINI_MODEL_IDS

from enum import Enum
from typing import Optional
from dataclasses import dataclass, field

class ModelType(Enum):
    GPT4O = "gpt-4o"
    O1 = "o1-preview"
    GPT4_TURBO = "gpt-4-turbo"
    GPT5_NANO = "gpt-5-nano"
    GPT5_MINI = "gpt-5-mini" 
    GPT5_1 = "gpt-5.1"  # Added
    GPT5_2 = "gpt-5.2"
    CLAUDE = "claude-3-5-sonnet-20241022"
    GEMINI = "gemini"
    MISTRAL = "mistral--large-latest"
    DEEPSEEK = "deepseek-chat"
    LLAMA3B = "meta-llama/Llama-3.2-3B-Instruct"
    LLAMA70B = "meta-llama/Llama-3.3-70B-Instruct"
    LLAMA7B_LOCAL = "meta-llama/Llama-3.2-3B-Instruct"
    LLAMA13B_LOCAL = "meta-llama/Llama-3.3-70B-Instruct"


@dataclass
class ModelConfig:
    type: ModelType
    temperature: float = 0
    max_tokens: Optional[int] = None


model_configs = {
        "gpt-4o": ModelConfig(ModelType.GPT4O),
        "gpt-o1": ModelConfig(ModelType.O1),
        "gpt-5-nano": ModelConfig(ModelType.GPT5_NANO),
        "gpt-5-mini": ModelConfig(ModelType.GPT5_MINI),  # Added
        "gpt-5.1": ModelConfig(ModelType.GPT5_1),  # Added
        "gpt-5.2": ModelConfig(ModelType.GPT5_2),
        "claude-3-5": ModelConfig(ModelType.CLAUDE),
        **{mid: ModelConfig(ModelType.GEMINI) for mid in GEMINI_MODEL_IDS},
        "mistral": ModelConfig(ModelType.MISTRAL),
        "deepseek-v3": ModelConfig(ModelType.DEEPSEEK),
        "llama-3.2-3b": ModelConfig(ModelType.LLAMA3B),
        "llama-3.3-70b": ModelConfig(ModelType.LLAMA70B),
        "llama-3b-local": ModelConfig(ModelType.LLAMA7B_LOCAL, max_tokens=512),
        "llama-70b-local": ModelConfig(ModelType.LLAMA13B_LOCAL, max_tokens=512)
}

@dataclass
class ExperimentConfig:
    """Top-level config for a single annotation experiment run."""

    # Models to annotate with
    model_list: list = field(default_factory=list)

    # Features to annotate (run together in one prompt when linked by sheet rules)
    feature_list: list = field(default_factory=list)

    # Data sources
    transcript_source: str = ""
    sheet_source: str = ""
    obs_list: object = "all"  # list[str] or "all"

    # Prompt context window
    bwd_context_count: int = 2
    fwd_context_count: int = 0

    # ── Per-feature rule overrides ──────────────────────────────────────
    # These override whatever the sheet says. Omitted features use sheet defaults.
    #
    # filter_if: which existing annotation columns to use for row exclusion.
    #   {"Directions": ["offtask"], "Coordinate": ["offtask"], "Mathcompetent": []}
    #   Empty list = no filtering even if the sheet says otherwise.
    filter_if: dict = field(default_factory=dict)

    # linked_with: run features together in one prompt (shared context).
    #   {"Directions": ["Coordinate"], "Coordinate": ["Directions"]}
    linked_with: dict = field(default_factory=dict)

    # subcode_of: treat feature as a subcode of another.
    #   {"Recording": "offtask"}
    subcode_of: dict = field(default_factory=dict)

    # extra_context_type: key into extra_context dict for prompt injection.
    #   {"Directions": "activity_instructions"}
    extra_context_type: dict = field(default_factory=dict)

    # Per-feature extra context injected into the prompt.
    # Keys must match values in extra_context_type above.
    # e.g. {"activity_instructions": "<paste activity text here>"}
    extra_context: dict = field(default_factory=dict)

    # Multimodal video (whole-segment, not pre-cut clips)
    use_video: bool = False
    # Google Sheet ID with observation metadata:
    #   - "Index" column: obsid → "25-0195" style value
    #   - "Video Link" column: Drive folder hyperlink containing OBS-25-XXXX_Video* files
    #   - "Segment Timestamps" column: free-text block mapping segment letters to time ranges
    obs_sheet_source: Optional[str] = None

    # Run control
    n_uttr: int = 1
    if_test: bool = False
    if_wait: bool = False
    save_dir: str = "result/"

    # Resume from existing batch IDs (skip submission, go straight to fetch)
    # e.g. {"gpt-4o": "batch_abc123"}
    resume_batch_ids: dict = field(default_factory=dict)

    # When True, skip annotation entirely — just evaluate a previous run's
    # output against the validation set.
    evaluate_only: bool = False

    # Prompt paths (use package defaults if empty)
    system_prompt_path: str = "data/prompts/system_prompt.txt"
    prompt_path: str = ""
    annotation_prompt_path: str = ""
    mode: str = ""

    def __post_init__(self):
        if self.use_video and not self.obs_sheet_source:
            raise ValueError("obs_sheet_source must be set when use_video=True")

    def get_feature_rules(self, feature: str, sheet_meta: dict = None) -> dict:
        """Return merged feature rules: config overrides take priority over sheet defaults.

        Args:
            feature: Feature name (e.g. "Directions").
            sheet_meta: Feature dict from generate_features() for this feature.
                If None, only config overrides are used.

        Returns:
            Dict with keys: filter_if (list), linked_with (list),
            subcode_of (str), extra_context_type (str).
        """
        defaults = {
            "filter_if": sheet_meta.get("filter_if", []) if sheet_meta else [],
            "linked_with": sheet_meta.get("linked_with", []) if sheet_meta else [],
            "subcode_of": sheet_meta.get("subcode_of", "") if sheet_meta else "",
            "extra_context_type": sheet_meta.get("extra_context_type", "") if sheet_meta else "",
        }

        if feature in self.filter_if:
            defaults["filter_if"] = self.filter_if[feature]
        if feature in self.linked_with:
            defaults["linked_with"] = self.linked_with[feature]
        if feature in self.subcode_of:
            defaults["subcode_of"] = self.subcode_of[feature]
        if feature in self.extra_context_type:
            defaults["extra_context_type"] = self.extra_context_type[feature]

        return defaults


annotation_configs = {
    "claude-3-5": batch_anthropic_annotate,
    "gpt-5-nano": batch_openai_annotate, 
    "gpt-5-mini": batch_openai_annotate,  # Added
    "gpt-5.1": batch_openai_annotate,  # Added
    "gpt-5.2": batch_openai_annotate,
    "llama-3b-local": lambda requests: batch_local_llm_annotate(requests, "meta-llama/Llama-3.2-3B-Instruct"),
    "llama-70b-local": lambda requests: batch_local_llm_annotate(requests, "meta-llama/Llama-3.3-70B-Instruct")
}
