"""Colab widget UI for experiment configuration.

Usage:
    from llm_annotator.ui import show_config_ui
    ui = show_config_ui()
    # User fills in widgets, then:
    config = ui.build_config()
    run_pipeline(config, ...)
"""

import ipywidgets as widgets
from IPython.display import display, HTML
from llm_annotator.config import ExperimentConfig


# Available models grouped by provider
MODEL_OPTIONS = [
    "gpt-4o", "gpt-5-nano", "gpt-5-mini", "gpt-5.1", "gpt-5.2",
    "claude-3-5",
    "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro",
    "gemini-3-flash-preview", "gemini-3-pro-preview",
]

FEATURE_OPTIONS = [
    "Offtask", "Recording", "Directions", "Coordinate",
    "Competent", "Language", "Understanding", "Tool",
    "claim", "reason", "agree", "disagree", "compare",
    "addon", "question", "revoice", "monitor", "nextstep",
    "redirect", "compliment", "apology",
]

STYLE = {"description_width": "180px"}
LAYOUT = widgets.Layout(width="600px")
NARROW = widgets.Layout(width="300px")


class ConfigUI:
    def __init__(self, drive_base=""):
        self.drive_base = drive_base

        # ── Core ──
        self.models = widgets.SelectMultiple(
            options=MODEL_OPTIONS, value=["gpt-5-mini"],
            description="Models", style=STYLE, layout=LAYOUT,
            rows=6,
        )
        self.features = widgets.SelectMultiple(
            options=FEATURE_OPTIONS, value=["Directions", "Coordinate"],
            description="Features", style=STYLE, layout=LAYOUT,
            rows=8,
        )
        self.obs_list = widgets.Text(
            value="17, 25", description="Obs IDs (or 'all')",
            style=STYLE, layout=LAYOUT,
            placeholder="17, 25, 195  or  all",
        )

        # ── Prompt ──
        self.n_uttr = widgets.IntSlider(
            value=5, min=1, max=20, description="Utterances/request",
            style=STYLE, layout=NARROW,
        )
        self.bwd = widgets.IntSlider(
            value=2, min=0, max=10, description="Backward context",
            style=STYLE, layout=NARROW,
        )
        self.fwd = widgets.IntSlider(
            value=0, min=0, max=10, description="Forward context",
            style=STYLE, layout=NARROW,
        )

        # ── Run control ──
        self.test_mode = widgets.Checkbox(
            value=True, description="Test mode (~20 rows)",
            style=STYLE,
        )
        self.wait = widgets.Checkbox(
            value=False, description="Wait for batch to complete",
            style=STYLE,
        )
        self.use_video = widgets.Checkbox(
            value=False, description="Include video (Gemini only)",
            style=STYLE,
        )
        self.verbose = widgets.Checkbox(
            value=True, description="Show detailed logs",
            style=STYLE,
        )

        # ── Feature rules (collapsible) ──
        self.filter_if_text = widgets.Textarea(
            value="", description="filter_if",
            placeholder="Directions: offtask\nCoordinate: offtask",
            style=STYLE, layout=widgets.Layout(width="600px", height="60px"),
        )
        self.linked_with_text = widgets.Textarea(
            value="", description="linked_with",
            placeholder="Directions: Coordinate\nCoordinate: Directions",
            style=STYLE, layout=widgets.Layout(width="600px", height="60px"),
        )
        self.subcode_of_text = widgets.Textarea(
            value="", description="subcode_of",
            placeholder="Recording: offtask",
            style=STYLE, layout=widgets.Layout(width="600px", height="60px"),
        )
        self.extra_context_type_text = widgets.Textarea(
            value="", description="extra_context_type",
            placeholder="Directions: activity_instructions",
            style=STYLE, layout=widgets.Layout(width="600px", height="60px"),
        )
        self.extra_context_text = widgets.Textarea(
            value="", description="extra_context",
            placeholder="activity_instructions: <paste text here>",
            style=STYLE, layout=widgets.Layout(width="600px", height="80px"),
        )

        # ── Resume ──
        self.resume_text = widgets.Textarea(
            value="", description="Resume batch IDs",
            placeholder="gpt-5-mini: batch_abc123",
            style=STYLE, layout=widgets.Layout(width="600px", height="50px"),
        )

        # ── Data sources (usually don't change) ──
        self.transcript_source = widgets.Text(
            value=drive_base + "/MOL Conceptual Pipeline Outputs/mol_formatted_data.csv" if drive_base else "",
            description="Transcript source", style=STYLE, layout=LAYOUT,
        )
        self.sheet_source = widgets.Text(
            value="1miMC8M_UkfY_3XhglTAdt9sC9H8D6d9slzC7_LgcLOc",
            description="Feature sheet ID", style=STYLE, layout=LAYOUT,
        )
        self.save_dir = widgets.Text(
            value=drive_base + "/result/" if drive_base else "result/",
            description="Save directory", style=STYLE, layout=LAYOUT,
        )

    def display(self):
        """Show the config UI."""
        display(HTML("<h3>Experiment Config</h3>"))

        display(HTML("<b>What to run</b>"))
        display(self.models, self.features, self.obs_list)

        display(HTML("<b>Prompt settings</b>"))
        display(self.n_uttr, self.bwd, self.fwd)

        display(HTML("<b>Run options</b>"))
        display(self.test_mode, self.wait, self.use_video, self.verbose)

        rules_accordion = widgets.Accordion(children=[
            widgets.VBox([
                HTML("<p style='color:gray'>Override sheet defaults. Format: <code>FeatureName: value</code>, one per line.</p>"),
                self.filter_if_text,
                self.linked_with_text,
                self.subcode_of_text,
                self.extra_context_type_text,
                self.extra_context_text,
            ]),
        ])
        rules_accordion.set_title(0, "Feature rule overrides (optional)")
        rules_accordion.selected_index = None  # collapsed by default
        display(rules_accordion)

        advanced = widgets.Accordion(children=[
            widgets.VBox([
                self.resume_text,
                self.transcript_source,
                self.sheet_source,
                self.save_dir,
            ]),
        ])
        advanced.set_title(0, "Advanced / Resume (optional)")
        advanced.selected_index = None
        display(advanced)

    def build_config(self) -> ExperimentConfig:
        """Build ExperimentConfig from current widget values."""
        obs = self.obs_list.value.strip()
        if obs.lower() == "all":
            obs_list = "all"
        else:
            obs_list = [o.strip() for o in obs.split(",") if o.strip()]

        return ExperimentConfig(
            model_list=list(self.models.value),
            feature_list=list(self.features.value),
            obs_list=obs_list,
            transcript_source=self.transcript_source.value,
            sheet_source=self.sheet_source.value,
            bwd_context_count=self.bwd.value,
            fwd_context_count=self.fwd.value,
            n_uttr=self.n_uttr.value,
            if_test=self.test_mode.value,
            if_wait=self.wait.value,
            use_video=self.use_video.value,
            save_dir=self.save_dir.value,
            resume_batch_ids=_parse_dict(self.resume_text.value),
            filter_if=_parse_list_dict(self.filter_if_text.value),
            linked_with=_parse_list_dict(self.linked_with_text.value),
            subcode_of=_parse_dict(self.subcode_of_text.value),
            extra_context_type=_parse_dict(self.extra_context_type_text.value),
            extra_context=_parse_dict(self.extra_context_text.value),
        )


def _parse_dict(text: str) -> dict:
    """Parse 'key: value' lines into a dict."""
    result = {}
    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        result[key.strip()] = val.strip()
    return result


def _parse_list_dict(text: str) -> dict:
    """Parse 'key: val1, val2' lines into {key: [val1, val2]}."""
    result = {}
    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        result[key.strip()] = [v.strip() for v in val.split(",") if v.strip()]
    return result


def show_config_ui(drive_base="") -> ConfigUI:
    """Create and display the config UI. Returns the UI object to call build_config() on."""
    ui = ConfigUI(drive_base=drive_base)
    ui.display()
    return ui
