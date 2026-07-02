"""Colab widget UI for experiment configuration.

Auto-populates features from the MOL Roles Features sheet and validation
files from Drive. New sheet tabs (e.g. Multimodal) appear automatically.

Usage:
    from llm_annotator.ui import show_config_ui
    ui = show_config_ui(gc=gc, drive_base=DRIVE_BASE, sheets=SHEETS)
    config = ui.build_config()
"""

import os
import ipywidgets as widgets
from IPython.display import display
from llm_annotator.config import ExperimentConfig


MODEL_OPTIONS = [
    "gpt-4o", "gpt-5-nano", "gpt-5-mini", "gpt-5.1", "gpt-5.2",
    "claude-3-5",
    "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro",
    "gemini-3-flash-preview", "gemini-3-pro-preview",
]

STYLE = {"description_width": "180px"}
LAYOUT = widgets.Layout(width="600px")
NARROW = widgets.Layout(width="300px")


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic data loaders
# ═══════════════════════════════════════════════════════════════════════════

def _load_feature_options(gc, features_sheet_id: str) -> list[tuple[str, str]]:
    """Read feature codes from all tabs. Returns [(code, tab_name), ...]."""
    if not gc or not features_sheet_id:
        return []
    try:
        from llm_annotator.utils import read_sheet_as_dataframes
        tabs = read_sheet_as_dataframes(gc, features_sheet_id)
        FEATURE_TABS = {"conceptual", "discursive"}
        results = []
        for tab_name, df in tabs.items():
            if tab_name.strip().lower() not in FEATURE_TABS:
                continue
            if df.empty:
                continue
            code_col = next((c for c in df.columns if c.strip().lower() == "code"), None)
            if code_col is None:
                continue
            for code in df[code_col].dropna().astype(str):
                if code.strip():
                    results.append((code.strip(), tab_name))
        return results
    except Exception as e:
        print(f"[ui] Could not load features from sheet: {e}")
        return []



def _load_obsids_from_csv(path: str) -> list[str]:
    """Read unique obsids from a CSV file."""
    try:
        import pandas as pd
        df = pd.read_csv(path, usecols=["obsid"])
        return sorted(df["obsid"].dropna().astype(str).unique().tolist())
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════
# Config UI
# ═══════════════════════════════════════════════════════════════════════════

class ConfigUI:
    def __init__(self, drive_base="", gc=None, sheets=None):
        self.drive_base = drive_base
        self.gc = gc
        self.sheets = sheets or {}
        self._feature_tab_map = {}  # code → tab name

        # ── Load dynamic options ──
        feat_pairs = _load_feature_options(gc, self.sheets.get("features", ""))
        if feat_pairs:
            feature_codes = [code for code, _ in feat_pairs]
            self._feature_tab_map = {code: tab for code, tab in feat_pairs}
            # Group labels: "Directions (Conceptual)"
            feature_labels = [f"{code} ({tab})" for code, tab in feat_pairs]
        else:
            feature_codes = [
                "Offtask", "Recording", "Directions", "Coordinate",
                "Competent", "Language", "Understanding", "Tool",
                "claim", "reason", "agree", "disagree", "compare",
                "addon", "question", "revoice", "monitor", "nextstep",
                "redirect", "compliment", "apology",
            ]
            feature_labels = feature_codes
        self._feature_codes = feature_codes

        self._obsid_cache = {}

        # ── Validation CSV path (text input) ──
        self.validation_file = widgets.Text(
            value="",
            description="Validation CSV",
            placeholder="/content/drive/.../mol_videoset_annotated_updated_5626.csv",
            style=STYLE, layout=LAYOUT,
        )

        # ── Obs IDs (auto-populated from validation CSV) ──
        self.obs_list = widgets.SelectMultiple(
            options=[], description="Observation IDs",
            style=STYLE, layout=LAYOUT, rows=8,
        )
        self.obs_all = widgets.Checkbox(
            value=False, description="Run all observations",
            style=STYLE,
        )
        self.obs_all.observe(self._on_obs_all_change, names="value")

        # ── Core ──
        self.models = widgets.SelectMultiple(
            options=MODEL_OPTIONS, value=["gpt-5-mini"],
            description="Models", style=STYLE, layout=LAYOUT, rows=6,
        )
        self.features = widgets.SelectMultiple(
            options=list(zip(feature_labels, feature_codes)),
            value=["Directions", "Coordinate"] if "Directions" in feature_codes else feature_codes[:2],
            description="Features", style=STYLE, layout=LAYOUT, rows=min(10, len(feature_codes)),
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
        _cb = {"description_width": "initial"}
        _cbl = widgets.Layout(width="500px")
        self.test_mode = widgets.Dropdown(
            options=[
                ("1 segment (default test)", "1_segment"),
                ("1 transcript (all segments)", "1_transcript"),
                ("N rows", "n_rows"),
                ("Full run", "full"),
            ],
            value="1_segment",
            description="Test mode",
            style=_cb, layout=_cbl,
        )
        self.test_n_rows = widgets.IntText(
            value=20, description="N rows (if N rows mode)",
            style=_cb, layout=_cbl,
        )
        self.wait = widgets.Checkbox(value=False, description="Wait for batch to complete", style=_cb, layout=_cbl)
        self.use_video = widgets.Checkbox(value=False, description="Include video (Gemini only)", style=_cb, layout=_cbl)
        self.verbose = widgets.Checkbox(value=True, description="Show detailed logs", style=_cb, layout=_cbl)
        self.resume_mode = widgets.Checkbox(value=False, description="Resume (skip submission, fetch results)", style=_cb, layout=_cbl)
        self.evaluate_only = widgets.Checkbox(value=False, description="Evaluate only (skip annotation, compare previous run to validation)", style=_cb, layout=_cbl)

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

        # ── Data sources ──
        self.sheet_source = widgets.Text(
            value=self.sheets.get("features", "1iIzzfXqq2nYMSbzDgu2wQJqnIIwwUY6r"),
            description="Feature sheet ID", style=STYLE, layout=LAYOUT,
        )
        self.save_dir = widgets.Text(
            value=drive_base + "/result/" if drive_base else "result/",
            description="Save directory", style=STYLE, layout=LAYOUT,
        )

    def _on_validation_change(self, change):
        path = change["new"]
        if not path:
            self.obs_list.options = []
            return
        if path not in self._obsid_cache:
            self._obsid_cache[path] = _load_obsids_from_csv(path)
        obsids = self._obsid_cache[path]
        self.obs_list.options = obsids
        if obsids:
            self.obs_list.value = [obsids[0]]

    def _on_obs_all_change(self, change):
        if change["new"]:
            self.obs_list.disabled = True
        else:
            self.obs_list.disabled = False

    def display(self):
        display(widgets.HTML("<h3>🔬 Experiment Config</h3>"))

        display(widgets.HTML("<b>Data source</b>"))
        display(self.validation_file)
        display(widgets.HBox([self.obs_list, widgets.VBox([self.obs_all])]))

        display(widgets.HTML("<b>What to run</b>"))
        display(self.models, self.features)

        display(widgets.HTML("<b>Prompt settings</b>"))
        display(self.n_uttr, self.bwd, self.fwd)

        display(widgets.HTML("<b>Run options</b>"))
        display(widgets.VBox([
            self.test_mode, self.test_n_rows, self.wait, self.use_video,
            self.verbose, self.resume_mode, self.evaluate_only,
        ], layout=widgets.Layout(width="400px")))

        rules_accordion = widgets.Accordion(children=[
            widgets.VBox([
                widgets.HTML("<p style='color:gray'>Override sheet defaults. Format: <code>FeatureName: value</code>, one per line.</p>"),
                self.filter_if_text,
                self.linked_with_text,
                self.subcode_of_text,
                self.extra_context_type_text,
                self.extra_context_text,
            ]),
        ])
        rules_accordion.set_title(0, "Feature rule overrides (optional)")
        rules_accordion.selected_index = None
        display(rules_accordion)

        advanced = widgets.Accordion(children=[
            widgets.VBox([
                self.resume_text,
                self.sheet_source,
                self.save_dir,
            ]),
        ])
        advanced.set_title(0, "Advanced / Resume (optional)")
        advanced.selected_index = None
        display(advanced)

    def build_config(self) -> ExperimentConfig:
        if self.obs_all.value:
            obs_list = "all"
        else:
            obs_list = list(self.obs_list.value) if self.obs_list.value else "all"

        return ExperimentConfig(
            model_list=list(self.models.value),
            feature_list=list(self.features.value),
            obs_list=obs_list,
            sheet_source=self.sheet_source.value,
            bwd_context_count=self.bwd.value,
            fwd_context_count=self.fwd.value,
            n_uttr=self.n_uttr.value,
            test_mode=self.test_mode.value,
            test_n_rows=self.test_n_rows.value,
            if_wait=self.wait.value,
            use_video=self.use_video.value,
            save_dir=self.save_dir.value,
            evaluate_only=self.evaluate_only.value,
            resume_batch_ids=_parse_dict(self.resume_text.value),
            filter_if=_parse_list_dict(self.filter_if_text.value),
            linked_with=_parse_list_dict(self.linked_with_text.value),
            subcode_of=_parse_dict(self.subcode_of_text.value),
            extra_context_type=_parse_dict(self.extra_context_type_text.value),
            extra_context=_parse_dict(self.extra_context_text.value),
        )

    def get_validation_path(self) -> str:
        return self.validation_file.value or ""


def _parse_dict(text: str) -> dict:
    result = {}
    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        result[key.strip()] = val.strip()
    return result


def _parse_list_dict(text: str) -> dict:
    result = {}
    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        result[key.strip()] = [v.strip() for v in val.split(",") if v.strip()]
    return result


def show_config_ui(drive_base="", gc=None, sheets=None) -> ConfigUI:
    """Create and display the config UI. Returns the UI object."""
    ui = ConfigUI(drive_base=drive_base, gc=gc, sheets=sheets)
    ui.display()
    return ui
