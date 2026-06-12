"""Colab widget UI for experiment configuration.

Auto-populates features from the MOL Roles Features sheet and validation
files from Drive. New sheet tabs (e.g. Multimodal) appear automatically.

Usage:
    from llm_annotator.ui import show_config_ui
    ui = show_config_ui(gc=gc, drive_base=DRIVE_BASE, sheets=SHEETS)
    config = ui.build_config()
"""

import os
import glob
import ipywidgets as widgets
from IPython.display import display, HTML
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
        spreadsheet = gc.open_by_key(features_sheet_id)
        results = []
        for ws in spreadsheet.worksheets():
            data = ws.get_all_values()
            if len(data) < 2:
                continue
            headers = data[0]
            code_idx = next((i for i, h in enumerate(headers) if h.strip().lower() == "code"), None)
            if code_idx is None:
                continue
            for row in data[1:]:
                if code_idx < len(row) and row[code_idx].strip():
                    results.append((row[code_idx].strip(), ws.title))
        return results
    except Exception as e:
        print(f"[ui] Could not load features from sheet: {e}")
        return []


def _find_validation_csvs(drive_base: str) -> list[str]:
    """Find CSV files in the pipeline outputs folder."""
    search_dir = os.path.join(drive_base, "MOL Conceptual Pipeline Outputs")
    if not os.path.isdir(search_dir):
        return []
    csvs = sorted(glob.glob(os.path.join(search_dir, "*.csv")))
    return csvs


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
        feat_pairs = _load_feature_options(gc, self.sheets.get("features_definitions", ""))
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

        val_csvs = _find_validation_csvs(drive_base)
        val_options = [(os.path.basename(p), p) for p in val_csvs]
        self._obsid_cache = {}

        # ── Validation file selector ──
        self.validation_file = widgets.Dropdown(
            options=[("(none)", "")] + val_options,
            value="",
            description="Validation CSV",
            style=STYLE, layout=LAYOUT,
        )
        self.validation_file.observe(self._on_validation_change, names="value")

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
        self.test_mode = widgets.Checkbox(value=True, description="Test mode (~20 rows)", style=STYLE)
        self.wait = widgets.Checkbox(value=False, description="Wait for batch to complete", style=STYLE)
        self.use_video = widgets.Checkbox(value=False, description="Include video (Gemini only)", style=STYLE)
        self.verbose = widgets.Checkbox(value=True, description="Show detailed logs", style=STYLE)
        self.resume_mode = widgets.Checkbox(value=False, description="Resume (skip submission, fetch results)", style=STYLE)

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
        self.transcript_source = widgets.Text(
            value=drive_base + "/MOL Conceptual Pipeline Outputs/mol_formatted_data.csv" if drive_base else "",
            description="Transcript source", style=STYLE, layout=LAYOUT,
        )
        self.sheet_source = widgets.Text(
            value=self.sheets.get("features", "1miMC8M_UkfY_3XhglTAdt9sC9H8D6d9slzC7_LgcLOc"),
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
        display(HTML("<h3>🔬 Experiment Config</h3>"))

        display(HTML("<b>Data source</b>"))
        display(self.validation_file)
        display(widgets.HBox([self.obs_list, widgets.VBox([self.obs_all])]))

        display(HTML("<b>What to run</b>"))
        display(self.models, self.features)

        display(HTML("<b>Prompt settings</b>"))
        display(self.n_uttr, self.bwd, self.fwd)

        display(HTML("<b>Run options</b>"))
        display(self.test_mode, self.wait, self.use_video, self.verbose, self.resume_mode)

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
        rules_accordion.selected_index = None
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
        if self.obs_all.value:
            obs_list = "all"
        else:
            obs_list = list(self.obs_list.value) if self.obs_list.value else "all"

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
