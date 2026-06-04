import gspread
import os
import pandas as pd
import numpy as np

from typing import List, Dict

import llm_annotator.utils as utils

try:
    from google.colab import drive
    from google.colab import auth
    from google.auth import default
    
    IN_COLAB = True
    print("Running in Google Colab.")

    if IN_COLAB:
        auth.authenticate_user()
        creds, _ = default()
        gc = gspread.authorize(creds)
        drive.mount('/content/drive')
        
        # Setup PyDrive for file downloads
        try:
            from pydrive.auth import GoogleAuth
            from pydrive.drive import GoogleDrive
            gauth = GoogleAuth()
            gauth.credentials = creds
            gdrive = GoogleDrive(gauth)
        except ImportError:
            print("PyDrive not available, Google Drive file downloads will be limited")
            gdrive = None
except ImportError:
    IN_COLAB = False
    print("Running in Local Environment.")
    gc = None
    gdrive = None


class DataLoader:
    def __init__(self,
                 sheet_source: str,
                 transcript_source: str,
                 save_dir: str = "../results"):
        self.gc = gc if IN_COLAB else None
        self.gdrive = gdrive if IN_COLAB else None
        self.save_dir = save_dir
        self.transcript_df = self.__load_transcript(transcript_source)
        self.sheets_data = self.__load_features(sheet_source)
        self.features = {}

    def __load_transcript(self, transcript_source: str):
        try:
            if os.path.exists(transcript_source):
                print("Loading local file")
                return pd.read_csv(transcript_source)
            else:
                # Try loading as Google Sheet first
                if self.gc:
                    try:
                        sheet = self.gc.open_by_key(transcript_source).sheet1
                        data = sheet.get_all_records()
                        return pd.DataFrame(data)
                    except:
                        # If Google Sheet fails, try as Google Drive file
                        if self.gdrive:
                            print("Trying to download from Google Drive...")
                            file = self.gdrive.CreateFile({'id': transcript_source})
                            file.GetContentFile('temp.csv')
                            return pd.read_csv('temp.csv')
                        else:
                            print("Warning: Google Drive access not available in local environment")
                            print("Please provide a local file path instead of a Google Drive ID")
                            raise ValueError("Google Drive access not available in local environment")
                else:
                    print("Warning: Google Sheets access not available in local environment")
                    print("Please provide a local file path instead of a Google Sheet ID")
                    raise ValueError("Google Sheets access not available in local environment")
        except FileNotFoundError:
            raise FileNotFoundError("Transcript file not found")

    def __load_features(self, source: str):
        # Check if the source is a local file
        if os.path.exists(source):
            feature_sheet = pd.ExcelFile(source)
            sheet_names = feature_sheet.sheet_names
            print("Available Sheets:", sheet_names)
            
            # Extract data from each sheet
            self.sheets_data = {}
            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(source, sheet_name=sheet_name)
                    
                    # Fill in the missing Code Type
                    if "Code Type" in df.columns:
                        df["Code Type"] = df["Code Type"].replace("", None).ffill()
                    
                    self.sheets_data[sheet_name] = df
                except Exception as e:
                    raise ValueError(f"Error reading sheet '{sheet_name}': {e}")

        # Check if the source is a Google Sheet ID
        else: 
            try:
                feature_sheet = self.gc.open_by_key(source)
            except:
                raise ValueError("The provided source is neither a valid local file nor a valid Google Sheet ID.")

            sheet_names = [sheet.title for sheet in feature_sheet.worksheets()]

            # Extract the individual features from seperate sheets.
            self.sheets_data = {}
            for sheet_name in sheet_names:
                try:
                    worksheet = feature_sheet.worksheet(sheet_name)
                    data = worksheet.get_all_values()
                except:
                    raise ValueError(f"The sheet '{sheet_name}' is not found.")
                df = pd.DataFrame(data[1:], columns=data[0])

                # Fill in the missing Code Type
                if "Code Type" in df.columns:
                    df["Code Type"] = df["Code Type"].replace("", None).ffill()
                self.sheets_data[sheet_name] = df

        return self.sheets_data


@utils.component("load_feature")
def load_features(dataloader: DataLoader):
    return "feature_df", dataloader.sheets_data


@utils.component("load_transcript")
def load_transcript(dataloader: DataLoader):
    return "transcript_df", dataloader.transcript_df


@utils.component("generate_features")
# TO-DO: Change the examples to be dynamic
def generate_features(dataloader: DataLoader, feature: str = [])\
        -> Dict:
    if dataloader.sheets_data is None:
        dataloader.__load_feautres()

    dataloader.features = {}
    for sheet_name, df in dataloader.sheets_data.items():  # Iterate over sheet names and dataframes
        for idx, row in df.iterrows():
            if "Code" in df.columns and feature == row['Code']:
                def _col(col):
                    return str(row[col]).strip() if col in df.columns and pd.notna(row[col]) and str(row[col]).strip() != "" else ""

                dataloader.features[feature] = {
                    "definition": row["Definition"],
                    "format": "Answer 1 if the utterance relates to the category, 0 if the utterances doesn't relate to the category.",
                    "example1": _col("example1"),
                    "example2": _col("example2"),
                    "example3": _col("example3"),
                    "nonexample1": _col("nonexample1"),
                    "nonexample2": _col("nonexample2"),
                    "nonexample3": _col("nonexample3"),
                    # Feature rules — filled in by the sheet author, never hardcoded here.
                    # filter_if: comma-separated code names; exclude utterances already
                    #   labeled with any of these codes before annotating this feature.
                    "filter_if": [c.strip() for c in _col("filter_if").split(",") if c.strip()],
                    # linked_with: run this feature together with these in one prompt.
                    "linked_with": [c.strip() for c in _col("linked_with").split(",") if c.strip()],
                    # subcode_of: this feature is a subcode of another code.
                    "subcode_of": _col("subcode_of"),
                    # extra_context_type: key into ExperimentConfig.extra_context dict.
                    "extra_context_type": _col("extra_context_type"),
                }
    return "feature_dict", dataloader.features



