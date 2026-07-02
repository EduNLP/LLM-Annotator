"""Extract text from student materials PDFs for prompt injection.

Pulls "Student Materials Folder Link" from the Tracker sheet, lists PDF
files in that Drive folder, downloads them, and extracts text. The
concatenated text is returned for injection as extra_context
(key: "activity_instructions") for features like Directions.

For text-based PDFs, uses pypdf. For scanned/image PDFs, falls back to
pdf2image + pytesseract (requires tesseract installed in Colab).
"""

import os
import re
import tempfile
from typing import Optional


def get_materials_folder_id(gc, tracker_sheet_id: str, obsid: int, tracker_tab: str = "Tracker") -> Optional[str]:
    """Get the Drive folder ID for student materials from Tracker."""
    from llm_annotator.materials_loader import extract_folder_id

    ws = gc.open_by_key(tracker_sheet_id).worksheet(tracker_tab)
    all_data = ws.get_all_values()
    headers = all_data[0]

    idx_col = next((i for i, h in enumerate(headers) if h.strip().lower() == "index"), None)
    mat_col = next(
        (i for i, h in enumerate(headers) if "student materials" in h.lower() and "link" in h.lower()),
        None,
    )
    if idx_col is None or mat_col is None:
        return None

    for ri, row in enumerate(all_data[1:], 2):
        if idx_col < len(row):
            m = re.search(r"\d{2}-0*(\d+)", row[idx_col].strip())
            if m and int(m.group(1)) == int(obsid):
                from gspread.utils import rowcol_to_a1
                cell_addr = rowcol_to_a1(ri, mat_col + 1)
                formula = ws.acell(cell_addr, value_render_option="FORMULA").value
                url = None
                if formula and "HYPERLINK" in str(formula).upper():
                    hm = re.search(r'HYPERLINK\("([^"]+)"', str(formula))
                    if hm:
                        url = hm.group(1)
                if not url:
                    url = ws.acell(cell_addr).value
                return extract_folder_id(str(url)) if url else None
    return None


def list_drive_folder_pdfs(gdrive, folder_id: str) -> list[dict]:
    """List PDF files in a Drive folder."""
    file_list = gdrive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=false and mimeType='application/pdf'"
    }).GetList()
    return [
        {"name": f["title"], "file_id": f["id"]}
        for f in sorted(file_list, key=lambda f: f["title"])
    ]


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF, trying pypdf first, then OCR fallback."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if text.strip():
            return text.strip()
    except ImportError:
        pass
    except Exception:
        pass

    try:
        from pdf2image import convert_from_path
        import pytesseract
        images = convert_from_path(pdf_path)
        text = "\n".join(pytesseract.image_to_string(img) for img in images)
        return text.strip()
    except ImportError:
        print(f"  ⚠️  Could not extract text from {os.path.basename(pdf_path)}: "
              "install pypdf or (pdf2image + pytesseract) for OCR")
        return ""


def ocr_materials_folder(
    gc,
    gdrive,
    tracker_sheet_id: str,
    obsid: int,
    override_folder_id: str = "",
    tracker_tab: str = "Tracker",
) -> str:
    """Extract text from all PDFs in a materials folder.

    Args:
        gc: Authorized gspread client.
        gdrive: Authorized PyDrive GoogleDrive instance.
        tracker_sheet_id: Tracker Google Sheet ID.
        obsid: Observation ID.
        override_folder_id: If set, use this folder instead of Tracker lookup.
        tracker_tab: Tab name in Tracker.

    Returns:
        Concatenated text from all PDFs in the folder.
    """
    folder_id = override_folder_id or get_materials_folder_id(
        gc, tracker_sheet_id, obsid, tracker_tab
    )
    if not folder_id:
        print(f"  [materials] No materials folder found for obs {obsid}")
        return ""

    pdfs = list_drive_folder_pdfs(gdrive, folder_id)
    if not pdfs:
        print(f"  [materials] No PDFs in materials folder for obs {obsid}")
        return ""

    print(f"  [materials] Found {len(pdfs)} PDF(s) for obs {obsid}")
    texts = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for pdf_info in pdfs:
            local_path = os.path.join(tmpdir, pdf_info["name"])
            f = gdrive.CreateFile({"id": pdf_info["file_id"]})
            f.GetContentFile(local_path)
            text = extract_text_from_pdf(local_path)
            if text:
                texts.append(f"--- {pdf_info['name']} ---\n{text}")
                print(f"    ✓ {pdf_info['name']}: {len(text)} chars")
            else:
                print(f"    ✗ {pdf_info['name']}: no text extracted")

    return "\n\n".join(texts)
