"""Download student materials (PNGs) from Google Drive folders.

Reads "Student Materials Folder Link" from the Tracker sheet, lists image
files in that Drive folder, and downloads them with structured naming:
    {obsid}_{segment_letter}_{i}.png   — when segment mapping is known
    {obsid}_{i}.png                    — when segments are unknown
"""

import os
import re
from typing import Optional


def list_drive_folder_images(gdrive, folder_id: str) -> list[dict]:
    """List PNG/JPG files in a Drive folder.

    Returns list of dicts with keys: name, file_id, mime_type.
    """
    file_list = gdrive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=false and "
             f"(mimeType='image/png' or mimeType='image/jpeg')"
    }).GetList()
    return [
        {"name": f["title"], "file_id": f["id"], "mime_type": f["mimeType"]}
        for f in sorted(file_list, key=lambda f: f["title"])
    ]


def list_drive_folder_videos(gdrive, folder_id: str, obsid: int) -> list[dict]:
    """List video files matching OBS-{yy}-{obsid}_video* in a Drive folder."""
    file_list = gdrive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=false"
    }).GetList()

    prefix = f"OBS-25-{obsid:04d}"
    alt_prefix = f"OBS-26-{obsid:04d}"
    videos = []
    for f in sorted(file_list, key=lambda f: f["title"]):
        name = f["title"]
        if name.startswith(prefix) or name.startswith(alt_prefix):
            vid_num = 1
            m = re.search(r'video[_\s]*(\d+)', name, re.IGNORECASE)
            if m:
                vid_num = int(m.group(1))
            videos.append({
                "name": name, "file_id": f["id"],
                "video_index": vid_num, "size": f.get("fileSize", "?"),
            })
    return videos


def extract_folder_id(url: str) -> Optional[str]:
    """Extract Drive folder ID from a URL."""
    if not url:
        return None
    m = re.search(r'/folders/([a-zA-Z0-9_-]+)', url)
    if m:
        return m.group(1)
    m = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    return m.group(1) if m else None


def build_download_plan(
    images: list[dict],
    obsid: int,
    segment_map: Optional[dict] = None,
) -> list[dict]:
    """Build a list of {src_name, dest_name, file_id} for downloading images.

    If segment_map is provided (letter → image indices), uses obsid_letter_i naming.
    Otherwise falls back to obsid_i naming.

    segment_map example: {"a": [0, 1], "b": [2]}  (indices into images list)
    """
    plan = []
    if segment_map:
        for letter in sorted(segment_map.keys()):
            for seq, img_idx in enumerate(segment_map[letter], 1):
                if img_idx < len(images):
                    img = images[img_idx]
                    ext = os.path.splitext(img["name"])[1] or ".png"
                    dest = f"{obsid}_{letter}_{seq}{ext}"
                    plan.append({"src_name": img["name"], "dest_name": dest, "file_id": img["file_id"]})
    else:
        for i, img in enumerate(images, 1):
            ext = os.path.splitext(img["name"])[1] or ".png"
            dest = f"{obsid}_{i}{ext}"
            plan.append({"src_name": img["name"], "dest_name": dest, "file_id": img["file_id"]})
    return plan


def download_images(gdrive, plan: list[dict], dest_dir: str, dry_run: bool = False) -> list[str]:
    """Download images according to the plan. Returns list of saved paths."""
    os.makedirs(dest_dir, exist_ok=True)
    paths = []
    for item in plan:
        out_path = os.path.join(dest_dir, item["dest_name"])
        if dry_run:
            print(f"  [dry-run] {item['src_name']} → {item['dest_name']}")
        else:
            f = gdrive.CreateFile({"id": item["file_id"]})
            f.GetContentFile(out_path)
            print(f"  ✓ {item['dest_name']}")
        paths.append(out_path)
    return paths
