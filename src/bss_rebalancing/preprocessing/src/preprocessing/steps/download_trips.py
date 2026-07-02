"""
Download trip data from BlueBikes.

This module downloads and extracts the BlueBikes trip data for the specified year.
"""

import os
import shutil
import zipfile
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from preprocessing.config import PreprocessingConfig
from preprocessing.core.sources import Source
from preprocessing.core.converter import TripDataConverter


def download_file(url: str, target_directory: str, tbar: tqdm = None) -> str | None:
    """
    Download a single file. Does NOT extract — that's a separate phase now,
    on purpose, so a mid-batch network failure doesn't leave you standing
    in a half-extracted crime scene.

    Returns the saved path, or None if the download failed (already logged).
    """
    os.makedirs(target_directory, exist_ok=True)
    filename = os.path.basename(urlparse(url).path)
    save_path = os.path.join(target_directory, filename)

    if tbar is not None:
        tbar.set_description(f"Downloading {filename}")
    else:
        print(f"Downloading file from {url}...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return save_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None


def extract_zip(zip_path: str, target_directory: str, tbar: tqdm = None) -> bool:
    """
    Extract a single zip in place. Zip is kept — nobody deletes it, we
    learned that lesson already. Returns True on success.
    """
    filename = os.path.basename(zip_path)

    if not zipfile.is_zipfile(zip_path):
        print(f"{filename} is not a valid ZIP archive. Skipping extraction — "
              f"either the download is corrupt or the provider handed you garbage.")
        return False

    if tbar is not None:
        tbar.set_description(f"Extracting {filename}")
    else:
        print(f"Extracting {zip_path}...")

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_directory)
        return True
    except zipfile.BadZipFile as e:
        print(f"Corrupt zip, extraction failed for {filename}: {e}")
        return False


def run(config: PreprocessingConfig, source: Source) -> None:
    save_path = config.trips_path
    os.makedirs(save_path, exist_ok=True)

    all_month_pairs = [(year, month) for year in source.years for month in source.months]

    # ── Phase 1: download every zip first. All of them. No extraction yet. ──
    zip_paths = []
    tbar = tqdm(all_month_pairs, desc="Downloading files", position=0, leave=True)
    for year, month in all_month_pairs:
        filename = source.zip_filename(year, month)
        zip_path = os.path.join(save_path, filename)
        if os.path.exists(zip_path):
            tbar.set_description(f"Already have {filename}, skipping")
        else:
            downloaded = download_file(source.url_for(year, month), save_path, tbar)
            if downloaded is None:
                # Don't silently swallow a failed month — you'd rather find
                # out now than three steps deeper when preprocess_data.py
                # quietly loads incomplete data and gives you numbers that
                # look plausible but are wrong. Wrong-but-plausible is the
                # worst kind of bug.
                raise RuntimeError(
                    f"Failed to download {filename} for source '{source.id}'. "
                    f"Aborting before touching anything already on disk."
                )
        zip_paths.append(zip_path)
        tbar.update(1)
    tbar.close()

    # ── Phase 2: extract everything, now that we know every zip landed. ──
    ebar = tqdm(zip_paths, desc="Extracting files", position=0, leave=True)
    for zip_path in zip_paths:
        extract_zip(zip_path, save_path, ebar)
        ebar.update(1)
    ebar.close()

    macosx_path = os.path.join(save_path, "__MACOSX")
    if os.path.exists(macosx_path):
        shutil.rmtree(macosx_path)

    # ── Phase 3: hand off to the converter. One merged CSV, raw CSVs purged. ──
    output_csv = os.path.join(save_path, f"{source.id}-{source.month_str}-tripdata.csv")
    converter = TripDataConverter(source)
    converter.convert(trips_dir=save_path, output_file=output_csv, cleanup=True)
