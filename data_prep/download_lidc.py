"""Download selected LIDC-IDRI CT series from The Cancer Imaging Archive (TCIA).

Uses TCIA's public NBIA REST API (no account or API key needed for
LIDC-IDRI). Only the series listed in data_prep/selected_series.json are
downloaded (about 0.6 GB), not the full 130+ GB collection.

Files are stored in the layout pylidc expects:
    data/raw/LIDC-IDRI/<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm

Usage:
    python data_prep/download_lidc.py            # all selected series
    python data_prep/download_lidc.py --patient LIDC-IDRI-0294

Data licence: LIDC-IDRI is released under CC BY 3.0.
Citation DOI: 10.7937/K9/TCIA.2015.LO9QL9SX
"""
import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import requests

NBIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "LIDC-IDRI"
SELECTED = Path(__file__).resolve().parent / "selected_series.json"


def series_dir(entry):
    return RAW_DIR / entry["patient_id"] / entry["study_uid"] / entry["series_uid"]


def download_series(entry, retries=4):
    out = series_dir(entry)
    if out.exists() and any(out.glob("*.dcm")):
        print(f"  already present: {entry['patient_id']}")
        return out
    url = f"{NBIA}/getImage"
    for attempt in range(retries):
        try:
            resp = requests.get(
                url, params={"SeriesInstanceUID": entry["series_uid"]}, timeout=600
            )
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            wait = 2 ** (attempt + 1)
            print(f"  attempt {attempt + 1} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError(f"could not download {entry['series_uid']}")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".dcm"):
                (out / Path(name).name).write_bytes(zf.read(name))
    n = len(list(out.glob("*.dcm")))
    print(f"  saved {n} DICOM files -> {out.relative_to(ROOT)}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--patient", help="download only this PatientID")
    args = parser.parse_args()

    entries = json.loads(SELECTED.read_text())
    if args.patient:
        entries = [e for e in entries if e["patient_id"] == args.patient]
        if not entries:
            sys.exit(f"{args.patient} is not in {SELECTED.name}")
    for entry in entries:
        print(f"{entry['patient_id']} ({entry['series_uid'][-12:]})")
        download_series(entry)


if __name__ == "__main__":
    main()
