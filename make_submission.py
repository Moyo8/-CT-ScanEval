"""Build the hackathon ZIP and run the pre-submission safety checks.

    python make_submission.py   ->  dist/CT-ScanEval_submission.zip

Excludes raw DICOM data, .env, caches and videos; fails if the ZIP is over
50 MB or if anything that looks like an API key is found.
"""
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "dist" / "CT-ScanEval_submission.zip"
INCLUDE = [
    "README.md", "requirements.txt", "requirements-data.txt", ".env.example", ".gitignore",
    "run_eval.py", "make_submission.py", "CT_ScanEval_Colab.ipynb",
    "tasks", "reference", "graders", "harness", "analysis", "tests", "data_prep",
    "review", "results", "transcripts",
]
SKIP_PARTS = {"__pycache__", ".pytest_cache", "raw", ".venv", "venv"}
SKIP_SUFFIX = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".pyc", ".dcm"}
KEY_PATTERNS = [
    re.compile(rb"sk-ant-[A-Za-z0-9_\-]{20,}"),
    re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_\-]{32,}"),
    re.compile(rb"AIza[0-9A-Za-z_\-]{30,}"),
]
MAX_BYTES = 50 * 1024 * 1024


def files():
    for name in INCLUDE:
        p = ROOT / name
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not (set(f.relative_to(ROOT).parts) & SKIP_PARTS) and f.suffix.lower() not in SKIP_SUFFIX:
                    yield f


def main():
    problems = []
    selected = list(files())
    for f in selected:
        if f.name == ".env":
            problems.append(f"{f} is a .env file")
        data = f.read_bytes()
        for pat in KEY_PATTERNS:
            if pat.search(data):
                problems.append(f"possible API key in {f.relative_to(ROOT)}")
    if problems:
        sys.exit("NOT BUILT:\n  " + "\n  ".join(problems))
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in selected:
            zf.write(f, Path("CT-ScanEval") / f.relative_to(ROOT))
    size = OUT.stat().st_size
    with zipfile.ZipFile(OUT) as zf:
        bad = zf.testzip()
    print(f"{OUT.relative_to(ROOT)}: {len(selected)} files, {size / 1e6:.1f} MB, integrity {'OK' if bad is None else 'BAD: ' + bad}")
    if size > MAX_BYTES:
        sys.exit("ZIP is over the 50 MB limit")


if __name__ == "__main__":
    main()
