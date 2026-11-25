"""Download the study's six UCI datasets once, then check that every task loads.

    python fetch_data.py

Each dataset is one zip file from the UCI Machine Learning Repository, licensed
CC BY 4.0. Only the plain files at the top of the zip are written, as bytes, to
data/<dataset>/. Nothing in a download is run. A dataset folder that already
exists is not downloaded again, so the command is safe to rerun, and an
interrupted download leaves only a <dataset>.partial folder that the next run replaces.

Then every task in arrfs.data.TASKS is loaded the way the experiment loads it,
which fails on any size the study does not expect, and its class counts and
missing values are printed. data/SOURCES.txt records where each file came from.
"""
import hashlib
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from arrfs.data import DATA_DIR, TASKS, load_task

UCI = "https://archive.ics.uci.edu"
SOURCES = {  # folder in data/: (UCI dataset id, name in the UCI URLs, DOI)
    "arrhythmia": (5, "arrhythmia", "10.24432/C5BS32"),
    "lymphography": (63, "lymphography", "10.24432/C54598"),
    "dermatology": (33, "dermatology", "10.24432/C5FK5P"),
    "spect": (95, "spect+heart", "10.24432/C5P304"),
    "statlog-heart": (145, "statlog+heart", "10.24432/C57303"),
    "wdbc": (17, "breast+cancer+wisconsin+diagnostic", "10.24432/C5DW2B"),
}


def fetch(folder: str, uci_id: int, slug: str) -> None:
    """Write the top-level files of one UCI zip into data/<folder>/, unless that folder exists."""
    target = DATA_DIR / folder
    if target.exists():
        return
    url = f"{UCI}/static/public/{uci_id}/{slug}.zip"
    request = urllib.request.Request(url, headers={"User-Agent": "dataset-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    partial = DATA_DIR / f"{folder}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    for member in archive.infolist():
        if not member.is_dir() and Path(member.filename).name == member.filename:
            (partial / member.filename).write_bytes(archive.read(member))
    partial.rename(target)
    print(f"downloaded {url} -> {target}")


def write_sources() -> None:
    lines = ["Downloaded by fetch_data.py from the UCI Machine Learning Repository.",
             "Every dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0),",
             "https://creativecommons.org/licenses/by/4.0/", ""]
    for folder, (uci_id, slug, doi) in SOURCES.items():
        lines += [f"{folder}/", f"  page   {UCI}/dataset/{uci_id}/{slug}",
                  f"  zip    {UCI}/static/public/{uci_id}/{slug}.zip", f"  doi    https://doi.org/{doi}"]
        lines += [f"  sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}"
                  for p in sorted((DATA_DIR / folder).iterdir())]
        lines.append("")
    (DATA_DIR / "SOURCES.txt").write_text("\n".join(lines))


def describe(task: str) -> str:
    """For example 'wdbc: 569 records x 30 attributes; classes benign 357, malignant 212; no missing values'."""
    data = load_task(task)
    counts = ", ".join(f"{name} {n}" for name, n in zip(data.class_names, np.bincount(data.y)))
    missing = [f"{name} {n}" for name, n in zip(data.feature_names, np.isnan(data.X).sum(axis=0)) if n]
    return (f"{task}: {len(data.y)} records x {data.X.shape[1]} attributes; classes {counts}; "
            + (f"missing values in {', '.join(missing)}" if missing else "no missing values"))


if __name__ == "__main__":
    for folder, (uci_id, slug, _) in SOURCES.items():
        fetch(folder, uci_id, slug)
    write_sources()
    for task in TASKS:
        print(describe(task))
