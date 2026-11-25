"""Download the nine gene-expression datasets of Khaseeb et al. 2025 (their Table 1) and store them as arrays.

    python -I fetch_gems.py

They are the GEMS collection of Statnikov, Aliferis, Tsamardinos, Hardin & Levy 2005 (Bioinformatics 21:631).
Each is one MATLAB matrix with the class code in the first column and one gene per further column. The authors'
site, gems-system.org, is offline, so the files come from its copies in the Internet Archive. Each download
must have the checksum listed here, which is the checksum of the file as fetched on 2026-11-23.

Nothing in a download is run. The matrix is read with scipy, checked against the size in Khaseeb et al.'s
Table 1, and written to data/gems/<name>.npz as X (expression values as distributed), y (class index) and
classes. A file that already exists in data/gems/raw/ is checked and not downloaded again.
data/gems/SOURCES.txt records where each file came from.

This script imports nothing from the project, so that it can run with -I.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import loadmat

DATA = Path(__file__).resolve().parent / "data" / "gems"
ARCHIVE = "https://web.archive.org/web/{snapshot}if_/http://{host}/datasets/{name}.mat"
DATASETS = {  # name: (archive snapshot, host as archived, sha256 of the .mat, (instances, genes, classes) of Table 1)
    "SRBCT": ("20130106224708", "www.gems-system.org",
              "cfc9e1ff8ee4249b87bde81f8ddb83d6877bb0fa6d66bd557364895ef9e380be", (83, 2308, 4)),
    "Brain_Tumor2": ("20130106224708", "www.gems-system.org",
                     "db9b6cef72a8e0e88496a3c48e81f7d615f5c28d9e381e8740ddc5e8303e8343", (50, 10367, 4)),
    "Leukemia1": ("20130106224708", "www.gems-system.org",
                  "66bce714f4f92a8334ed330a08b500f17bc5d1c3f7deaeefb55885e9b8b9e5e7", (72, 5327, 3)),
    "DLBCL": ("20130106224708", "www.gems-system.org",
              "85d99041f90b46e5f16a1ae3e8456355d7ce3d844ea4a18a175e25caeaca5a65", (77, 5469, 2)),
    "Brain_Tumor1": ("20130106224708", "www.gems-system.org",
                     "00387ac401df59e2dbeb75a7b209edac43d63f9e57bbbf56d049ecb03a2e90c1", (90, 5920, 5)),
    "Leukemia2": ("20130106224708", "www.gems-system.org",
                  "b4a9cbda30828264de939c7d40c9b1e760281b30b627f91492a387b50c3642a5", (72, 11225, 3)),
    "Prostate_Tumor": ("20130106224708", "www.gems-system.org",
                       "23b76ddfc617686c5563696a3bbe4b2b212d3b13cd95c0fdff3502c2fa0a5453", (102, 10509, 2)),
    "11_Tumors": ("20130106224708", "www.gems-system.org",
                  "3b872c6fca8f7c4a5f2bf3c3bce005f475d4ec53bc5a3a4a08b94273d9bfb8dc", (174, 12533, 11)),
    "14_Tumors": ("20141015143421", "gems-system.org",  # the 2013 copy of this file is cut short
                  "67d565bd6a4bd5f3e84299811c4c710006a6ffbd1f13cb3628e644ccd508cd09", (308, 15009, 26)),
}


def url(name: str) -> str:
    snapshot, host, _, _ = DATASETS[name]
    return ARCHIVE.format(snapshot=snapshot, host=host, name=name)


def fetch(name: str) -> Path:
    """The dataset's .mat file in data/gems/raw/, downloaded if it is not there, with its checksum verified."""
    raw = DATA / "raw" / f"{name}.mat"
    if not raw.exists():
        request = urllib.request.Request(url(name), headers={"User-Agent": "dataset-fetch/1.0"})
        with urllib.request.urlopen(request, timeout=300) as response:
            content = response.read()
        partial = raw.with_suffix(".partial")
        partial.write_bytes(content)
        partial.rename(raw)
        print(f"downloaded {url(name)}")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    assert digest == DATASETS[name][2], f"{raw} has sha256 {digest}, not the one listed in fetch_gems.py"
    return raw


def convert(name: str, raw: Path) -> str:
    """Write data/gems/<name>.npz and return the sha256 of its arrays (X bytes, then y bytes)."""
    (matrix,) = [value for key, value in loadmat(raw).items() if not key.startswith("__")]
    codes, y = np.unique(matrix[:, 0], return_inverse=True)
    X = matrix[:, 1:].astype(np.float64)
    assert (len(y), X.shape[1], len(codes)) == DATASETS[name][3] and np.isfinite(X).all(), (name, matrix.shape, len(codes))
    y = y.astype(np.int64)
    np.savez(DATA / f"{name}.npz", X=X, y=y, classes=np.array([f"class {int(code)}" for code in codes]))
    return hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()


def main() -> int:
    (DATA / "raw").mkdir(parents=True, exist_ok=True)
    lines = ["Downloaded by fetch_gems.py: the GEMS gene-expression datasets of Statnikov, Aliferis, Tsamardinos,",
             "Hardin & Levy 2005 (Bioinformatics 21(5):631-643), as used by Khaseeb et al. 2025 (Applied Sciences 15:489).",
             "Source: the Internet Archive's copies of http://www.gems-system.org/ , the authors' site, which is offline.",
             "That page carries a copyright notice (2003-2005, the GEMS authors) and states no licence. Cite the paper",
             "above and the original study of each dataset, and do not pass the files on.", ""]
    for name, (_, _, digest, (n, genes, classes)) in DATASETS.items():
        content = convert(name, fetch(name))
        lines += [f"{name}/", f"  url     {url(name)}", f"  sha256  {digest}  raw/{name}.mat",
                  f"  content {content}  {name}.npz", f"  size    {n} instances, {genes} genes, {classes} classes", ""]
        print(f"{name:15s} {n} x {genes}, {classes} classes")
    (DATA / "SOURCES.txt").write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
