#!/usr/bin/env python3
"""Download the two small NLTK resources speech-to-speech imports at startup.

NLTK's own downloader refuses hosts that resolve to a private address (some
local DNS/VPN setups do), so fetch the official nltk_data archives directly and
keep them inside this project's .nltk_data directory.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NLTK_DIR = ROOT / ".nltk_data"

PACKAGES = {
    "tokenizers/punkt_tab": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip",
    "taggers/averaged_perceptron_tagger_eng": (
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/"
        "taggers/averaged_perceptron_tagger_eng.zip"
    ),
}


def _download(url: str) -> bytes:
    import urllib.request

    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def ensure() -> Path:
    """Make sure both resources exist, then point NLTK at the local copy."""
    NLTK_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["NLTK_DATA"] = str(NLTK_DIR)
    for relative, url in PACKAGES.items():
        destination = NLTK_DIR / relative
        if destination.exists():
            continue
        print(f"Downloading NLTK resource: {relative}", flush=True)
        archive = zipfile.ZipFile(io.BytesIO(_download(url)))
        parent = destination.parent
        for member in archive.namelist():
            target = (parent / member).resolve()
            if parent.resolve() not in target.parents and target != parent.resolve():
                raise RuntimeError(f"Unsafe path in NLTK archive: {member}")
        archive.extractall(parent)
    return NLTK_DIR


if __name__ == "__main__":
    print(ensure())
