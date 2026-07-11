"""Download the example datasets and pretrained checkpoint bundled with this project from
their hosted sources, instead of committing large binary/data files to git.

Two sources are used:
- Zenodo record 7157412 (https://zenodo.org/records/7157412), for files that are part of
  that record.
- A GitHub release (release-assets-v1, on this repository's `release-assets` orphan branch),
  for files that are not on Zenodo: the pretrained checkpoint in its converted format.

Every download streams to disk in chunks - the full file is never held in memory - and is
verified against a known MD5 checksum before being moved into place.
"""
from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ZENODO_RECORD = "7157412"
_ZENODO_URL = "https://zenodo.org/records/{record}/files/{filename}?download=1"

GITHUB_REPO = "OlivierBeq/SMILES-corrector"
GITHUB_RELEASE_TAG = "release-assets-v1"
_GITHUB_RELEASE_URL = "https://github.com/{repo}/releases/download/{tag}/{filename}"


def _zenodo_url(filename: str) -> str:
    return _ZENODO_URL.format(record=ZENODO_RECORD, filename=filename)


def _release_url(filename: str) -> str:
    return _GITHUB_RELEASE_URL.format(repo=GITHUB_REPO, tag=GITHUB_RELEASE_TAG, filename=filename)


@dataclass(frozen=True)
class Asset:
    dest: str              # path relative to the repo root
    url: str | None        # None until a hosting URL is configured
    md5: str | None = None


ASSETS: list[Asset] = [
    Asset("rawdata/gbd_8.csv", _zenodo_url("gbd_8.csv"), "e82cf958f1d22f5b3c0d096aab13b449"),
    Asset("data/selective_ki.csv", _zenodo_url("selective_ki.csv"), "d3998addd8e4472b8d40f231921eda38"),
    Asset("generated/rnn.tsv", _zenodo_url("rnn.tsv"), "8a8a23c4f227379315cc92a614a83048"),
    Asset("generated/vae.csv", _zenodo_url("vae.csv"), "63293ed363764e94d1069e2592b8d7c0"),
    Asset(
        "data/errors/PAPYRUS_200_multiple_12_errors.csv",
        _zenodo_url("PAPYRUS_200_multiple_12_errors.csv"),
        "9a49d09e229e9b2979ea0d411d4dbdba",
    ),
    # Zenodo's "gan.csv" and "rnn_target_directed.tsv" - confirmed byte-identical (modulo
    # line endings) to these locally-named files.
    Asset("generated/gan_ckpt100.csv", _zenodo_url("gan.csv"), "b7d8854083240823e96e9ecfec27cbb2"),
    Asset(
        "generated/rl.tsv", _zenodo_url("rnn_target_directed.tsv"),
        "39b82702a6606db151cab682750d126c",
    ),
    # Not on Zenodo - hosted as a GitHub release asset (see module docstring). This is the
    # converted checkpoint (bundled state_dict + hyperparams + vocab), verified bit-for-bit
    # identical in behavior to the original Zenodo weights.
    Asset(
        "data/performance/transformer_multiple_12_PAPYRUS_200_16_3.pkg",
        _release_url("transformer_multiple_12_PAPYRUS_200_16_3.pkg"),
        "d413560926b409d0311cd3a93fc6f77d",
    ),
]


def _md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest: Path, md5: str | None = None, chunk_size: int = 1 << 20) -> None:
    """Streams url to dest (via a .part temp file, moved into place only after a successful,
    checksum-verified download), creating parent directories as needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(dest) + ".part")
    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    if md5 is not None:
        actual = _md5sum(tmp_path)
        if actual != md5:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(f"checksum mismatch for {dest}: expected {md5}, got {actual}")
    tmp_path.replace(dest)


def fetch_all(root: str = ".", only: list[str] | None = None, force: bool = False) -> None:
    """Downloads every configured asset into `root` (matching this project's own directory
    layout - data/, generated/, rawdata/). Already-present files are skipped unless `force`.
    `only` restricts to specific `dest` paths (e.g. ["rawdata/gbd_8.csv"])."""
    root_path = Path(root)
    for asset in ASSETS:
        if only is not None and asset.dest not in only:
            continue
        dest = root_path / asset.dest
        if dest.exists() and not force:
            print(f"skip (already present): {asset.dest}")
            continue
        if asset.url is None:
            print(f"skip (no hosting URL configured yet): {asset.dest}")
            continue
        print(f"downloading {asset.dest} ...")
        download_file(asset.url, dest, asset.md5)
        print(f"  done -> {dest}")
