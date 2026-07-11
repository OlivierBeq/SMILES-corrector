import hashlib
import io
from unittest.mock import patch

import pytest

from uncorrupt_smiles.fetch_data import Asset, download_file, fetch_all


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(content: bytes):
    def _opener(url):
        return _FakeResponse(content)
    return _opener


def test_download_file_writes_content_and_verifies_checksum(tmp_path):
    content = b"hello world" * 1000
    md5 = hashlib.md5(content).hexdigest()
    dest = tmp_path / "sub" / "file.csv"

    with patch("urllib.request.urlopen", _fake_urlopen(content)):
        download_file("http://example.invalid/file.csv", dest, md5=md5, chunk_size=17)

    assert dest.exists()
    assert dest.read_bytes() == content
    assert not (tmp_path / "sub" / "file.csv.part").exists()


def test_download_file_rejects_checksum_mismatch(tmp_path):
    content = b"some content"
    dest = tmp_path / "file.csv"

    with patch("urllib.request.urlopen", _fake_urlopen(content)):
        with pytest.raises(ValueError, match="checksum mismatch"):
            download_file("http://example.invalid/file.csv", dest, md5="0" * 32)

    assert not dest.exists()
    assert not (tmp_path / "file.csv.part").exists()


def test_download_file_without_checksum_still_writes(tmp_path):
    content = b"no checksum provided"
    dest = tmp_path / "file.csv"
    with patch("urllib.request.urlopen", _fake_urlopen(content)):
        download_file("http://example.invalid/file.csv", dest)
    assert dest.read_bytes() == content


def test_fetch_all_skips_existing_files_unless_forced(tmp_path):
    dest = tmp_path / "rawdata" / "gbd_8.csv"
    dest.parent.mkdir(parents=True)
    dest.write_text("already here")

    assets = [Asset("rawdata/gbd_8.csv", "http://example.invalid/gbd_8.csv", None)]
    with patch("uncorrupt_smiles.fetch_data.ASSETS", assets):
        with patch("uncorrupt_smiles.fetch_data.download_file") as mock_dl:
            fetch_all(root=str(tmp_path))
            mock_dl.assert_not_called()

            fetch_all(root=str(tmp_path), force=True)
            mock_dl.assert_called_once()


def test_fetch_all_skips_assets_with_no_configured_url(tmp_path):
    assets = [Asset("generated/pending.csv", None, None)]
    with patch("uncorrupt_smiles.fetch_data.ASSETS", assets):
        with patch("uncorrupt_smiles.fetch_data.download_file") as mock_dl:
            fetch_all(root=str(tmp_path))
            mock_dl.assert_not_called()
    assert not (tmp_path / "generated" / "pending.csv").exists()


def test_fetch_all_only_filters_to_requested_dest_paths(tmp_path):
    assets = [
        Asset("a.csv", "http://example.invalid/a.csv", None),
        Asset("b.csv", "http://example.invalid/b.csv", None),
    ]
    with patch("uncorrupt_smiles.fetch_data.ASSETS", assets):
        with patch("uncorrupt_smiles.fetch_data.download_file") as mock_dl:
            fetch_all(root=str(tmp_path), only=["b.csv"])
            assert mock_dl.call_count == 1
            assert mock_dl.call_args[0][0] == "http://example.invalid/b.csv"
