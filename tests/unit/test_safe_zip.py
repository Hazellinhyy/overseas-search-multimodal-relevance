from pathlib import Path
from zipfile import ZipFile

import pytest

from overseas_search_relevance.data.safe_zip import safe_extract


def test_safe_extract(tmp_path: Path) -> None:
    source = tmp_path / "safe.zip"
    with ZipFile(source, "w") as archive:
        archive.writestr("folder/value.txt", "ok")
    files = safe_extract(source, tmp_path / "out")
    assert files[0].read_text() == "ok"


def test_path_traversal_rejected(tmp_path: Path) -> None:
    source = tmp_path / "unsafe.zip"
    with ZipFile(source, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="不安全路径"):
        safe_extract(source, tmp_path / "out")
