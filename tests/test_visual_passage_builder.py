from pathlib import Path

import pytest

from experiments.build_visual_passage import verify_hash
from songviz.ingest import sha256_file


def test_review_input_checksum_rejects_changed_evidence(tmp_path: Path):
    path = tmp_path / "input.json"
    path.write_text('{"beat": 1}')
    original = sha256_file(path)
    verify_hash(path, original)
    path.write_text('{"beat": 2}')
    with pytest.raises(ValueError, match="Review input changed"):
        verify_hash(path, original)
