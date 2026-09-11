from pathlib import Path

import pytest

from experiments.repair_structure_review import repair


def test_existing_review_is_never_overwritten(tmp_path: Path):
    with pytest.raises(FileExistsError):
        repair(tmp_path/'missing', tmp_path)


def test_nested_output_rejected_before_reading_or_writing(tmp_path: Path):
    with pytest.raises(ValueError, match='inside'):
        repair(tmp_path, tmp_path/'nested')
    assert not (tmp_path/'nested').exists()
