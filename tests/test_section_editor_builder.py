import json
from pathlib import Path

import pytest

from experiments.build_section_editor import build, editor_data, embedded_json


def test_prediction_information_not_passed_to_blank_editor():
    review = {'song_title': 'test', 'waveform': [{'time_s':0, 'value':0}],
              'sections': {'candidate':[{'label':'outro','start_s':42}]},
              'boundary_questions':[{'time_s':42}], 'repeat_questions':[{}]}
    data = editor_data(review, 'a'*64, 'b'*64, 90)
    assert set(data) == {'schema_version','song_title','duration_s','audio_path','audio_sha256','source_audio_sha256','waveform'}
    assert data['audio_sha256'] == 'a'*64
    assert data['duration_s'] == 90


def test_embedded_data_cannot_end_script_and_roundtrips():
    data = {'song_title':'</script><script>alert("x")</script>&'}
    encoded = embedded_json(data)
    assert '<' not in encoded and '&' not in encoded
    assert json.loads(encoded) == data


def test_never_overwrite_existing_review(tmp_path: Path):
    with pytest.raises(FileExistsError):
        build(tmp_path/'missing', tmp_path)


def test_no_nested_output(tmp_path: Path):
    with pytest.raises(ValueError, match='inside'):
        build(tmp_path, tmp_path/'new')
    assert not (tmp_path/'new').exists()
