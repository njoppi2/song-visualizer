from pathlib import Path

import pytest

from experiments.build_structure_review import boundary_questions, build, choose_repeats


def test_existing_directory_is_never_overwritten(tmp_path: Path):
    with pytest.raises(FileExistsError):
        build(tmp_path, tmp_path/'missing-parent')


def test_boundary_questions_include_disagreement_and_context_bounds():
    old = {'sections': [{'start_s': t} for t in [0, 5, 40, 43, 80]]}
    new = {'sections': [{'start_s': t} for t in [0, 5, 43, 80]]}
    questions = boundary_questions(old, new, 100)
    assert any(q['time_s'] == 40 for q in questions)
    assert all(0 <= q['start_s'] < q['end_s'] <= 100 for q in questions)
    assert len({q['id'] for q in questions}) == len(questions)


def test_repetition_selection_includes_both_scales_and_contrast():
    spans = [{'start_s': s, 'end_s': s+8} for s in [0, 40, 80, 120]]
    results = [{'scale_beats': scale, 'spans': spans,
                'pairs': [{'a': 0, 'b': b, 'similarity': score, 'stem_similarities': {'mix': score}}
                          for b, score in [(1, .95), (2, .8), (3, .1)]]} for scale in [16, 32]]
    selected = choose_repeats(results)
    assert len(selected) == 3
    assert {q['scale_beats'] for q in selected} == {16,32}
    assert selected[-1]['similarity'] == .1


def test_no_recurrence_evidence_produces_no_leading_questions():
    assert choose_repeats([{'scale_beats': 16, 'spans': [], 'pairs': []}]) == []
