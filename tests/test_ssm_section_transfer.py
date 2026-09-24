from experiments.run_ssm_section_transfer import (
    boundaries_in_window,
    fgi_distance_diagnostics,
    screen_gate,
)


def test_fixed_windows_are_closed_and_count_only_final_boundaries():
    boundaries = [92.633333333, 94.0, 95.633333333, 96.0]
    assert boundaries_in_window(boundaries, 92.633333333, 95.633333333) == boundaries[:3]
    assert boundaries_in_window([16.0, 21.0, 26.0, 26.001], 16.0, 26.0) == [16.0, 21.0, 26.0]


def test_gate_requires_both_positive_windows_and_empty_fgi_control_only():
    passing = screen_gate({
        "agnes": [105.0],
        "castlecomer": [94.0],
        "arctic_monkeys": [141.0],
        "feel_good_inc": [],
    })
    assert passing["status"] == "pass"
    assert passing["arctic_monkeys_not_scored"] is True

    assert screen_gate({
        "agnes": [], "castlecomer": [94.0], "arctic_monkeys": [], "feel_good_inc": [],
    })["status"] == "fail"
    assert screen_gate({
        "agnes": [105.0], "castlecomer": [94.0], "arctic_monkeys": [], "feel_good_inc": [20.0],
    })["status"] == "fail"


def test_human_distance_diagnostic_is_descriptive_and_uses_fixed_cutoffs():
    diagnostic = fgi_distance_diagnostics([5.35], [5.35, 10.0])
    assert diagnostic["kind"] == "descriptive_in_sample_diagnostic_not_a_scoring_tolerance"
    assert diagnostic["human_boundaries_with_nearest_final_boundary_within_cutoff"] == {
        "0.5": 1, "1.0": 1, "3.0": 1,
    }
    assert diagnostic["human_boundaries"][1]["nearest_final_boundary_s"] == 5.35
