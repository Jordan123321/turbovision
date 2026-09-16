import copy
import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

from scorevision.utils.schemas import ChallengeResponse
from scorevision.utils.snooker_image import SnookerImagePrediction, SnookerImageGroundTruth, GROUNDTRUTH_TYPE
from scorevision.validator.central.private_track.snooker_image_scoring import score_snooker_image


def sample():
    return {"type": GROUNDTRUTH_TYPE, "width": 1920, "height": 1080,
            "balls": [{"label": "red", "bbox": [.4, .5, .02, .03]}],
            "pockets": [{"name": "top_left", "x": .1, "y": .1}],
            "surface": [[.1, .1], [.9, .1], [.9, .9], [.1, .9]]}


def test_round_trip_does_not_parse_as_empty_cricket():
    response = ChallengeResponse(challenge_id="1", prediction=sample(), processing_time=0)
    assert response.is_snooker_image
    assert not response.is_cricket
    assert response.prediction_count == 3
    assert ChallengeResponse.model_validate_json(response.model_dump_json()).is_snooker_image


def test_human_ground_truth_preserves_dense_contours_without_unbounding_miners():
    payload = sample()
    payload["surface"] = payload["surface"]*300
    assert len(SnookerImageGroundTruth(**payload).surface) == 1200
    with pytest.raises(ValidationError):
        SnookerImagePrediction(**payload)


def test_perfect_empty_wrong_dimensions_and_colour():
    truth = SnookerImagePrediction(**sample())
    assert score_snooker_image(truth, truth)[0] == pytest.approx(1)
    assert score_snooker_image(None, truth)[0] == 0
    changed = sample()
    changed["width"] = 1280
    assert score_snooker_image(SnookerImagePrediction(**changed), truth)[0] == 0
    changed = sample()
    changed["balls"][0]["label"] = "pink"
    assert score_snooker_image(SnookerImagePrediction(**changed), truth)[1]["balls"] == 0


def test_duplicate_ball_is_penalised_and_order_is_irrelevant():
    truth = SnookerImagePrediction(**sample())
    changed = sample()
    changed["balls"] *= 2
    score, detail = score_snooker_image(SnookerImagePrediction(**changed), truth)
    assert detail["ball_f1"] == pytest.approx(2/3)
    assert score < 1
    assert detail["ball_tp"] == detail["ball_fp"] == 1
    assert detail["ball_fn"] == 0
    assert detail["ball_count_delta"] == 1


def test_colour_swap_costs_one_false_positive_and_one_false_negative():
    truth = SnookerImagePrediction(**sample())
    changed = sample()
    changed["balls"][0]["label"] = "pink"
    _, detail = score_snooker_image(SnookerImagePrediction(**changed), truth)
    assert detail["ball_fp"] == detail["ball_fn"] == 1
    assert detail["ball_fp_pink"] == detail["ball_fn_red"] == 1
    assert detail["ball_count_delta"] == 0
    assert detail["ball_colour_count_l1"] == 2
    assert detail["ball_exact_colour_counts"] == 0


def test_missed_balls_are_false_negatives_not_false_positives():
    truth = SnookerImagePrediction(**sample())
    changed = sample()
    changed["balls"] = []
    _, detail = score_snooker_image(SnookerImagePrediction(**changed), truth)
    assert detail["ball_fn"] == 1 and detail["ball_fp"] == 0
    assert detail["ball_count_delta"] == -1
    assert score_snooker_image(None, truth)[1]["ball_fn"] == 1


def test_too_many_balls_are_flagged_and_not_silently_clipped():
    truth = SnookerImagePrediction(**sample())
    changed = sample()
    changed["balls"] *= 23
    _, detail = score_snooker_image(SnookerImagePrediction(**changed), truth)
    assert detail["ball_count_limit_violations"] == 2
    assert detail["ball_fp"] == 22
    assert detail["ball_f1"] == pytest.approx(2/24)


def test_missing_visible_pocket_and_extra_unlabelled_pocket():
    truth = SnookerImagePrediction(**sample())
    changed = sample()
    changed["pockets"] = []
    assert score_snooker_image(SnookerImagePrediction(**changed), truth)[1]["pockets"] == 0
    changed = sample()
    changed["pockets"].append({"name": "bottom_left", "x": .2, "y": .8})
    assert score_snooker_image(SnookerImagePrediction(**changed), truth)[1]["pockets"] == 1


@pytest.mark.parametrize("field,value", [("surface", [[0, 0]]), ("width", True), ("height", 10000)])
def test_invalid_geometry(field, value):
    payload = sample()
    payload[field] = value
    with pytest.raises(ValidationError):
        SnookerImagePrediction(**payload)


def test_duplicate_pockets_nan_and_out_of_bounds_rejected():
    for bbox in ([0, 0, float("nan"), .1], [.9, .9, .2, .2], [0, 0, 0, .1]):
        payload = sample()
        payload["balls"][0]["bbox"] = bbox
        with pytest.raises(ValidationError):
            SnookerImagePrediction(**payload)
    payload = sample()
    payload["pockets"] *= 2
    with pytest.raises(ValidationError):
        SnookerImagePrediction(**payload)


def test_wrong_snooker_payload_cannot_fall_back_to_cricket():
    payload = sample()
    payload["balls"][0]["label"] = "banana"
    with pytest.raises(ValidationError):
        ChallengeResponse(challenge_id="1", prediction=payload, processing_time=0)


def test_hidden_gate_rejects_existing_development_data():
    path = Path(__file__).resolve().parents[2]/"scripts/snooker_image_bundle.py"
    spec = importlib.util.spec_from_file_location("bundle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = [{"sha256": "a", "split": "test"}]
    failures = module.hidden_gate(records, {})
    assert "missing_source_match_id" in failures
    assert "never_used_for_training_or_selection" in failures
    assert "records_not_reserved_for_hidden_scoring" in failures


def test_bundle_keeps_labels_out_of_import_and_refuses_hidden(tmp_path):
    import hashlib
    import json
    path = Path(__file__).resolve().parents[2]/"scripts/snooker_image_bundle.py"
    spec = importlib.util.spec_from_file_location("bundle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = tmp_path/"fixture.png"
    image.write_bytes(b"synthetic file fixture")
    row = {"file": image.name, "image": str(image), "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
           "width": 100, "height": 100, "balls": [{"class": "red", "bbox": [10, 10, 2, 2]}],
           "pockets": {"top left": [1, 1]}, "surface": {"polygon": [1, 1, 99, 1, 99, 99]}, "split": "test"}
    with pytest.raises(ValueError, match="Hidden launch gate"):
        module.build_bundle([row], tmp_path/"hidden", "hidden_scoring")
    assert not (tmp_path/"hidden").exists()
    result = module.build_bundle([row], tmp_path/"dev", "development")
    assert not result["launch_ready"]
    tasks = json.loads((tmp_path/"dev/task-import.json").read_text())
    assert "balls" not in tasks[0] and "image" not in tasks[0]
    assert len(list((tmp_path/"dev/validator-only").glob("*.json"))) == 2
