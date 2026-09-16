from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

import httpx
import pytest

from scorevision.utils.schemas import ChallengeResponse
from scorevision.utils.snooker_image import SnookerImagePrediction, GROUNDTRUTH_TYPE, ELEMENT_ID
from scorevision.validator.central.private_track.challenges import Challenge, fetch_ground_truth, get_challenge_with_ground_truth
from scorevision.validator.central.private_track.miners import ChallengeAttempt
from scorevision.validator.central.private_track.registry import RegisteredMiner
from scorevision.validator.central.private_track.runner import _challenge_miner, _strip_for_public_shard

MODULE = "scorevision.validator.central.private_track.challenges"


def primitives():
    return SnookerImagePrediction(type=GROUNDTRUTH_TYPE, width=1280, height=720,
        balls=[{"label": "red", "bbox": [.4, .5, .02, .03]}],
        pockets=[{"name": "top_left", "x": .1, "y": .1}],
        surface=[[.1, .1], [.9, .1], [.9, .9], [.1, .9]])


@pytest.mark.asyncio
async def test_image_asset_is_not_routed_as_video():
    with patch(f"{MODULE}.fetch_next_challenge", new=AsyncMock(return_value={
            "task_id": "123", "asset_url": "https://example.com/test.png"})), \
         patch(f"{MODULE}.fetch_ground_truth", new=AsyncMock(return_value=primitives())), \
         patch(f"{MODULE}.complete_task_assignment", new=AsyncMock()):
        result = await get_challenge_with_ground_truth("manifest", ELEMENT_ID, None, GROUNDTRUTH_TYPE, max_retries=1)
    assert result.image_url.endswith("test.png")
    assert result.video_url is None


@pytest.mark.asyncio
async def test_video_cannot_silently_become_image_task():
    with patch(f"{MODULE}.fetch_next_challenge", new=AsyncMock(return_value={
            "task_id": "123", "video_url": "https://example.com/test.mp4"})), \
         patch(f"{MODULE}.fetch_ground_truth", new=AsyncMock()) as fetch:
        assert await get_challenge_with_ground_truth("manifest", ELEMENT_ID, None, GROUNDTRUTH_TYPE, max_retries=1) is None
        fetch.assert_not_called()


@pytest.mark.asyncio
async def test_signed_gt_adapter_parses_exact_primitive_contract():
    response = httpx.Response(200, json={"ground_truth": primitives().model_dump(mode="json")},
                             request=httpx.Request("GET", "https://validator.invalid/"))
    client = AsyncMock()
    client.get.return_value = response
    with patch(f"{MODULE}.get_settings", return_value=SimpleNamespace(PRIVATE_GT_API_URL="https://validator.invalid", SCOREVISION_API="")), \
         patch(f"{MODULE}.build_validator_query_params", return_value={"signature": "test"}), \
         patch(f"{MODULE}.httpx.AsyncClient") as factory:
        factory.return_value.__aenter__.return_value = client
        result = await fetch_ground_truth("123", None, ELEMENT_ID, GROUNDTRUTH_TYPE)
        assert result.model_dump() == primitives().model_dump()
        assert client.get.call_args.kwargs["params"]["element_id"] == ELEMENT_ID


@pytest.mark.asyncio
async def test_runner_scores_snooker_and_public_shard_omits_answers():
    truth = primitives()
    challenge = Challenge("123", truth, GROUNDTRUTH_TYPE, image_url="https://example.com/test.png")
    response = ChallengeResponse(challenge_id="123", prediction=truth, processing_time=.2)
    miner = RegisteredMiner(7, "test_hotkey", "127.0.0.1", 8000, "test/repo", "test", "sha256:test", 10)
    result, predictions, benchmark = await _challenge_miner(miner, challenge, None, 30, 100, ELEMENT_ID,
        None, "sha256:test", attempt=ChallengeAttempt(response, .3, False))
    assert result["score"] == pytest.approx(1)
    assert result["prediction_count"] == 3
    assert benchmark is None
    assert predictions[0]["type"] == GROUNDTRUTH_TYPE
    public = _strip_for_public_shard({**result, "ground_truth": truth.model_dump(), "source_match_id": "private"})
    assert "ground_truth" not in public
    assert "score_breakdown" not in public
    assert "video_url" not in public
    assert "source_match_id" not in public
    assert public["groundtruth_type"] == GROUNDTRUTH_TYPE
