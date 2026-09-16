# Snooker still-image primitives v1

Proposed element: `manako/DetectSnookerImagePrimitivesV1`.
Ground-truth type / prediction discriminator: `snooker_image_primitives_v1`.
This is a new contract, not a change to `DetectSnookerBallState` video v0.
Status: implementation and dry-run preparation; **not an activated paid task**.

Each request contains `challenge_id` and one `image_url`. No video, target-frame
list, match provenance, labels or human-review notes are sent to the miner.
Images retain their original orientation and dimensions, at most 4096 per side
and 8,388,608 pixels. Deployments serve one element; do not infer sport solely
from the fact that the request contains an image (TCG also uses images).

Response example (synthetic geometry, not private ground truth):

```json
{
  "challenge_id": "123",
  "prediction": {
    "type": "snooker_image_primitives_v1",
    "width": 1920,
    "height": 1080,
    "balls": [{"label": "red", "bbox": [0.4, 0.5, 0.02, 0.03]}],
    "pockets": [{"name": "top_left", "x": 0.1, "y": 0.1}],
    "surface": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
  },
  "processing_time": 0.25
}
```

All geometry uses normalised **image-edge** coordinates. Origin is top-left;
x increases right, y down. `bbox` is x/y/width/height, not xyxy. Ball labels:
white (cue ball), red, yellow, green, brown, blue, pink, black. Balls have no
potted/occluded state or track ID. The centre is derived from the box.
At most 64 boxes are accepted; duplicates cost score, including duplicate reds.

Pocket names are screen-relative: top_left, top_right, middle_left,
middle_right, bottom_left, bottom_right. Each name appears at most once.
Only visible, human-labelled pockets are scored. A missing predicted visible
pocket earns zero. Unlabelled/out-of-crop pockets are not scored, since an
omitted human label must not be invented as a known location. Return empty
lists for missed objects, not NaN or fabricated coordinates.

Miner surface is one ordered polygon with 3–256 vertices, or an empty list for a
miss. It describes playable cloth, not cushion rails. Scores use Pillow's
native-resolution polygon rasterisation with rounded pixel coordinates.
Private human contours allow up to 8192 vertices and are preserved in full;
the miner response bound must not truncate scoring ground truth.

## Deterministic score

- Balls: same-colour Hungarian assignment, IoU >= 0.5. Credit is matched IoU;
  score is `2 * sum(credit) / (truth_count + predicted_count)`. Missing and extra
  balls both reduce credit. Conventional class-aware F1 is also reported.
- Pockets: mean `max(0, 1 - error_1080p_pixels / 30)` over visible GT pockets.
- Surface: raster mask intersection-over-union.
- Total: 70% balls, 20% pockets, 10% surface. Missing/wrong-type predictions
  or incorrect image dimensions score zero. Invalid payloads fail validation.

The scorer and tolerances must be frozen before unseen baseline evaluation.
The 30-pixel tolerance is a proposed v1 contract value, not a measured human
agreement threshold; operator review and a double-labelled sample are required
before launch. Confidence thresholds are the miner's responsibility.

### False positives, misses and colour/count diagnostics

Scorer details include `ball_tp`, `ball_fp`, `ball_fn`, precision, recall,
signed `ball_count_delta` (predicted minus visible truth), absolute count error,
per-colour count L1 error and exact-colour-count agreement. Per-colour counts
are exposed as `ball_tp_red`, `ball_fp_red`, `ball_fn_red`, etc.

An extra/duplicate detection is a false positive (type I); a missing labelled
ball is a false negative (type II). A wrong-colour detection counts as one FP
for the predicted colour and one FN for the true colour. These are object
matching errors at IoU >= 0.5, not a colour-only classifier's confusion matrix.
There is no meaningful true-negative box population, so a false-positive rate
`FP/(FP+TN)` is not reported. Use precision and recall with raw counts.

`ball_count_limit_violations` flags each breached standard-snooker constraint:
at most 22 total, at most 15 reds and at most one of every other colour including
white. Predictions are **not** truncated or corrected before scoring; even an
extra red below the legal maximum still costs a false positive against truth.
Fewer than 22 visible balls is not an error by itself. Simultaneous extras and
misses can cancel in total count, so count accuracy alone is insufficient.

These diagnostics do not alter the v1 score weights or reveal private per-image
answers to miners. They belong in validator-private audit results; publish
aggregate metrics only after a deliberate disclosure review.

No trained model, private training labels or hidden scoring answers are included
in this repository. See [launch gates](../../../documentation/snooker-image-launch-gates.md)
for independent evaluation and separation of mining from validator access.

Backend must return the same primitive object under `ground_truth` from the
signed validator-only `/api/tasks/{id}/ground-truth` endpoint. The source
storage object, hidden answers and match IDs must never be public assets or
public score-shard fields. Existing signed requests, response-size limits and
private result storage remain in force. No live backend is configured here.

The generic starter miner's image path currently serves TCG. Snooker miners
must explicitly implement this contract and select their snooker predictor;
the starter must not be advertised as a functioning snooker model.
