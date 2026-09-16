"""Deterministic primitive scoring v1. Hidden labels stay validator-side."""
import math
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from scorevision.utils.snooker_image import SnookerImagePrediction

SCORING_VERSION = 1
POCKET_TOLERANCE_1080P = 30.0


def ball_error_metrics(actual, predicted, matched_pairs):
    """Visible object errors, with no count clipping or true-negative fiction."""
    truth = Counter(b.label for b in actual)
    guesses = Counter(b.label for b in predicted)
    hits = Counter(actual[i].label for i, _ in matched_pairs)
    tp = len(matched_pairs)
    names = ("white", "red", "yellow", "green", "brown", "blue", "pink", "black")
    result = {"ball_tp": tp, "ball_fp": len(predicted)-tp, "ball_fn": len(actual)-tp,
              "ball_precision": tp/len(predicted) if predicted else 0.0,
              "ball_recall": tp/len(actual) if actual else 0.0,
              "ball_count_delta": len(predicted)-len(actual),
              "ball_count_abs_error": abs(len(predicted)-len(actual)),
              "ball_colour_count_l1": sum(abs(guesses[c]-truth[c]) for c in names),
              "ball_exact_colour_counts": int(truth == guesses),
              "ball_count_limit_violations": int(len(predicted) > 22) + int(guesses["red"] > 15)
                  + sum(guesses[c] > 1 for c in names if c != "red")}
    for c in names:
        result.update({f"ball_tp_{c}": hits[c], f"ball_fp_{c}": guesses[c]-hits[c],
                       f"ball_fn_{c}": truth[c]-hits[c]})
    return {k: float(v) for k, v in result.items()}


def box_iou(a, b):
    x, y, w, h = a
    xx, yy, ww, hh = b
    overlap = max(0, min(x+w, xx+ww)-max(x, xx))*max(0, min(y+h, yy+hh)-max(y, yy))
    return overlap/(w*h+ww*hh-overlap)


def mask(polygon, width, height):
    result = Image.new("1", (width, height))
    if polygon:
        # Normalised edge coordinates, sampled on native pixel grid.
        ImageDraw.Draw(result).polygon([(min(width-1, round(x*width)), min(height-1, round(y*height)))
                                        for x, y in polygon], fill=1)
    return np.asarray(result, dtype=bool)


def score_snooker_image(prediction: SnookerImagePrediction | None,
                        ground_truth: SnookerImagePrediction) -> tuple[float, dict[str, float]]:
    zero = {"balls": 0.0, "pockets": 0.0, "surface": 0.0, "ball_f1": 0.0}
    if not ground_truth.surface or not ground_truth.pockets:
        raise ValueError("Scoring ground truth requires a surface and visible pocket labels")
    if prediction is None or (prediction.width, prediction.height) != (ground_truth.width, ground_truth.height):
        return 0.0, {**zero, **ball_error_metrics(ground_truth.balls, prediction.balls if prediction else [], [])}
    actual, predicted = ground_truth.balls, prediction.balls
    credits = np.zeros((len(actual), len(predicted)))
    for i, a in enumerate(actual):
        for j, b in enumerate(predicted):
            if a.label == b.label:
                value = box_iou(a.bbox, b.bbox)
                credits[i, j] = value if value >= .5 else 0
    pairs = list(zip(*linear_sum_assignment(-credits))) if credits.size else []
    matched = sum(credits[i, j] > 0 for i, j in pairs)
    error_metrics = ball_error_metrics(actual, predicted, [(i, j) for i, j in pairs if credits[i, j] > 0])
    denominator = len(actual)+len(predicted)
    balls = 2*sum(credits[i, j] for i, j in pairs)/denominator if denominator else 1.0
    f1 = 2*matched/denominator if denominator else 1.0
    by_name = {p.name: p for p in prediction.pockets}
    pocket_credit = 0.0
    for a in ground_truth.pockets:
        b = by_name.get(a.name)
        if b is not None:
            error = math.hypot((a.x-b.x)*ground_truth.width, (a.y-b.y)*ground_truth.height)*1080/ground_truth.height
            pocket_credit += max(0.0, 1-error/POCKET_TOLERANCE_1080P)
    pockets = pocket_credit/len(ground_truth.pockets)
    actual_mask = mask(ground_truth.surface, ground_truth.width, ground_truth.height)
    predicted_mask = mask(prediction.surface, ground_truth.width, ground_truth.height)
    union = (actual_mask | predicted_mask).sum()
    surface = float((actual_mask & predicted_mask).sum()/union) if union else 0.0
    total = .70*balls + .20*pockets + .10*surface
    return float(total), {"balls": float(balls), "pockets": float(pockets), "surface": surface,
                          "ball_f1": float(f1), **error_metrics}
