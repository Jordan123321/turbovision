"""Stage image tasks separately from validator-only answers. Never activates a task."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scorevision.utils.snooker_image import ELEMENT_ID, GROUNDTRUTH_TYPE, from_record


def hidden_gate(records, receipt):
    errors = []
    for field in ("human_qa_passed", "never_used_for_training_or_selection", "source_matches_verified",
                  "match_disjoint_from_development", "perceptual_duplicate_review_passed",
                  "validator_only_access_verified", "source_usage_rights_verified"):
        if receipt.get(field) is not True:
            errors.append(field)
    expected = {r["sha256"] for r in records}
    if len(expected) != len(records):
        errors.append("duplicate_image_hashes")
    if set(receipt.get("reviewed_image_sha256", [])) != expected:
        errors.append("receipt_does_not_cover_exact_images")
    matches = {r.get("source_match_id") for r in records}
    if None in matches or "" in matches:
        errors.append("missing_source_match_id")
    if len(matches-{None, ""}) < 12:
        errors.append("fewer_than_12_independent_source_matches")
    if any(count > 20 for count in Counter(r.get("source_match_id") for r in records).values()):
        errors.append("more_than_20_frames_per_match")
    if len(records) < 240:
        errors.append("fewer_than_240_scoring_images")
    if any(r.get("usage") != "hidden_scoring" for r in records):
        errors.append("records_not_reserved_for_hidden_scoring")
    if not receipt.get("review_evidence_refs") or not receipt.get("annotation_revision"):
        errors.append("missing_review_evidence_or_annotation_revision")
    return errors


def build_bundle(records, output, purpose, receipt=None):
    if purpose not in {"development", "hidden_scoring"}:
        raise ValueError("Unsupported purpose")
    failures = hidden_gate(records, receipt or {})
    if purpose == "hidden_scoring" and failures:
        raise ValueError("Hidden launch gate failed: " + ", ".join(failures))
    if not records or len({r["sha256"] for r in records}) != len(records):
        raise ValueError("Records must be nonempty and content-unique")
    prepared = []
    for row in records:
        digest = hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError("Image checksum mismatch: " + row["file"])
        truth = from_record(row)
        if not truth.surface or not truth.pockets:
            raise ValueError("Incomplete human primitive labels: " + row["file"])
        prepared.append((row, truth))
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    private = output / "validator-only"
    private.mkdir(mode=0o700)
    tasks = []
    for row, truth in prepared:
        task_id = row["sha256"][:24]
        tasks.append({"staging_id": task_id, "element_id": ELEMENT_ID,
                      "groundtruth_type": GROUNDTRUTH_TYPE,
                      "asset_key": f"snooker-images/{task_id}.png", "image_sha256": row["sha256"]})
        (private / f"{task_id}.json").write_text(json.dumps({"ground_truth": truth.model_dump(mode="json")}))
    (output / "task-import.json").write_text(json.dumps(tasks, indent=2))
    # Local upload map is private: do not send source names, match IDs or file paths to miners.
    (private / "source-map.json").write_text(json.dumps([
        {"staging_id": row["sha256"][:24], "image": row["image"],
         "source_match_id": row.get("source_match_id"), "split": row.get("split")}
        for row, _ in prepared], indent=2))
    summary = {"element_id": ELEMENT_ID, "purpose": purpose, "images": len(records),
               "launch_ready": False, "hidden_data_eligible": purpose == "hidden_scoring" and not failures,
               "hidden_gate_failures": failures,
               "remaining_launch_checks": ["Backend task import and signed asset delivery",
                  "Operator-reviewed baseline and manifest", "End-to-end validator/miner and access-control smoke test"],
               "activation_performed": False}
    (output / "readiness.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--purpose", required=True, choices=["development", "hidden_scoring"])
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text()) if args.receipt else None
    print(json.dumps(build_bundle(json.loads(args.records.read_text()), args.output, args.purpose, receipt), indent=2))


if __name__ == "__main__":
    main()
