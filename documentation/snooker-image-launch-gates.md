# Snooker still-image task: release boundaries and launch gates

This branch publishes a task contract, validator integration, deterministic
scorer, synthetic tests and a local bundle-building tool. It does **not**
publish a trained model, inference endpoint, training data, private source
ledger or scoring answers. There is no live manifest activation in this change.

Miners supply their own model implementing
[`SNOOKER_IMAGE_MINER_SPEC.md`](../scorevision/miner/private_track/SNOOKER_IMAGE_MINER_SPEC.md).
The generic image starter currently handles TCG and is not a snooker model.
The existing snooker video/table-state v0 contract is unchanged.

## Equal scoring and private answer separation

- Use one frozen contract and scoring implementation for every miner, including
  any miner affiliated with a task developer or operator.
- Hidden scoring images and labels must be disjoint from **all** participating
  model-development data. Developer-held training labels are not a hidden test.
- Keep validator-only answer storage, credentials and access separate from
  every mining runtime. Operators must not use hidden answers to train or tune
  their own mining model. Freeze the model before independent evaluation.
- Publish task semantics and scoring rules; do not require miners to receive
  another participant's private model weights. Follow applicable network rules
  for affiliated participation and operator approval before paid activation.

## Before paid activation

1. Reserve human-reviewed, development-disjoint source matches, maintain rights
   and source provenance, and check exact/perceptual duplicates. The staging
   gate requires at least 240 images from at least 12 verified independent
   matches, with at most 20 images per match. Separately source calibration
   examples. Double-label and reconcile at least 20% before fixing tolerances.
2. Preserve evidence for each assertion in the annotation/access receipt.
   `scripts/snooker_image_bundle.py` rejects a hidden bundle without the
   necessary receipt; a passed local gate is not proof of deployed security.
3. Verify signed image delivery and validator-only ground-truth delivery.
   Miner and anonymous requests for labels, provenance and result storage must
   fail. No private labels, answer URLs or source paths belong in miner payloads.
4. Run a frozen baseline on the independent set. Report per-colour precision,
   recall, false positives and false negatives, count excess/deficit, pocket
   misses and geometry error, mask overlap, latency and uncertainty by match.
5. Review task weights and proposed 30-pixel pocket tolerance against annotation
   agreement and measured results. Do not invent an activation block or baseline.
6. Test backend import and signed miner/validator end-to-end flow in separate
   compatible runtimes. Dependency compatibility and access denial are launch
   gates, not assumptions justified by unit tests alone.
7. Obtain responsible task-operator approval. A code push does not activate a
   task, publish a dataset or authorize payments.

The local bundle builder separates task metadata from `validator-only` answers
and uses owner-only directories. It never uploads images/labels or activates a
task. Development bundles remain explicitly ineligible for hidden scoring.
