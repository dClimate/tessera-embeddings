# 024 — A library upgrade ships on the cross-config envelope and measured usability, at a store boundary

**Status:** Accepted (2026-10-03, repo owner)

## Context

The GPU workers ran PyTorch 2.5.1 with CUDA 12.1 while the lockfile, the driver image and every
test ran 2.12. Nobody chose 2.5.1: the Ray AMI installed "the newest CUDA 12.1 build", and that
index stopped publishing at 2.5.1. Moving the workers to 2.14.1 (CUDA 13.0) raised the question
[ADR 012](012-validated-equivalence-for-inference-outputs.md) answers only in passing. Its fact 2
says a torch, cuBLAS or driver upgrade changes int8 outputs. It does not say which gate an upgrade
must meet, or when one may land in a store that already holds output from the old stack.

The upgrade was measured three ways, with the code, tiles, tiling and batch held fixed and only the
workers' PyTorch differing
([`../inference/inference-on-gpus.md`](../inference/inference-on-gpus.md) §8):

- **The same-config gate fails on every tile,** for both models: 98.4–99.4% of v1.1's int8 values
  exactly equal, a largest difference of 3 levels, and per-pixel scales up to 1.5% apart. v2 is about
  ten times closer and fails on the same two metrics.
- **The cross-config envelope passes** on all 25 production-path tile pairs.
- **What a user consumes barely moves.** Top-20 neighbours are 99.48% the same, against the 0.9940
  that two same-model stores agree on. 99.97% of pixels stay in the same k-means cluster. The mean
  per-pixel cosine distance is about 2,000 times smaller than the distance between adjacent pixels.

One further measurement removed the case for holding upgrades to the same-config gate. On today's
stack, a tile's output depends on queue order: a single-strip tile that arrives behind a prefetching
predecessor is split into a starter strip and a body, which regroups its sub-batches. That alone
moved one tile by up to 3 levels and 1.5% in scale. **`main` does not meet the same-config gate
against itself across two runs.**

## Decision

1. **A change confined to the library stack is gated by ADR 012's cross-config envelope, plus
   measured usability.** The library stack means PyTorch, CUDA, cuBLAS, cuDNN and the driver, with no
   change to our arithmetic. The envelope is judged with `te-compare-outputs --cross-config` on
   production-path tiles from both models. Usability means top-20 neighbour agreement at or above the
   same-model ceiling of [`../inference/validating-a-model-change.md`](../inference/validating-a-model-change.md)
   (0.9940), measured on the same tiles. Footprint and observation-count layers stay exact.
2. **An upgrade lands at a store boundary, never inside a store.** Areas and years started after the
   switch use the new stack. A store begun on the old stack is finished on it: the worker image is a
   per-run flow parameter (`ami_ssm_name`), so an append can name the old image.
3. **The workers run the lockfile's torch version,** as that version's CUDA build. Tests, the driver
   image and the workers then run one version, and moving it is a lockfile change.

## Rejected alternatives

- **The same-config gate for library upgrades.** It would forbid every upgrade indefinitely, and it
  holds the upgrade to a standard the current stack fails against itself across queue orders.
- **Bit-exactness.** Never a property of this pipeline across batch shapes, cards or libraries
  (ADR 012, fact 2).
- **Installing "the newest build" from a CUDA index.** That is how the workers came to sit on a
  version nobody chose, three releases behind the one the tests ran.

## Consequences

- A store's output is consistent within one library stack. **The stack is not recorded in store
  metadata,** because the code identity hashes our source and not our dependencies. The campaign
  record carries each switch's date and worker image instead.
- A switch must be scheduled between fills. Merging the AMI change while a fill or an append is
  running would put two stacks into one store.
- Every future upgrade repeats the measurement in `inference-on-gpus.md` §8, on both models and on
  every card the fleet may use. Speed and card memory are part of it, because they can move in
  either direction: 2.14.1 is about 6% faster per tile on the L40S but 10% slower on the A10G's
  deepest bucket.
