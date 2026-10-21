# 025 — A library upgrade ships on the cross-config envelope and measured usability, at a store boundary

**Status:** Accepted (2026-10-03, repo owner). v1.1 kept on PyTorch 2.5.1 (2026-10-21, repo owner).

## Context

The GPU workers ran PyTorch 2.5.1 with CUDA 12.1 while the lockfile, the driver image and every
test ran 2.12. Nobody chose 2.5.1: the Ray AMI installed "the newest CUDA 12.1 build", and that
index stopped publishing at 2.5.1. Moving the workers to 2.14.1 (CUDA 13.0) raised the question
[ADR 012](012-validated-equivalence-for-inference-outputs.md) answers only in passing. Its fact 2
says a torch, cuBLAS or driver upgrade changes int8 outputs. It does not say which gate an upgrade
must meet, or when one may land in a store that already holds output from the old stack.

The upgrade was measured with the code, tiles, tiling and batch held fixed and only the workers'
PyTorch differing ([`../inference/inference-on-gpus.md`](../inference/inference-on-gpus.md) §8),
twice: first on code that still ran v1.1's `nn.GRU`, then on the code that ships, after #212 replaced
that GRU with a compiled step ([ADR 026](026-v1-1-runs-the-gru-it-was-trained-with.md)). On the code
that ships:

- **The same-config gate fails on every tile, for both models.** v2 keeps 99.96–99.99% of int8
  values exactly equal and v1.1 88.7–90.0%; both reach a largest difference of 3 levels and per-pixel
  scales up to 1.5% apart.
- **The cross-config envelope passes** on every tile of both models.
- **What a user of v2 consumes barely moves.** 99.99% of top-20 neighbours are kept, against the
  0.9940 that two same-model stores agree on, and 99.999% of pixels stay in the same k-means group.
- **v1.1 moves as far as a change of card.** It keeps 98.2% of top-20 neighbours on the Iowa tiles
  and 91.9% on the more uniform `m10` tile, below that ceiling; on the `nn.GRU` code it kept 99.48%.
  On the same Iowa tiles an A10G and an L40S running identical code differ by as much. The likely
  cause is the compiled GRU step: its code generator changed between the stacks, and the recurrence
  carries the difference through up to 256 steps.

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
2. **An upgrade lands at a store boundary, never inside a store.** Stores started after the switch use
   the new stack. A store begun on the old stack is finished on it, including every year, area or
   update later appended to it, such as a new year in the global store: the worker image is a per-run
   flow parameter (`ami_ssm_name`), so an append can name the old image. The staging fingerprint does
   not include the image either, so a run on a different stack from the one that staged its tiles
   passes a fresh `force_staging_restage` token rather than reusing them.
3. **v2 moves to PyTorch 2.14.1; v1.1 stays on 2.5.1.** v2 clears the gate and v1.1 does not. Every
   v1.1 run names the 2.5.1 worker image in its `ami_ssm_name`: the published store's updates and
   any new v1.1 store alike. Nothing planned needs v1.1 on the new stack: its one store must finish on
   2.5.1 under rule 2, and v1.1 is expected to give way to v2 (ADR 026).
4. **The workers run the lockfile's torch version,** as that version's CUDA build, with v1.1's runs
   as the one exception. Tests, the driver image and v2's workers then run one version, and moving it
   is a lockfile change.

## Rejected alternatives

- **The same-config gate for library upgrades.** It would forbid every upgrade indefinitely, and it
  holds the upgrade to a standard the current stack fails against itself across queue orders.
- **Bit-exactness.** Never a property of this pipeline across batch shapes, cards or libraries
  (ADR 012, fact 2).
- **Installing "the newest build" from a CUDA index.** That is how the workers came to sit on a
  version nobody chose, three releases behind the one the tests ran.
- **Moving v1.1 too, judged against a change of card instead of the same-model ceiling,** as ADR 030
  (PR #241) judges reorderings of our own arithmetic. v1.1 would pass on the Iowa tiles, but nothing needs it on the new stack, the `m10` tile
  has no card comparison, and it would loosen the bar for the one model that measurably moves.

## Consequences

- A store's output is consistent within one library stack. **The stack is not recorded in store
  metadata,** because the code identity hashes our source and not our dependencies. The campaign
  record carries each switch's date and worker image instead.
- A switch must be scheduled between fills. Merging the AMI change while a fill or an append is
  running would put two stacks into one store.
- **Two worker images stay in service.** The 2.5.1 image must stay addressable for as long as v1.1
  runs, and v1.1 cannot use Blackwell cards (`g7`, `g7e`), for which 2.5.1 has no kernels.
- Every future upgrade repeats the measurement in `inference-on-gpus.md` §8, on the code that will
  ship, on both models and on every card the fleet may use. Speed and card memory are part of it,
  because they can move in either direction.
