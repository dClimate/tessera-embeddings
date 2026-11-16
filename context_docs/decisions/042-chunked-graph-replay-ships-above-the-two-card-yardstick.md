# 042 — v2's chunked graph replay ships although it moves outputs further than a card change

**Status:** Accepted by merging PR #273 (ruled 2026-11-13)

## Context

[ADR 030](030-forward-pass-reorderings-ship-at-a-store-boundary.md) gates a forward-pass reordering on
ADR 012's cross-config envelope plus usability, and sets the usability bar at what two GPU models
running identical code already do to each other.

v2's forward now runs each sub-batch in small chunks that replay a captured CUDA graph
([`../inference/v2-chunked-graph-replay.md`](../inference/v2-chunked-graph-replay.md)). The formula is
unchanged. A 64-pixel matrix product takes a different path through the GPU's linear algebra library
than a 7,168-pixel one, so the sums are ordered differently. On an L40S the change takes about 9% off
inference time and 7% off tile time.

It passes the cross-config envelope on every tile measured. It does not meet ADR 030's usability bar.
Same script and sampling. The chunked figures are from Iowa tiles on one L40S; the two-card figures
are ADR 030's, an RTX PRO 6000 against an L40S:

| | 64-pixel chunks vs whole batch | two cards, same code (ADR 030's bar) |
|---|---:|---:|
| int8 values exactly equal | 93.8–95.0% | 97.2–97.5% |
| largest int8 difference | 3 levels | 3 levels |
| largest scale drift | 1.55% | 1.38% |
| top-20 neighbours kept | 0.974 | 0.981 |
| same k-means cluster | 99.76% | 99.85% |
| mean per-pixel cosine distance | 2.1e-5 | 1.1e-5 |

The mean shift is about one hundredth of the median distance between adjacent pixels. The chunked
forward is bit-identical to itself across processes on the same card.

## Decision

1. **Chunked graph replay ships for v2.** A shift of about twice the two-card yardstick is accepted
   for the GPU-hours it saves.
2. **ADR 030's bar stands for other changes.** This is one measured exception, not a new bar: the
   next reordering is compared with the two-card figures, and exceeding them needs its own ruling.
3. **It lands at a store boundary**, as ADR 030 §2 requires.

## Consequences

- A v2 store started after this change differs from one started before it by the figures above. A
  store begun before it is finished by the code that began it.
- The chunk size is per card, so a store filled by a mixed fleet holds outputs from more than one
  chunk size, on top of the card-to-card difference it already holds.
- v1.1 is untouched: only the v2 builder sets a chunk size.

## Rejected alternatives

- **Holding the change to ADR 030's bar.** It would forgo the saving for a difference a hundred times
  smaller than the distance between neighbouring pixels.
- **80-pixel chunks.** Level with 64 on real tiles and further from the whole batch (top-20
  neighbours 0.968 on one tile, against 0.974).
