# 017 — The product excludes Antarctica, and says so

**Status:** Accepted (2026-08-18, repo owner)

## Context

External review, on `ingest/land_mask.py`:

> The accepted registry ends at 59.45°S, so Antarctica contributes zero cells, while the UTM
> layout also cannot represent land south of 80°S. The campaign can therefore complete while
> omitting Antarctic embeddings. Add polar coverage/storage or explicitly narrow the product
> contract before publishing this as global.

The finding is correct on both mechanisms, and they are independent:

1. **The coverage registry stops at 59.45°S.** The v1.1 mask spans **59.45°S to 83.65°N**, so
   no Antarctic land cell enters the campaign work list at all — nothing is skipped, because
   nothing is ever offered.
2. **The UTM grid could not hold it anyway.** UTM's usable range is 84°N to 80°S, so a cell
   whose centre is beyond it has no representable position on any zone grid. `zone_coverage`
   counts these as `n_clipped` and `build_all` surfaces the count rather than dropping it
   silently — a guard which, against the current registry, correctly reports zero.

So the exclusion is not a defect in the mask or the grid. It is the product's southern edge,
arriving from upstream and reinforced by the projection.

## Decision

**Antarctica is out of scope. The word "global" in this product means the registry's extent —
59.45°S to 83.65°N — and that is now stated rather than implied.**

Extending south would need a different projection for the polar cap (UTM cannot express it), a
coverage source that includes Antarctic land, and a store layout for zones the current scheme
does not define. That is a second product, not a parameter.

## Rejected alternatives

**Add polar coverage and storage**, the reviewer's first option. Rejected on scope: it is a
projection change, a mask change and a layout change, for ground with no agricultural or
commercial demand behind this campaign, at the moment the campaign is being launched.

**Leave it undocumented** — the status quo, and the reason this record exists. A consumer
reading "global embeddings" reasonably expects land coverage to the poles, and the absence
would present as missing data rather than as an excluded region. Nothing in the store would
have said which it was.

**Emit empty Antarctic zone groups to make the exclusion visible in the store.** Rejected: an
all-fill group is indistinguishable from an unfilled one — the same ambiguity `optical_skips`
exists to resolve elsewhere — so it would add bytes and a new question rather than an answer.

## Consequences

- **The southern bound is a published fact.** Any statement of coverage — the README, the
  store's own documentation, an open-data listing — states 59.45°S to 83.65°N rather than
  "global", and this record is the reference for why.
- `n_clipped` stays, and stays surfaced. It is zero against this registry, and it is the guard
  that would report a *future* registry quietly reaching past what UTM can represent.
- A consumer wanting Antarctic embeddings is asking for a different product with a different
  projection, not for a wider run of this one.

## Related

- [ADR 010](010-landmask-registry-coverage.md) — where the coverage registry comes from
- [ADR 008](008-global-store-architecture.md) — the UTM zone layout this bound follows from
