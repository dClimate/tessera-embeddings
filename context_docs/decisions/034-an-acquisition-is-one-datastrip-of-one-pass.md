# 034 — An acquisition is one datastrip of one pass

**Status:** Accepted 2026-10-20 (Robert). Amends decision 4 of
[ADR 020](020-boa-offset-applies-to-every-valid-dn.md). Evidence in
[`ingest/split-datastrips.md`](../ingest/split-datastrips.md).

## Context

Duplicate selection keeps one copy per acquisition, because `odc.stac.load` fuses every item of a
solar day and two copies of one image would be blended. ADR 020 decision 4 named an acquisition by
its datatake, the satellite pass. A pass that crosses a datastrip boundary reaches a tile as one
granule per datastrip, with the same datatake and baseline and different ground. Named by the
datatake, they competed and the loser's ground was dropped for the day: about 1.2% of clear optical
observations, and up to a fifth of a year where a boundary crosses a tile on most passes.

## Decision

1. **An acquisition is named by the datatake plus the datastrip's sensing start**, read from
   `s2:datatake_id` and `s2:datastrip_id` (`ingest/duplicates.image_identity`), and nothing else.
   Every granule of a split pass is kept. Reprocessing keeps the sensing start, so copies of one datastrip still compete and the
   newest baseline still wins.
2. **Coverage outranks one baseline per pass.** Where a reprocessing omits a datastrip, the older
   copy of it is kept beside the newer ones, so a day can mix baselines across datastrips, never
   within one. Before ADR 021 that mix could mis-correct pixels; with the offset corrected per image
   it costs only a less precise `baselines_applied` and two reads where one would do.
3. **Copies are merged only on proof.** A copy whose pass or datastrip cannot be read is kept as an
   image of its own, with no spares, and the catalogue timestamp is not used to place it. Keeping a
   copy that was in fact a duplicate costs a second read and a missing fallback rung; merging two
   images loses ground. This retires the timestamp clustering and arbitrary attachment that ADR 020
   decision 4 kept as fallbacks.

## Consequences

- The ingest code identity moves, so a part-built mosaic needs `allow_ingest_code_mismatch` to
  resume, and no fill or update should straddle the deploy.
- Every published v1.1 cell was ingested under the old rule, and stays as it is: the fix applies to
  future runs only. The loss mostly shortened pixels' observation sequences rather than leaving them
  empty ([`split-datastrips.md`](../ingest/split-datastrips.md) §7).
- The result no longer depends on how large an area was queried: before, a query too small to
  return one granule of a split pass kept the other one, and a zone-wide query might not.
