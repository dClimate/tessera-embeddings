# 018 — Refuse pixels below 15 optical observations, and accept the deferral premium

**Status:** Accepted (line decided 2026-08-17; recorded here 2026-08-18). Built.

## Context

A yearly embedding is meant to describe a year. A pixel with very few usable optical
observations produces one anyway, and it is indistinguishable at read time from a well-observed
one unless the consumer inspects `s2_obs_count` themselves. The question was whether to publish
those pixels with a quality flag, or to refuse them.

`OPTICAL_MIN_OBS = 15`, applied per pixel per year, strictly fewer than 15 being refused.

## Decision

**The line is 15, and coverage was chosen over reproducibility knowingly.** It keeps 94% of
pixels against 79% at 25, pixel-weighted over 40 cells, and the accepted cost is that two
independent embeddings of the same ground agree less well.

**15 is not the reproducibility elbow, and that is the decision rather than an oversight.** The
elbow is at 25: over 741 blocks of doubly-embedded ground, agreement runs three to four times the
pipeline's own noise floor everywhere below 25 and halves crossing into 25–30. A line at 15 admits
the worst-reproducing band in the measurement — 15–19 sits at 3.83× the floor, worse even than
under-15 at 2.21×.

Two things weigh the other way, and both post-date the elbow.

**Refusal is irreversible while filtering is not.** `s2_obs_count` is published per pixel, so any
consumer can apply a stricter line at read time, and nobody can recover a pixel we refused.

**Count is not a reliable proxy for temporal spread, and it is weakest exactly at the line.** What
plausibly matters is whether the year is *covered*, not how many observations there are. Measured
over 919 million pixels across 24 cells, **33.7% of pixels at 15–19 observations already cover ten
or more months.** So some of what the elbow would refuse does describe a year. The same measurement
found the rule blind in the other direction too: the 40+ band is worse-distributed than 30–39 (11.1
months against 11.5), so a count line admits deep pixels that are blind for a season.

Neither makes 15 *better* than 25 on the evidence. They are why a coverage-first reading is
defensible, and the decision is that not excluding useful data outranks reproducibility here.

## Consequences, and the one that is a real cost

**The rule is part of the store's write-once root identity**, so moving the line means a new store
rather than a migration. A cell filled under a different line than its neighbours would be
undetectable afterwards: a refused pixel is indistinguishable from one that had no optical input.

**The deferred spend is a deferral with a premium, not a saving.** Refused pixels are the thin ones
and cost is token-denominated, so roughly 18% of pixels carry closer to 9% of tokens — order $50K of
a ~$573K inference spend. Re-running a shard later costs *more* than embedding it once now: a new
fleet, re-ingested mosaics, and the healthy pixels of every selected shard paid for twice. The
ruling (2026-08-13) is that the campaign expects budget left over and the premium is affordable.
**Recorded so nobody presents the $50K as a net gain.**

**A top-up is a new cycle, not a re-tag**, and its unit is a shard rather than a pixel — rewriting
any refused pixel rewrites its whole shard. The work list is shards whose OWN mean is below the
line, not every shard containing a refused pixel, which is what makes a later pass affordable. The
registry's per-shard `s2_obs_mean` is the selection column.

## Rejected alternatives

**Publish everything with a quality flag.** The reversible option, and the counter-argument was put
directly: `s2_obs_count` is already per-pixel, so a determined user could always filter at 25
themselves, and a flag can be tightened later while a refusal cannot be loosened without a re-run.

Two objections were raised on 2026-08-13 and answered. That the branch being taken is the
irreversible one while the reversible one is free. And that a hole is not obviously kinder to a
careless user than a noisy value — a thin embedding degrades a downstream model slightly, whereas a
chunk-shaped hole changes the geometry of a study area and the length of a time series, and for
yield modelling a NaN is not the gentler failure.

**The ruling stands**: the dataset's reputation is what is being protected, and erring toward high
quality serves that even where it costs a consumer convenience. This is a judgement about
reputation rather than a measurement, and it is recorded as one because those are the objections a
reader will raise.

**A line at 30, then 25.** 30 rested on how the pictures looked; blind re-measurement showed
legibility was tracking rendered contrast rather than information, and spatial organisation is flat
from 12 to 45 observations. 25 was then an explicit placeholder so the machinery could be built
while evidence was gathered. The number moved three times and the record of why matters more than
the number.

**A spread rule instead of a count rule.** Attractive on the evidence above and not supportable
yet: nothing has shown that better-spread pixels produce better embeddings. The measurement that
would settle it — zone-overlap reproducibility re-stratified by spread at fixed count — is runnable
on adjacent readable pairs, and is not a prerequisite for this line.

## Related

- [`../inference/minimum-optical-depth.md`](../inference/minimum-optical-depth.md) — the evidence
  and the build record
- [ADR 008](008-global-store-architecture.md) — the write-once root identity this rule joins
