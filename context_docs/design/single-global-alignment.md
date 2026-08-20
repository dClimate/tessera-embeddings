# Aligning the single-ROI pipeline with the global campaign

**Dated 2026-08-03.** Why the single-ROI and campaign paths were allowed to diverge,
what the divergence cost, and what remains unaligned. The current behaviour is
documented in the code and READMEs; this file is the reasoning behind the change and
the record of what was deliberately left alone.

---

## Why the two paths were allowed to diverge, and what it cost

The single-ROI and campaign paths ran different geometry for a period: the campaign moved to a 2048
inference tile and a shared sharded layout while single-ROI stayed as it was. **Aligning them was the
right call and the cost of the divergence was the reason.**

**What the divergence cost:** every change to the write path had to be reasoned about twice, and the
two paths' guards drifted — the campaign gained a full coordinate-vector check while single-ROI kept
an endpoint-only one, which is a defect that survived until 2026-08-18 precisely because nobody was
comparing them.

**What was deliberately left alone:** the single-ROI store's time axis is the window-end label and
extendable, where the campaign's is a fixed calendar-year axis declared at seeding. Those are
different products rather than a divergence to fix, and ADR-011 covers the windowed-variant design
if that ever changes.

The current behaviour is in the code and the READMEs. **The load-bearing warning:** the single-ROI
path has no flow-level tests, so a green suite is weak evidence about it — the whole geometry change
broke nothing there because nothing there was exercised.
