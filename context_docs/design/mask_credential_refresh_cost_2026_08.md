# What the per-read mask credential refresh costs

Measurement record, 2026-08-05, `global-tessera-dev`. Checks whether resolving the ROI
mask's S3 credentials per read — instead of freezing them once per leg — introduced a
throughput regression. The acceptance bar set for this change was **no worse than 4–5%**.

Companion to `ingest_read_failure_causes_2026_08.md`, which records why the change was made.

## Verdict

**No regression at the bar. The measured upper bound is 1.33%, and the realistic figure is
below that.**

Two components, measured separately because they have very different sizes.

| component | cost | how it was measured |
|---|---|---|
| resolving the credential | **3.8 µs per read** | 200 calls to `iam_s3_storage_options`, wall clock |
| rebuilding the mask graph per date | **~340 ms per date** | 8 timed `read_roi_mask` calls against 37N's mask, warmed first |

The credential resolution is irrelevant at 3.8 microseconds: `_resolve_iam_credentials`
caches a live refreshable credential, so each call is a frozen copy of an already-resolved
one and touches no network. That was the part under suspicion and it is not the cost.

The cost that exists at all is the **per-date mask graph rebuild**, which the same change
introduced on the optical path so that each date resolves its own credential.

## Sizing the graph rebuild against real work

Per-date totals from the live fleet, taken from the `Stage timings` / `Batch timings` lines
over a five-hour window (127 dates across three concurrently-ingesting zones), as medians:

| zone | dates | build | gate | write | **total per date** | 340 ms is |
|---|---:|---:|---:|---:|---:|---:|
| 31S | 20 | 5.0 s | 7.2 s | 24.1 s | **37.2 s** | 0.91% |
| 37N | 59 | 18.4 s | 17.9 s | 263.4 s | **303.1 s** | 0.11% |
| 41S | 48 | 3.5 s | 4.3 s | 17.1 s | **25.5 s** | **1.33%** |

**The 1.33% is a deliberate overestimate**, and the reason is worth keeping: the 340 ms
figure comes from **37N's** mask — 933,888 × 67,584 px, 3,876 graph tasks, the largest zone
in the campaign — while it is being divided into **41S's** per-date total, one of the
smallest. A small zone's mask has far fewer chunks and builds proportionally faster, so its
true added fraction is smaller than the table shows. The bound holds without needing that
extra measurement, which is why it was not taken.

## What this does and does not cover

**Covered, by measurement:** the client-side credential resolution, and the client-side
graph construction that the per-date rebuild adds. Both land in the `build` term of the
per-date timing line, which is why that line is the right instrument.

**Covered by reasoning, not measurement — and flagged as such:** the volume of S3 reads of
the mask is unchanged. The mask was never persisted (the surrounding comment says so
explicitly: "No persist — ... a per-date re-read beats pinning the grid"), and dask does not
share results across separate `compute()` calls, so each date already re-read the mask
chunks it needed. Reusing one graph object across dates saved graph *construction*, not
reads — and that saving is exactly the 340 ms now paid per date.

**Not measured:** a same-zone before-and-after on the live fleet. It was considered and
rejected as an instrument rather than skipped for convenience:

- Ingest is idempotent, so re-running a zone-year writes nothing and times nothing.
- Cross-zone comparison cannot resolve a 4–5% effect. **31S and 41S both hold six live
  tiles and differ by 64% in cost per window** (4.95 vs 8.10 s/window) — the between-zone
  spread is an order of magnitude larger than the effect being looked for.
- Splitting one zone-year across the deployment boundary compares different dates, and
  per-date cost varies with season and with windows-per-date (see
  `normalise-throughput-before-comparing`).

Isolating the changed code and dividing by measured real work answers the question at a
resolution the fleet cannot provide, which is why it was done that way.

## The radar path

Unaffected in frequency. `s1_roi` reads the mask once per **batch**, not per date, and that
cadence did not change; only the resolution moved from frozen to per-call, which is the
3.8 µs above. `live_windows_for_mask` runs once at leg entry.

## Method notes

- Both arms were warmed with one discarded call before timing, so neither paid for a cold
  connection pool.
- In a direct A/B of the two credential shapes on the same mask, the provider arm measured
  **10% faster** than the frozen arm. That is network variance, not a speedup — the arms
  differ by 3.8 µs on a 340 ms operation — and it is recorded here as a caution: at this
  scale run-to-run noise on an S3 metadata read exceeds the effect entirely, so a single
  paired comparison of whole reads cannot answer the question either.
