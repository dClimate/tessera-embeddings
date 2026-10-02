# Inference on 2048-px mosaics: what changes, ranked, and how we test it

The mosaics are now stored in 2048-px chunks, the same size as the inference tile. This record
ranks what that lets inference change and sets out how each change is tested on the finished Iowa
mosaic. The experiment branch is `experiment/frisky-inference`, cut from the frozen ingest work
(`experiment/frisky-ingest`, PR #205). How inference works today:
[`../../src/tessera_embeddings/inference/README.md`](../../src/tessera_embeddings/inference/README.md);
the GPU record it rests on: [`inference-on-gpus.md`](inference-on-gpus.md).

## What 2048 changes, measured

**Where the money is.** Graphics cards were $537K of the last campaign's $828K, 65%
([`../campaign/campaign-cost-model.md`](../campaign/campaign-cost-model.md) §12). By hours they
were 57% `g5.2xlarge` (A10G) and 43% `g6e.xlarge` (L40S), and the A10G costs 1.42× as much per
unit of work. The forward pass is 90% of a chunk's wall clock, so a saving comes from paying less
per card-hour or from keeping the card busier; fewer bytes read only helps where reads stall the
card.

**The read is now exact.** Each tile reads one storage chunk per band and date. At 4000² it
decompressed four times the pixels it used, and paid about 13 s of fixed cost per strip read
(`read_plan._EST_FIXED_READ_S`), mostly that re-decompression.

**The GPU bill did not move, and nor did the memory.** The year of Iowa on 2048 mosaics (flow run
`d7e25533`, against `a60550ae` on 4000²):

| | 4000² mosaics | 2048² mosaics |
|---|---|---|
| GPU-hours, peak actors × span | about 34 | 33.5 |
| GPU-idle overhead per chunk, median | about 6 s | 5.7 s |
| GPU utilisation, median over each actor's life | 89–93% | 92% |
| Host RAM peak per actor | about 16.1 GB of 30.9 | 16.1 GB median, 17.2 GB max |
| Strips per tile, median | – | 3 |

The card was already fed, so exact reads bought no GPU time. The memory peak is unchanged because
it is set by budgets, not by the storage chunk: the strip budget (`_S2_STRIP_BYTE_BUDGET`, 5.75
GiB), its pair ceiling and the cross-tile prefetch cap (2 GiB). What 2048 changes is the price of
a strip: a smaller budget now costs a few seconds of read per extra strip rather than 13.

## The options, ranked by impact against complexity

| # | Change | Expected impact | Complexity | Needs 2048 |
|---|---|---|---|---|
| 1 | Fit an actor into 16 GiB of host RAM, so the A10G fallback runs on `g5.xlarge` ($1.006/h) instead of `g5.2xlarge` ($1.212/h) | 17% off every fallback card-hour: about $42K at the last campaign's 206,130 A10G hours, 8% of the card line | low to medium: the budgets, sized per card the way the batch is (`inference-on-gpus.md` §5) | yes: a smaller budget means more strips, which were 13 s each at 4000² |
| 2 | Re-measure the strip planner for 2048, then collapse it if strips are now cheap: the three-regime plan, the starter strip and the cross-tile prefetch tiers reduce to fixed strips with the next one prefetched | small on card time (overhead is 2.5% of it); large on code, most of `read_plan.py` and the cross-tile path in `actors.py` | medium, with a bit-identical gate | yes: the planner trades against a 4000²-era fixed read |
| 3 | Drop the S2 easting crop (`x_sub`) | none on card time: a 2048 chunk is read whole either way, so the crop saves only resident memory | low to medium; 55 references across three modules | yes |
| 4 | Retune the per-card budgets on 32 GiB boxes upward, for fewer strips | at most a few seconds a tile, hidden behind compute already | low | yes |

**Outside this list, for scale.** The largest lever on the card line does not depend on the chunk:
running the encoder's matrix multiplies in FP8 on the L40S, whose dense FP8 ceiling is twice its
BF16 one. It changes the outputs, so it would need the equivalence gate of ADR 012 and an explicit
decision on accuracy before any test. Running L40S only, where capacity allows, is a scheduling
decision rather than a code one.

## Testing plan

**Inputs.** The Iowa year ingested on version C, `s3://arbol-tessera-inputs-dev/_frisky_e2e_c/`
(mosaics, ROI and checkpoint under one prefix), against the full-run baseline `d7e25533` on the
same mosaics and code. Outputs go to `s3://arbol-tessera-embeddings-dev/_frisky_e2e_c/`, with an
`output_name_suffix` per arm.

**One code version per arm, without an image build.** Ray workers pull their source from
`s3://{code_bucket}/code/src{code_suffix}.tar.gz` at start-up, and `code_suffix` is a run
parameter. Each variant is a tarball of its worktree's `src/`, uploaded under its own suffix
(`-inf-<variant>`), so actor-side changes (`read_plan.py`, `data_loading.py`, `actors.py`) need no
deploy. Driver-side changes (scheduling, the flow, the fleet mix) do need one, on a YE dev branch.
A variant's workers then run code whose `inference_code_identity` the driver does not see, so
every arm uses a fresh run id and its staging is deleted afterwards; no staged tile from a
variant is ever reused.

**Arm shape.** About 8 actors each, several arms at once, each with its own Ray cluster and
observed for about 25 minutes (some 40 chunks) before it is cancelled. Every round carries a
control arm on today's code, because card and S3 conditions drift. Arms are held to `g6e.xlarge`,
except where the card is what is under test, and record their card type per chunk.

**What each arm is read on.**
- Per chunk, from `CHUNK_SUMMARY` (`scripts/inference_profile.py` in yield-embeddings): total,
  inference, overhead and prologue seconds, and strips.
- Per actor, from the `RESOURCES:` lines in the flow runner's log stream: host RAM and its peak,
  GPU utilisation, load.
- Chunks are compared by label: every arm draws from the same queue order, so their first chunks
  overlap.
- Card-hours per chunk is the cost figure; GPU-idle overhead is the mechanism.

**Correctness.** None of the options changes what is computed. Option 3 changes only which bytes
are held, so it must be bit-identical. Options 1, 2 and 4 move strip boundaries, which regroups
pixels into different sub-batches; the GPU record found its strip change was not bit-identical for
that reason, so they are held to ADR 012's equivalence thresholds instead. Either way the check is
`te-compare-outputs --labels <common labels>` between the arm's staging and the control's, with
`cleanup_staging` off for both until it has run.

**Rounds.**
1. Four arms at once: the control; the cross-tile prefetch off (`TESSERA_DISABLE_XCHUNK_PREFETCH`,
   set in that arm's tarball), which tests how much of the prefetch machinery still pays at 2048; budgets small enough for 16
   GiB, on `g6e.xlarge`, for its RAM peak and its cost in throughput; and the planner's constants
   re-measured for 2048.
2. The 16 GiB budgets on `g5.xlarge` against today's code on `g5.2xlarge`, the A10G on both: equal
   throughput per card and a RAM peak with margin under 16 GiB pass option 1.
3. Whichever simplification rounds 1 and 2 support (option 2, then 3), against the control.

**Cost.** About $7 an arm (8 L40S for 25 minutes, plus the head node); round 1 about $28, round 2
about $12, round 3 about $14, so about $55 in all.

## Open questions

- Whether `g5.xlarge` capacity is as available as `g5.2xlarge` in us-west-2. Round 2 shows it for
  one run, not for a campaign's fleet.
- Whether the fleet-mix rung list (`providers/aws/fleet_mix.py`) needs a per-card memory budget
  passed to the actor, the way the batch size already is, or one budget small enough for every
  card.
