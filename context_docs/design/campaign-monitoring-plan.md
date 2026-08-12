# Monitoring the campaign — what we watch, and the three decisions it feeds

The campaign is **not built to be stopped.** Once it is running there is almost no surgical
intervention available: per-zone and per-chunk retries are in the code because they have to be,
and the only hand tool is editing Zarr attrs or a manifest. Everything else is either a side run
or a second wave at the end.

So monitoring is not a dashboard. It exists to produce **one of three decisions** as early as
possible:

| | decision | when |
|---|---|---|
| **1** | **Stop everything, fix, restart** | the defect taints work the campaign has not done yet, so every hour it keeps running is wasted money |
| **2** | **Let it run; fix in a second wave** | the problem is confined to named zones and does not touch the rest |
| **3** | **Ignore; maybe patch attrs later** | a property of the input or of an older image, not a defect |

Class 1 is only affordable in the **first hours**. Class 2 is the common outcome and its whole
product is a **list of zones** to re-run at the end. Class 3 is most of what monitoring will see,
and the design problem is stopping it from burying the other two.

Two things follow, and they shape everything below. **The cost of a missed class-1 signal is the
whole remaining campaign**, so those checks run on the fastest cadence and page. And **the cost of a
noisy class-3 signal is that nobody reads the channel**, so a standing finding must be said once.

---

## 1. The mechanism

**One flow, `campaign-watch`, on a 5-minute cron. Stateless per round; its memory lives in S3.**

It is a thin wrapper around `campaign_health.py`, which already is the detector: seventeen checks in
a fixed order, a verdict at the top, and `--json` giving every check a stable `slug`, structured
`subjects`, a drill-down command, and a **`fingerprint` over `(slug, status, subjects)`** that tells
a new finding from one already seen. Nothing about the detection is new here. What is new is that
something runs it unattended, keeps the fingerprints, and pushes.

**Why a scheduled round rather than a long-lived watcher.** The dedup state has to be durable
anyway — a watcher that restarts and re-posts every standing warning is the failure mode that makes
a channel useless — and once the state is in S3, a long-lived process buys only a saved cold start
while costing a single point of failure and a restart to pick up a code change. A cron round is also
its own liveness signal: the absence of runs *is* the alarm, and no second mechanism is needed to
watch the watcher.

**Two cadences, because the cost is in the log queries.** Measured: one 60-minute Logs Insights
query scans ~2.2 GB at 36-cell width, ~$0.011, and `campaign_health` issues five of them —
**$0.055 per full round**. At 5 minutes that is ~$16/day, and 12 consecutive rounds would re-measure
the same 60-minute window. So:

* **every round (5 min) — the free checks.** Placement, fleet widths, sweep success, concurrency
  limits, **crashed/failed runs**, cost accrual, **cell validation**. No Insights query, so the
  round is essentially free; and this set carries the two most decision-relevant signals we have.
* **every third round (15 min) — the paid checks.** Commit rates, per-cell roster silence, fill
  scope, fleet duty cycle, GPU starvation. This keeps the measured **~$5/day** cost model, which is
  a rounding error against one idle 20-worker fleet-hour (~$37).

**What each round publishes**, all under `s3://global-tessera-embeddings/monitoring/`:

| artifact | what it is for |
|---|---|
| `rounds/<timestamp>.json` | the full `campaign_health --json` report, verbatim. The audit trail, and what the agent reads |
| `state.json` | fingerprint → `first_seen`, `last_posted`, `times_seen`. The dedup and ageing memory |
| `shelf.json` | the class-2 list: per zone, what is wrong, the evidence, and whether it is resolved (§4) |
| `latest.json` | the newest round, for a cheap read |

**A round never mutates the campaign.** No cancelling, no reopening, no dispatching a refill. The
one write it makes is its own record. A monitor that can act is a monitor that can act on a false
positive, and a refill costs hours and real money.

---

## 2. Slack — `#alerts-global-tessera`

*Details to be filled in: the webhook URL goes in the GitHub secret
`ARBOL_SLACK_WEBHOOK_ALERTS_GLOBAL_TESSERA` and the channel joins `infra/slack/channels.py`,
following `#alerts-data`. Everything below is independent of those values.*

**What gets posted, and once per fingerprint:**

| grade | posted | re-posted |
|---|---|---|
| `PROBLEM` (exit 2) | immediately, with `@here` | every 30 min while it stands |
| `DID NOT RUN` | immediately — unknown is not milder than known-degraded | hourly |
| `WARN` | immediately, no mention | **never** — once, then it lives on the shelf |
| `NO DATA` / `NOTED` / `OK` | never | — |
| a **round that did not happen** | after two missed ticks | hourly |

Three rules that matter more than the table:

**A message carries the decision class, not just the finding.** `PROBLEM: cell validation — 2 cells
failed` is a finding; `CLASS 1 CANDIDATE — 2 cells failed the same check` is a decision. The class
comes from §3, which is a lookup, not a judgement.

**Every message carries its `follow_up` command.** `campaign_health` already computes the real
invocation for each finding, filled with this round's window and roster. A message that names a
problem without the next command makes the reader open a runbook at 3 a.m.

**A resolved finding gets one closing message.** A fingerprint that stops appearing is posted once
as resolved and dropped from the state. Without that, the channel is a list of things that may or
may not still be true.

---

## 3. Signals → decision class

This is the heart of the plan. Everything monitoring reports maps here.

### Class 1 — stop everything

Each of these means the work the campaign has *not* done yet is also affected.

| signal | source | why it stops the run |
|---|---|---|
| **the same validation check fails on ≥2 cells** | `cell validation` | one cell is bad luck; two of the same is systemic assembly or model, and every cell after it inherits it |
| **an embedding seam or constant-dimension failure at all** | `cell validation` | both are assembly- or encoder-wide by nature; neither is zone-specific |
| **quantization invariant fails** (scale shares far from 1.0) | `cell validation` | the stored data is not what its readers assume, everywhere |
| **the campaign flow itself CRASHED or FAILED** | `crashed/failed runs` | its own deterministic gate refused something; nothing is being filled while it is down |
| **no GPU fleet places at the requested width, twice running** | `placement`, `fleet widths` | quota or capacity; the campaign is burning wall-clock and ingest storage producing nothing |
| **GPU starvation: instances billing, zero chunks completed** | `gpu starvation` | the one failure with no other symptom — a fleet holding nothing publishes no progress line |
| **Fargate vCPU quota exhausted** | `placement` | ingest cannot start; the fill queue drains and the fleet idles |
| **mass false cancellations / event loss** | `crashed/failed runs`, `orchestrator load` | measured before: the in-memory broker drops events, and the crash automation is event-driven. Healthy runs get declared dead and cancelled |
| **cost accrual far above model with progress flat** | `cost accrual` + `commit rates` | the two together, never either alone: high burn with progress is just a wide fleet |

**The stop itself is not free, so the message says what stopping costs.** Cancel the *tasks* before
stopping infrastructure or runs sit in `CANCELLING` forever, looking live to every guard; a resumed
run measures the remainder rather than the whole; and restart dead **legs**, not cells that landed.

### Class 2 — shelve it, fix in a second wave

| signal | source | what goes on the shelf |
|---|---|---|
| **a zone failed every attempt** | the campaign's own `unfilled` list; `per-cell roster` | zone, year, last error |
| **a cell failed validation, alone** | `cell validation` | zone, year, blocking slugs, verdict URI |
| **an unexplained input-window gap** | `published input coverage`, `cell validation` | which source store was short, by how many months |
| **a published cell with NO verdict** | `cell validation` | the dispatch was lost; needs a validator run, not a refill |
| **a shard written outside the land mask** | `shard placement` | zone, year — a placement defect in one cell |
| **coverage reconciliation does not close** | `cell validation` | written + skipped vs live |
| **orphaned staging under a foreign identifier** | `orphaned staging` | a resume that lost its `run_id`, or a deliberate invalidation |

### Class 3 — expected; note, do not alert

These are the ones that would drown the channel. They are counted per round and never posted.

| signal | why it is not a defect |
|---|---|
| **cells skipped by a deterministic rule** — the optical preflight refusing a cell whose catalogue publishes nothing over its live land; the coverage gate refusing a short window; a cell whose every live tile was skipped | the rule fired correctly. Each still goes on the shelf as a *named zone*, because "correctly skipped" and "we meant to publish it" are different |
| **observation-count seams** | neighbouring MGRS tiles keep different date sets after cloud screening, so the count genuinely steps at a tile edge. Present on every cell |
| **a thin cell** (high `s2_thin_pct`) | optical depth varies by an order of magnitude globally; publish and label |
| **radar-free zones** | parts of the globe have no dual-pol coverage at all, and Sentinel-1B's failure is already documented |
| **a provenance field absent on an older cell** | the fill predates the field. An attrs patch at the end, if it matters at all |
| **a check reporting UNAVAILABLE** | unknown is not wrong. It is graded neither pass nor fail by design |

---

## 4. The shelf — the second product of monitoring

**A durable, per-zone list of everything that needs individual attention at the end.** This is the
half of monitoring that is not alerting, and the half that is most valuable if nothing goes wrong:
the campaign's own retry machinery handles the transient, so what is left when it finishes is a set
of *named zones* — refused by a rule, failed twice, published over a gap, or failed validation.

One entry per `(zone, year, reason)`, appended as findings appear, each with the evidence and the
follow-up already resolved. The second wave is then dispatched from the shelf rather than
reconstructed from nine days of logs. Two properties it needs and neither is free:

* **It is append-only and idempotent.** A finding seen sixty times is one entry, first-seen
  preserved.
* **It closes entries it can.** A zone that later lands, or a cell whose second validation passes,
  is marked resolved rather than deleted — the record of what needed attention *is* the audit trail.

---

## 5. The AI reviewer

**An agent reads what the flow publishes, triages, and keeps a running narrative.** Three jobs:

**1. Triage new findings against §3** and escalate class 1 immediately, with the reasoning stated.
The class table is a lookup, so the agent's value is not the classification — it is noticing the
*combinations* that the per-check grades cannot see: two zones failing differently but at the same
moment, a warning whose subjects are growing round over round, a cost curve bending while progress
holds.

**2. Review the windows** — the one check that otherwise does not scale. The per-cell validation
publishes 10 native-resolution windows and a verdict per cell, and nobody eyeballs a thousand cells.
Per cell the review answers three concrete questions, and *not* a score: which real-world features
are present (field boundaries, drainage, coastline, settlement), which artifacts are present
(blocking, a straight edge crossing the window, repeated texture, uniform noise), and a verdict of
**plausible / cannot-tell / suspicious** where cannot-tell is mandatory. It is given that cell's own
`s2_obs_mean` and told to judge against it, because a noise-like window on a thin cell is the
correct output. **Suspicious flags for a human and never blocks** — "looks wrong to a model" is not
evidence any other check would confirm. Its value is ranking a thousand cells so a human reads the
worst twenty.

**3. Post a running summary** — cells landed, rate against plan, spend against model, the shelf's
length — on a slow cadence (hourly), so the channel has a pulse that is not only bad news.

**Two constraints.** The agent is **not the alerting mechanism of record**: the flow posts the
deterministic warnings itself, and the agent adds interpretation on top. An alert path whose only
link is a model is a nondeterministic single point of failure. And the agent **acts on nothing** —
it recommends a decision and a human takes it, for the same reason the flow mutates nothing.

---

## 6. The first day

Class-1 decisions are only affordable early, so the first day has a protocol of its own. Each line
is a thing that can only be checked once.

| when | check | stop if |
|---|---|---|
| **T+15 min** | the first fleet placed at the width asked; the first ingest committed | no task placed, or the quota refused |
| **T+1 h** | the **first cell's validation verdict** — the first real answer about the product | any blocking finding at all: at one cell, a defect is systemic until proven otherwise |
| **T+2 h** | the first cell's cost against the model; duty cycle above ~80% | burn far above model, or a fleet idling |
| **T+4 h** | the first *window figures* reviewed by a human, not only the AI | the pictures do not look like Earth |
| **T+8 h** | the shelf's growth rate | rules refusing cells far faster than the survey predicted |
| **T+24 h** | cells/day against the deadline; spend against model | the arithmetic no longer reaches 2026-09-11 |

**The T+1 h validation verdict is the single most valuable reading in the campaign.** It is the
first time anything says whether the pixels are right, it costs three minutes and three cents, and
it is the only class-1 signal available before real money has been spent.

---

## 7. What exists, and what this needs

**Built:** the detector (`campaign_health.py`, seventeen checks, `--json`, stable slugs, subjects,
fingerprints, per-finding follow-up commands); every tool it composes; the per-cell validation and
its published verdicts; `cost_accrual.py`, `campaign_progress.py`, `fleet_placement.py`,
`inference_profile.py`, `sweep_health.py`, `prefect_load.py`; and the Slack helper plus its channel
registry.

**To build:**

1. **the `campaign-watch` flow** — cron `*/5`, `concurrency_limit=1` with `CANCEL_NEW` so a slow
   round cannot pile up, tiered cadence, publishing the four artifacts of §1.
2. **the Slack sink** — one channel, the posting rules of §2, and the state file that makes "once
   per fingerprint" true.
3. **the shelf** — §4, which is mostly a projection of `subjects` across rounds.
4. **the agent loop** — §5, reading `rounds/` and the verdicts, with the window review.

**One measurement to take before launch:** a full round against a live dev campaign, timed and
priced, to confirm the $0.055 figure holds at the width we will actually run and that a round
finishes inside its 5-minute tick. A round that overruns its cadence is the one way this design
fails quietly — `CANCEL_NEW` then silently drops rounds, and the schedule stops being a heartbeat.
