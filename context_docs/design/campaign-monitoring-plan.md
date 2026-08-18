# Monitoring the campaign — what we watch, and the three decisions it feeds

The campaign is **not built to be stopped.** Once it is running there is almost no surgical
intervention available: per-zone and per-chunk retries are in the code because they have to be,
and the only hand tool is editing Zarr attrs or a manifest. Everything else is either a side run
or a second wave at the end.

So monitoring is not a dashboard. It exists to produce **one of three decisions** as early as
possible:

| decision | what it means | when it applies |
|---|---|---|
| **INTERVENE NOW** | stop everything, fix, restart | the defect taints work the campaign has not done yet, so every hour it keeps running is wasted money |
| **RESOLVE LATER** | let it run; fix in a second wave | the problem is confined to named zones and does not touch the rest |
| **TAKE NOTE** | record it; maybe patch attrs at the end | a property of the input or of an older image, not a defect |

**Intervene Now** is only affordable in the **first hours**. **Resolve Later** is the common outcome
and its whole product is a **list of zones** to re-run at the end. **Take Note** is most of what
monitoring will see, and the design problem is stopping it from burying the other two.

Two things follow, and they shape everything below. **The cost of a missed Intervene Now signal is
the whole remaining campaign**, so those checks run on the fastest cadence and page. And **the cost
of a noisy Take Note signal is that nobody reads the channel**, so a standing finding must be said
once.

---

## 1. The mechanism

**One flow, `campaign-watch`, on a 5-minute cron. Stateless per round; its memory lives in S3.**

It is a thin wrapper around `campaign_health.py`, which already is the detector: eighteen checks in
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

* **every round (5 min) — the nine free checks.** Fleet strength, empty clusters, sweep success,
  concurrency limits, **crashed/failed runs**, cost accrual, published input coverage, shard
  placement, **cell validation**. No Insights query, so the round is essentially free — and this
  set carries the two most decision-relevant signals we have.
* **every third round (15 min) — the eight paid checks.** Commit rates, per-cell progress,
  orchestrator load, throttle pressure, fill scope, orphaned staging, GPUs kept busy, GPU
  starvation. Between them these are the five Insights queries; the rest read the same rows. This
  keeps the measured **~$5/day** cost model, a rounding error against one idle 20-worker
  fleet-hour (~$37).

The split lives in `campaign_health.py` as `PAID_CHECKS`, and `--cheap` is what a round passes. A
skipped check reports **`SKIPPED`**, which is deliberately neither `OK` nor `DID NOT RUN`: graded
unrunnable it would report a healthy campaign as degraded eight checks at a time, and graded healthy
it would tell the poll that a standing warning had cleared — which posts a false all-clear now and
the same finding as new fifteen minutes later.

**What each round publishes**, all under `s3://global-tessera-embeddings/monitoring/`:

| artifact | what it is for |
|---|---|
| `rounds/<timestamp>.json` | the full `campaign_health --json` report, verbatim. The audit trail, and what the agent reads |
| `state.json` | fingerprint → `first_seen`, `last_posted`, `times_seen`. The dedup and ageing memory |
| `shelf.json` | the Resolve Later list: per zone, what is wrong, the evidence, and whether it is resolved (§4) |
| `latest.json` | the newest round, for a cheap read |

**A round never mutates the campaign.** No cancelling, no reopening, no dispatching a refill. The
one write it makes is its own record. A monitor that can act is a monitor that can act on a false
positive, and a refill costs hours and real money.

**Three things a round deliberately no longer raises, each because it paged for one on 2026-08-18.**
The rule behind all three is the same: an alert has to describe the campaign as it is now, not the
record the campaign left behind.

| what it used to raise | why that was wrong | what it says now |
|---|---|---|
| a failure whose retry has since succeeded | ingest re-runs a failed stream by design, so the failed attempt keeps its record long after the work landed. One zone was still paging with `@here` two hours after its retry finished and its cell had published | reports it as absorbed, naming the zone and time, so an operator who saw the first alert learns the incident is closed |
| a shrinking fleet at the end of a run | workers do not vanish when their work ends, they shut down over minutes. One surviving worker read as nought per cent of what the run had asked for | says nothing while no ingest is running — with nothing asking for workers, there is no width for them to fall short of |
| a cell described by the wrong phase | a cell whose radar stream had already died still read as "waiting to infer" — true of its two surviving streams, false about the cell | names the dead stream and the time, so the per-cell view and the failure list agree |

A fourth is a wording fix rather than a suppression. The commit-rate figures are **totals across
every cell that is ingesting**, so a fall in them belongs to no single zone and is often just one
cell finishing. The alert now says that, and points at the per-cell check for attribution.

---

## 2. Slack — `#alerts-global-tessera`

**Two paths post to this channel, and the split is deliberate.** One is fast and lossy, the
other slow and reliable, and neither substitutes for the other.

### 2.1 The fast path: server-side automations — **BUILT**

Four things go wrong in a way the orchestrator observes the instant it happens, so for those a
Prefect automation posts with no polling and no code of ours running.
`scripts/register_alert_automations.py` registers them, and they are live on both accounts:

| automation | fires on | urgency in the message |
|---|---|---|
| `notify-cell-validation-failed` | a `validate-zone-year` run FAILED | the highest — the only alert about the published product |
| `notify-campaign-driver-stopped` | `run-global-campaign` FAILED or CRASHED | the show has stopped; every hour is deadline, not money |
| `notify-fill-run-lost` | a fill run FAILED or CRASHED | context; the driver retries and the sweep reclaims the fleet |
| `notify-ingest-failed` | an `ingest-zone-year` run FAILED or CRASHED | a **warning**, said as one: the cell keeps what it committed and is retried |

Choices in there worth keeping:

* **`CANCELLED` is not an alerting state.** An operator cancelling a run already knows, and the
  campaign's own recovery cancels children by design — including in a normal teardown — so
  alerting on it would post routinely, which is how a channel becomes unread.
* **The ingest alert exists for the REPEAT, not the failure.** A single failed ingest resolves
  itself: everything already fetched is committed, the cell is retried on the standing cluster and
  then across dispatch rounds, and a resume continues from the first missing date. What is not
  self-resolving is a pattern — the same cell failing every time (a deterministic gate), or
  different cells failing the same way at once (a catalogue refusing requests, a quota, a provider
  timing out over one region). No single event can see either, so the message names both readings
  and points at the check that counts. Without it the first report of a bad cell is the driver's
  end-of-run list, days later.
* **The validation alert does not guess *why* it failed.** "Found a defect" and "could not run"
  both end as a FAILED run and the difference is the exception type in the log, so the body names
  both readings and says they need opposite responses — refill nothing, re-dispatch the validator.
* **Every message names the run four ways and carries the commands that open it.** The run name a
  person recognises, the run id every tool takes as an argument, the deployment, and the fully
  qualified event resource an events query keys on — then `watch_run.py` and `campaign_health.py`
  with the account and the id already filled in. An alert that names a problem without the next
  command sends the reader to find a runbook at 3 a.m., and an agent given an id and a command can
  look for itself.
* **Those expressions were checked against the server rather than assumed.** An invalid Jinja
  expression makes an action fail with *nothing sent and nothing wrong on the run*, invisible
  outside the orchestrator's own event log. The server builds an action's template context by
  scanning the template for the native objects it knows (`flow_run`, `deployment`, `flow`, …) and
  fetching them from its own API, and its Jinja environment uses `ChainableUndefined` — so a lookup
  that fails renders EMPTY rather than raising. An earlier version of this plan allowed only
  `{{ event.resource.id }}` for fear of the silent-failure case; that was wrong twice over, and
  reading the server's source plus firing a real event is what settled it.

**This path is lossy, by measurement.** Prefect's event broker is in-memory and drops events
under load — we have watched it declare healthy runs dead — and it cannot fire at all when the
server itself is the unwell thing, which is one of the conditions most worth alerting on. Hence
the second path.

### 2.2 The reliable path: the periodic poll — **BUILT**

The round of §1 posts what the automations cannot: everything that is a *reading* rather than an
event (stalls, how busy the GPUs are, cost, quota, a published cell with no verdict), plus a second chance
at everything above. Its rules:

| grade | posted | re-posted |
|---|---|---|
| `PROBLEM` (exit 2) | immediately, with `@here` | every 30 min while it stands |
| `DID NOT RUN` | immediately — unknown is not milder than known-degraded | hourly |
| `WARN` | immediately, no mention | **never** — once, then it lives on the shelf |
| `NO DATA` / `NOTED` / `OK` | never | — |
| a **round that did not happen** | after two missed ticks | hourly |

Three rules that matter more than the table:

**A message carries the decision class, not just the finding.** `PROBLEM: cell validation — 4
cells failed` is a finding; `INTERVENE NOW? — 4 cells failed the same check` is a decision.
The class comes from §3, which is a lookup, not a judgement. The check's `systemic_checks` subject
is what the lookup keys on, so the class never depends on re-counting prose.

**Every message carries its `follow_up` command.** `campaign_health` already computes the real
invocation for each finding, filled with this round's window and roster. A message that names a
problem without the next command makes the reader open a runbook at 3 a.m.

**A resolved finding gets one closing message.** A fingerprint that stops appearing is posted
once as resolved and dropped from the state. Without that, the channel is a list of things that
may or may not still be true.

#### How the round learns the roster and the store, and the asymmetry between them (2026-08-17)

A cron fires with no arguments, so the round must *discover* what the campaign was dispatched with. It
reads the live dispatch rather than a stored parameter, because storing it on the deployment put it
somewhere `deploy_flow.py` overwrites and every re-registration silently dropped it. Two dispatch
shapes exist and they name the roster differently: the driver carries `zones` (crossed with `years`), a
directly-dispatched fill carries `cells` as `[["49S", 2022], …]`.

**The defect this fixes, found by running the shape that triggers it.** `_live_dispatch` read only
`run-global-campaign`, while `campaign_health.live_cells` already read `fill-*` runs to learn which
cells were being worked. So a fill dispatched on its own — how a rehearsal or a single-cell recovery is
run — was **seen as live by five checks while the roster and store stayed unknown**, and the round
graded *the detector's default store* instead of the one being filled. The cell-validation check, the
only one about the published product, examined a store nobody was filling and reported nothing wrong.
The single outward sign was a log line, `roster NOT GIVEN … store the detector's default`, which reads
like a benign notice.

**The two are not symmetric, and the rule is worth stating because the obvious fix is wrong.**

| | where it comes from | why |
|---|---|---|
| **roster** | the drivers when any driver run is live, **else** the fills — never both | a driver-led campaign dispatches one fill per cell, so its fills name only what is in flight. Unioning them would *narrow* the roster to the cells running at that instant and drop the queued and finished ones |
| **store** | either shape, whichever names one | a store name cannot narrow anything, and a fill names the store it is filling exactly as authoritatively as a driver does |

The test that asserted the old exclude-fills-outright behaviour was kept and re-aimed at the property
it was really protecting — that a driver's roster survives its own fills — rather than deleted.

### 2.3 Where the webhook lives

**Secrets Manager, `global-tessera/slack-webhooks`, in each tessera account** — one JSON key per
channel (`alerts-global-tessera`), so dev alerts cannot land where prod alerts go. The
registration script reads it from the deployment's own account and writes it into a Prefect
`SlackWebhook` block, which the server stores encrypted; the automations reference the block by
document id. Rotating the webhook means updating the secret and re-running the script with
`--apply`. **The URL appears in no file in either repository.**

**One thing that is NOT a bug, so nobody re-fixes it.** Subject and body reach Slack as a legacy
attachment's `title` and `text` — two fields, rendered as a bold line above the body. Copying such
a message out of Slack joins them with no newline (`...FAILED VALIDATIONA cell validation run
FAILED`), which reads like a formatting fault and is not one. A leading newline was briefly added
to "fix" it and only produced a blank first line; capturing apprise's payload settled it.

**One trap, paid for once.** `Block.save()` defaults to the *ambient* Prefect client, and Prefect
resolves its settings at import — so on a laptop carrying a `prefect cloud login` profile, the
first version of the script wrote the live webhook into an unrelated **Prefect Cloud** workspace.
It looked like it worked: the block saved, `load()` read it straight back, and the only symptom
was the automation's action failing with a *404 for a block id the deployment's own server had
never heard of*. A secret in the wrong system, and an alert path that appeared registered. The
save now goes through `explicit_client()` — the same client the automations use, bound to the URL
the session exported — and the script refuses to write anything if that URL is not the target's.

## 3. Signals → decision class

This is the heart of the plan. Everything monitoring reports maps here.

### Intervene Now — stop everything

Each of these means the work the campaign has *not* done yet is also affected.

| signal | source | why it stops the run |
|---|---|---|
| **the same validation check fails on ≥4 cells** | `cell validation` | one or two is bad luck — the checks describe the input as much as the code, and a 1,008-cell roster is heterogeneous enough to produce a pair on its own. Four of a kind is systemic assembly or model, and every cell after it inherits it. The check reports the running total per check from the first failure, so the count is visible on the way up |
| **an embedding seam or constant-dimension failure at all** | `cell validation` | both are assembly- or encoder-wide by nature; neither is zone-specific |
| **quantization invariant fails** (scale shares far from 1.0) | `cell validation` | the stored data is not what its readers assume, everywhere |
| **the campaign flow itself CRASHED or FAILED** | `crashed/failed runs` | its own deterministic gate refused something; nothing is being filled while it is down |
| **no GPU fleet places at the requested width, twice running** | `fleet strength`, `empty clusters` | quota or capacity; the campaign is burning wall-clock and ingest storage producing nothing |
| **GPU starvation: instances billing, zero chunks completed** | `gpu starvation` | the one failure with no other symptom — a fleet holding nothing publishes no progress line |
| **Fargate vCPU quota exhausted** | `fleet strength` | ingest cannot start; the fill queue drains and the fleet idles |
| **mass false cancellations / event loss** | `crashed/failed runs`, `orchestrator load` | measured before: the in-memory broker drops events, and the crash automation is event-driven. Healthy runs get declared dead and cancelled |
| **cost accrual far above model with progress flat** | `cost accrual` + `commit rates` | the two together, never either alone: high burn with progress is just a wide fleet |
| **a concurrency gate DEACTIVATED while fills are live** | `concurrency limits` | it does the opposite of what it sounds like: the server grants slots against an inactive limit, so the work it throttled runs unthrottled. A cost control silently gone, and one command to fix. (A gate at *zero* is a pause lever and is reported without being graded: `tessera-global-ingests` at zero holds ingest, `tessera-global-inference` at zero holds inference, and in both cases work waits rather than failing) |

**The stop itself is not free, so the message says what stopping costs.** Cancel the *tasks* before
stopping infrastructure or runs sit in `CANCELLING` forever, looking live to every guard; a resumed
run measures the remainder rather than the whole; and restart dead **legs**, not cells that landed.

### Resolve Later — shelve it, fix in a second wave

| signal | source | what goes on the shelf |
|---|---|---|
| **a zone failed every attempt** | the campaign's own `unfilled` list; `cells still making progress` | zone, year, last error |
| **a cell failed validation, alone** | `cell validation` | zone, year, blocking slugs, verdict URI |
| **an unexplained input-window gap** | `published input coverage`, `cell validation` | which source store was short, by how many months |
| **a published cell with NO verdict** | `cell validation` | the dispatch was lost; needs a validator run, not a refill |
| **a shard written outside the land mask** | `shard placement` | zone, year — a placement defect in one cell |
| **coverage reconciliation does not close** | `cell validation` | written + skipped vs live |
| **orphaned staging under a foreign identifier** | `orphaned staging` | a resume that lost its `run_id`, or a deliberate invalidation |

### Take Note — expected; record, do not alert

**This is a list of things that will appear and must not be escalated.** That is its only purpose:
each row is something a reader could reasonably mistake for a defect, so the table's job is to say
why it is not one and stop it consuming an hour at 3 a.m. They are counted per round and never
posted.

| signal | why it is not a defect |
|---|---|
| **cells skipped by a deterministic rule** — the optical preflight refusing a cell whose catalogue publishes nothing over its live land; the coverage gate refusing a short window; a cell whose every live tile was skipped | the rule fired correctly. Each still goes on the shelf as a *named zone*, because "correctly skipped" and "we meant to publish it" are different |
| **observation-count seams** | the counts genuinely step at an MGRS tile edge, because neighbouring tiles keep different date sets after cloud screening. It is listed only because the check's NAME reads like a defect and it is present on nearly every cell — the validator classes it as an expected upstream finding and does not even count it, so it never reaches a message. What would be worth reading is a seam in the **embeddings**, which is a different check and is not on this list |
| **a thin cell** (high `s2_thin_pct`) | optical depth varies by an order of magnitude globally; publish and label |
| **radar-free zones** | parts of the globe have no dual-pol coverage at all, and Sentinel-1B's failure is already documented |
| **a provenance field absent on an older cell** | the fill predates the field. An attrs patch at the end, if it matters at all |
| **a check reporting UNAVAILABLE** | the check could not be evaluated — which is not the same as the check failing. It fires when something the comparison needs is *absent* rather than wrong: most often a recorded figure the cell's fill never wrote, because that fill predates the field the check reads. The absence is in the RECORD, not in the pixels, and grading it either way asserts something untrue — "pass" hides a comparison that never happened, "fail" reports a store's age as a data fault. This is not hypothetical: an earlier version turned a missing recorded value into `NaN`, compared it, and reported the *record* as wrong on a healthy cell |

#### OPEN — the 5xx arm of `orchestrator load` fires on n=1 (2026-08-17)

Round 1398 of the `tessera-min15` rehearsal graded the campaign **DEGRADED** on this evidence:

> dropped events (broker overflow): **0**; server CPU now: **2%**; latency now: **16ms** mean at
> 1 req/s; HTTP 5xx from the API: **1**

The arm is `if signals["err_5xx"]:` — **any** non-zero count warns. At 1 req/s over a 60-minute
window that is one error in roughly 3,600 requests, a 0.03% rate, against an orchestrator that is
otherwise idle by every other signal it just measured. The message reads "The orchestrator returned
1 server errors" — which also has the n=1 grammar bug.

**Not a false alert**: the 5xx was real, and it routes to `warnings`, so it never paged. But it is
the shape the corrections register calls *an infra hiccup promoted to a finding*, and it puts
"DEGRADED" on the verdict line of a healthy campaign, which is what makes the channel tiring to
read. A rate- or count-based floor is wanted instead of truthiness — the other three signals in the
same detail line are the evidence that one 5xx means nothing.

**Deliberately not changed while the rehearsal was in flight**, because `campaign-watch` pulls its
image at each round launch: fixing it mid-run would have split the observation across two versions of
the detector and made the message-volume count meaningless.

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

## 5. The AI reviewer — **BUILT 2026-08-13**

**An agent reads what the flow publishes, triages, and keeps a running narrative.** Three jobs:

**1. Triage new findings against §3** and escalate an Intervene Now immediately, with the reasoning
stated.
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

**Where it posts: `#alerts-global-tessera`, the same channel as everything else.** One place to
look, because splitting the narrative from the alerts splits attention exactly when attention
matters. Two rules make that safe. Its posts are **prefixed and attributed** so a reader can tell an
interpretation from a deterministic finding at a glance — the automations and the poll state facts,
the agent states readings. And its cadence is **bounded**: urgent escalations immediately, the
narrative hourly, nothing else. If the hourly pulse ever turns out to bury real alerts, the fix is
to move the pulse elsewhere and keep the escalations here, never the other way round.

**Two constraints.** The agent is **not the alerting mechanism of record**: the flow posts the
deterministic warnings itself, and the agent adds interpretation on top. An alert path whose only
link is a model is a nondeterministic single point of failure. And the agent **acts on nothing** —
it recommends a decision and a human takes it, for the same reason the flow mutates nothing.

### What was built, and the four things measuring it changed

**The reviewer is a committed skill**, `.claude/skills/campaign-review/SKILL.md`, so it is versioned,
reviewable in a pull request, and its commit is stamped into every review it produces — which is how a
later reader tells which instructions produced a verdict. It drives `scripts/review_cells.py` and
nothing else: that script hands out cells worst-first, validates and stores reviews, and posts prefixed
messages, and none of it can modify the store, a run or a deployment. **That is what makes "the agent
acts on nothing" mechanical rather than advisory.**

**It is measured before it is trusted.** `scripts/review_calibration.py` builds a labelled set from
real published windows, half damaged in six ways a placement, quantization or write defect would
produce, with opaque names, with the flat-but-healthy windows deliberately among the clean cases, and
with each window's damaged twin in the opposite arm so damage cannot be found by comparing. The score
card prints catch rate and false-alarm rate together and names a collapsed distribution as a stuck
instrument. A blind session scored **7 of 7 caught, 0 false alarms** on one arm.

Four things the measurement changed in the instructions, each of which had already produced a wrong
reading:

* **Judge against the window's OWN optical depth**, never the cell's. A cell holds windows differing
  severalfold, and one cell's mean described nowhere in its own frames.
* **Pale is a statement about a window relative to its cell**, not about its quality — the shared
  stretch compresses a low-variation window rather than amplifying it.
* **Faint is not empty.** Windows were measured carrying a full drainage network at a tenth of the
  amplitude the stretch renders visible.
* **Swath boundaries are upstream and appear in about one window in sixteen** — a dead-straight line
  fixed in geography, recurring at the same place across years, coinciding exactly with a step in the
  observation count. Without that, they are the artifact a reviewer flags most confidently and most
  wrongly.

**Coverage of the review is itself a monitored signal** — check 18, `window review`. It reports how
many published cells have been looked at and never grades that as a fault, because the review is a
deliberate worst-first sample. The one reading it does grade is **unanimity**: a reviewer that has
collapsed onto a single answer is indistinguishable from a campaign going well, and in production
there are no labels, so unanimity is the only symptom available.

---

## 6. The first day

Intervene Now decisions are only affordable early, so the first day has a protocol of its own. Each line
is a thing that can only be checked once.

| when | check | stop if |
|---|---|---|
| **T+15 min** | the first fleet placed at the width asked; the first ingest committed | no task placed, or the quota refused |
| **T+1 h** | the **first cell's validation verdict** — the first real answer about the product | any blocking finding at all: at one cell, a defect is systemic until proven otherwise |
| **T+2 h** | the first cell's cost against the model; GPUs busy above ~80% of the time | burn far above model, or a fleet idling |
| **T+4 h** | the first *window figures* reviewed by a human, not only the AI | the pictures do not look like Earth |
| **T+8 h** | the shelf's growth rate | rules refusing cells far faster than the survey predicted |
| **T+24 h** | cells/day against the deadline; spend against model | the arithmetic no longer reaches 2026-09-11 |

**The T+1 h validation verdict is the single most valuable reading in the campaign.** It is the
first time anything says whether the pixels are right, it costs three minutes and three cents, and
it is the only Intervene Now signal available before real money has been spent.

---

## 7. What exists, and what is left

**Everything in this plan is built.** The fast path (server-side automations), the reliable path
(the periodic poll), the Take Note tier, and the AI reviewer all shipped, and each section above
carries its own BUILT marker with what measuring it changed.

The hundred-line build-status inventory that stood here is cut: it enumerated components, owners and
readiness for work that is now running, and the code plus those markers say it better. What it
recorded that the code does not:

**The two paths are deliberately redundant and neither is a fallback for the other.** Prefect's event
broker drops events under load and cannot fire when the server itself is unwell, so the fast path is
fast and unreliable by construction; the periodic poll is the mechanism of record. A design where the
poll existed only as a backstop would have been quietly wrong, because the case that breaks the fast
path is exactly the case you need an alarm in.

**The AI reviewer is not on the alerting path at all** — an alert whose only link is a model is a
nondeterministic single point of failure, so the deterministic findings post whether or not a review
ever runs.

