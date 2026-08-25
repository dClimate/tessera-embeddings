# EC2 launch throttling: where it comes from and what we changed

**Status:** measured and implemented, default-off, 2026-08-25.
**Code:** `inference/scheduling.py::ACTOR_REQUEST_HEADROOM`,
`providers/aws/ray.py::LAUNCH_PACING_ENV`.

---

## 1. The problem, in one paragraph

A global campaign runs several GPU inference fleets at once, each on its own Ray
cluster, each asking for 250 `g6e.xlarge` actors. Every cluster's autoscaler launches
instances by calling EC2's `RunInstances`, and the account meters that call with a
request bucket shared by every caller in the account. Five clusters requesting at the
same moment drain it, everything behind them is refused, the refusals are retried, and
the retries drain it again.

The quota, read from Service Quotas in **both** `global-tessera-prod` and
`global-tessera-dev` (they agree, because AWS does not vary it):

| Quota | Code | Value | Adjustable |
|---|---|---|---|
| RunInstances request bucket maximum capacity | `L-8EEB235C` | **5** | **No** |
| RunInstances request bucket refill rate | `L-A2D70E6C` | **2 / sec** | **No** |

There is no increase to ask for. The ceiling is fixed and the only variable is how we
behave against it.

CloudTrail over one twelve-minute production window:

| Outcome | Count |
|---|---|
| `Client.RequestLimitExceeded` | **402** |
| `Server.InsufficientInstanceCapacity` | 167 |
| `Client.UnauthorizedOperation` (our own fleet caps) | 29 |
| accepted | **2** |

Average call rate over that window was **0.97 / sec** — comfortably under the 2 / sec
refill. Being under the average rate and refused 67% of the time is the signature of a
burst problem: the bucket holds five tokens, and five autoscalers arriving together
empty it before the refill matters.

A separate, real capacity shortage sits underneath this (the 167). That half is AWS's
and is being raised with them. Everything below is the half we inflict on ourselves.

---

## 2. Where the retry lives — established before anything was designed

The launch path, traced through Ray 2.55.1 as installed:

1. `autoscaler/_private/autoscaler.py::launch_new_node` splits a fleet request into
   chunks of `AUTOSCALER_MAX_LAUNCH_BATCH` (default **5**) and queues them for
   `ceil(AUTOSCALER_MAX_CONCURRENT_LAUNCHES / AUTOSCALER_MAX_LAUNCH_BATCH)` launcher
   threads — with the defaults, **2 threads per cluster**.
2. `autoscaler/_private/aws/node_provider.py::_create_node` sets
   `MinCount=1, MaxCount=count` and calls `create_instances`.
3. That call goes through `self.ec2_fail_fast`, built by
   `aws/utils.py::resource_cache(..., max_retries=0)` — botocore performs **zero**
   retries on a launch, deliberately.
4. Ray wraps the call in **its own** retry loop: `for attempt in range(1, max_tries+1)`
   where `max_tries = max(BOTO_CREATE_MAX_RETRIES=5, len(subnet_ids))`, rotating to the
   next subnet on **any** `ClientError`, **with no sleep between attempts**.

**So we do not own the retry.** It is Ray's, it is a tight loop, and there is no
callback, no hook, and no configuration that reaches inside it. Two findings fell out of
reading it that matter beyond this change:

* **A throttled attempt spends a subnet rotation.** The loop rotates on any error, so
  refusals consume the multi-AZ capacity failover that `gotchas.md` documents. A fleet
  can exhaust its subnet list on throttling without ever having asked the later zones
  for capacity — which makes throttling and the capacity shortage compound rather than
  merely coexist.
* **`AWS_MAX_ATTEMPTS` cannot reach that client.** botocore's
  `args.py::_compute_retry_max_attempts` returns early when the client config names
  `max_attempts` explicitly, and Ray names it.

**What we can reach.** `_compute_retry_mode` returns early only when the client config
names a `mode`, and Ray never does — so the retry MODE is still resolved from the
environment. `AWS_RETRY_MODE=adaptive` therefore reaches Ray's launch client, and
`retries={'max_attempts': 0, 'mode': 'adaptive'}` is a legitimate combination: adaptive
mode registers `ClientRateLimiter.on_sending_request` on `before-send`, which runs on
every attempt regardless of how many attempts are allowed.

**The batch question, answered from the code:** one `RunInstances` call asks for up to
`AUTOSCALER_MAX_LAUNCH_BATCH` instances — **5** by default, not 1. CloudTrail does not
record the count, which is why it had to be read rather than inferred.

**The starvation bound, also from the code:** `botocore/retries/bucket.py` sets
`TokenBucket._MIN_RATE = 0.5` and `self._fill_rate = max(value, self._min_rate)`. A
paced client's send rate cannot fall below 0.5 / sec no matter how much throttling it
sees, so the induced delay is at most 2 seconds per call. The bound is structural and
lives in botocore, not in code we wrote.

---

## 3. The larger half: the request escalates on its own

Pacing makes each launch request cheaper. It does nothing about how many there are, and
that turned out to be the bigger term.

`inference/scheduling.py::_batch_actors_to_request` gates the next actor batch on the
previous batch's instances having joined the cluster, measured incrementally. That gate
is sound. Its escape hatch was not:

```
timed_out = secs_since_last_batch > placement_timeout_sec
```

On expiry it requested another **full** batch, on the reasoning that a shortfall must
not gate every remaining batch forever. Right for a slow pool, wrong for a closed one.
Production:

```
Requested actor batch: +50 (100/250 total, 1 GPU nodes placed) — placement timed out, requesting anyway
Requested actor batch: +50 (150/250 total, 2 GPU nodes placed) — placement timed out, requesting anyway
Requested actor batch: +50 (200/250 total, 3 GPU nodes placed) — placement timed out, requesting anyway
Requested actor batch: +50 (250/250 total, 5 GPU nodes placed) — placement timed out, requesting anyway
```

Five nodes placed, 250 requested, on nothing but four expiries. Every unplaced request
is a pending Ray actor, and a pending actor is unsatisfied resource demand the
autoscaler goes on trying to place — against the same five-token bucket — for as long
as the fill lives.

**That exposure just got 12x longer.** The first-actor wait was raised from 30 minutes
to 6 hours (`inference/lifecycle.py::ACTOR_INIT_TIMEOUT_SEC = 21600`), for good reasons
of its own. With a 300-second placement interval a fill now sits through up to 72
expiries instead of 6. Under the old rule it reaches its target after four of them and
then holds 245 unplaceable requests against the account for the remaining five and a
half hours. The relaxed deadline changes what throttling costs — fills no longer die of
it, they take much longer to reach useful width — and it makes bounding the request
more valuable, not less.

### The rule that replaced it

**A run may never request more actors than the GPU nodes it actually holds, plus a
fixed headroom.** `ACTOR_REQUEST_HEADROOM = 25`.

Not a climb toward the target in batches, but a small constant distance ahead of
reality. It needs no gate: each placement earns the right to ask for a little more, so
growth is automatic where capacity exists and simply stops where it does not. The
placement gate and its timeout are not consulted on this path — there is nothing left
for them to release. They remain in the code only because the historical path is still
the default; retiring that path retires them, along with `nodes_at_last_batch`,
`last_batch_size`, `secs_since_last_batch` and `placement_tolerance`.

Four properties, each of which had to be got right:

* **Nodes, not ready actors.** The bound is about hardware the run already holds and
  pays for. Counting readiness would make a slow checkpoint load look like a capacity
  shortage and stall a fleet that already has its instances.
* **The cold start.** At zero nodes the bound still permits `headroom` requests — the
  allowance is a distance ahead of the fleet, and at the start that distance is the
  whole ask. Applied in `inference/runner.py` too, since the first batch is requested
  before the scheduler loop begins.
* **A second ceiling, not a replacement.** `outstanding_work` caps the request by
  remaining WORK; this caps it by placement; the smaller wins. A short tail still
  cannot be over-provisioned.
* **It cannot stall.** A request is refused only while the fleet is already a full
  headroom ahead of its nodes, and every node that joins re-opens exactly that much
  room.

Why 25 is a constant rather than a fraction of the target: the quota it protects is a
property of the ACCOUNT. A fleet of 250 and a fleet of 20 draw on one bucket, and
sizing the allowance to the ask is what let the larger one drown the smaller.

---

## 4. The testing plan, and why it can fail

Three layers, each with a stated control and a stated way to lose.

| Layer | Question it can answer | Control |
|---|---|---|
| Pure unit walk of `_batch_actors_to_request` | Does the request stay within the headroom under a drought, and still reach the target when capacity returns? | The same walk with `headroom=None` |
| Local five-token-bucket harness driving **Ray's real node provider** | Does the pacing environment reduce refusals per instance placed, without costing fleet? | The same harness with the environment unset, run interleaved |
| Real EC2 API in `global-tessera-dev`, `DryRun=True` | Does the real service refuse with the code botocore treats as throttling, does the bucket behave as the quota says, and does adaptive mode pace against real responses? | Unpaced arms, interleaved with paced ones |

**What would have sunk it.** Any of: the paced arm placing fewer instances than the
control; the paced arm not reducing refusals per instance; the paced arm falling silent
while the bucket was shut; the bounded arm failing to reach its target once capacity
returned. The third is the one to fear — a refused request is retried, while a fleet
that stops asking waits for nobody — so it is asserted against a bucket held
deliberately closed rather than argued from the code.

**Every test was verified to fail with the change reverted**, by neutralising the new
branch in place and re-running: 10/10 scheduler tests, 4/4 integration tests, and the
provider tests (3 behaviourally, 1 structurally).

### What the harness is, and what it is not

`tests/integration/_launch_pacing_sim.py` implements the request bucket and speaks
enough of the EC2 query protocol for boto3 to parse a launch or raise
`RequestLimitExceeded`. **Only the endpoint is a stand-in.** The client, its retry
configuration, and the subnet-rotating retry loop are Ray's own code, driven unmodified.
Each simulated cluster runs in its **own process**, because Ray caches the EC2 resource
per `(region, max_retries)` with `lru_cache` — several clusters inside one process would
share a single adaptive limiter, a coordination they do not have in the field. Service
latency is simulated (50 ms refused, 400 ms accepted) so the offered load is shaped like
an API round trip rather than by loopback speed.

It models the **initial ramp** of several clusters, which is the burst. It does not
model a twelve-minute production average, and the two should not be compared.

---

## 5. Measurements

### 5.1 Local harness, five clusters x 250 nodes, one autoscaler process each

| Arm | Calls | Throttled | Instances placed | Refusals / instance | Wall | Instances / sec |
|---|---|---|---|---|---|---|
| control | 1206 | 1186 | 100 | 11.86 | 8.5 s | 11.8 |
| control (repeat) | 1202 | 1183 | 95 | 12.45 | 8.1 s | 11.7 |
| **pacing (full)** | **87** | **42** | **1125** | **0.037** | 22.1 s | **50.9** |
| **pacing (repeat)** | **81** | **37** | **1100** | **0.034** | 22.2 s | **49.5** |
| adaptive mode only | 448 | 212 | 1180 | 0.180 | 116.8 s | 10.1 |
| bigger batch only | 217 | 206 | 275 | 0.749 | 4.1 s | 67.6 |

Read it this way:

* The **control is stable** — 11.86 against 12.45 across two runs is a 5% spread, and
  the treatment sits 300x away from both. The separation is not weather.
* **Neither knob alone is the answer.** Adaptive mode alone places the fleet but takes
  117 seconds to do it, because at a batch of 5 it needs five times as many paced calls.
  A bigger batch alone cuts call volume but is still refused 95% of the time. Batching
  supplies the throughput; adaptive supplies the pacing.
* **Pacing is faster, not slower.** 50 instances per second against the control's 12.
  The relevant safety question was whether pacing costs fleet, and it does the opposite.

### 5.2 The starvation bound, bucket held shut for 30 seconds

| Arm | Calls | Throttled | Placed | Call rate while shut |
|---|---|---|---|---|
| pacing | 112 | 87 | 625 | **2.500 / sec** |
| control | 1250 | 1250 | **0** | 41.7 / sec |

2.500 / sec is exactly five clients times botocore's 0.5 / sec floor. The paced fleet
never went silent, and when the bucket reopened it still had attempts left and placed
625 instances. The control spent all 1250 of its attempts inside the closed window and
placed **nothing** — so under a sustained stall, the paced arm is the one that survives
to use the capacity when it returns.

### 5.3 Repo integration test scale, three clusters x 75 nodes

| Arm | Calls | Throttled | Placed | Refusals / instance |
|---|---|---|---|---|
| control | 196 | 186 | 50 | 3.720 |
| control (repeat) | 199 | 189 | 50 | 3.780 |
| pacing | 12 | 3 | 225 | 0.013 |
| pacing (repeat) | 12 | 3 | 225 | 0.013 |
| pacing, bucket shut 6 s | 15 | 9 | 150 | 0.060 |

Control spread 1.6%; the treatment is identical across runs. This is the scale the
committed test runs at — about 19 seconds.

### 5.4 Real EC2 API, `global-tessera-dev`, five processes x two threads, DryRun

Nothing was launched and nothing was billed: every call carried `DryRun=True`, so the
request was processed and metered but created no instance.

| Arm | Calls | Throttled | Accepted | Throttled fraction | Call rate | Accepted rate |
|---|---|---|---|---|---|---|
| legacy | 91 | 61 | 30 | 0.670 | 9.13 / s | 3.01 / s |
| adaptive | 63 | 32 | 31 | 0.508 | 6.44 / s | 3.17 / s |
| legacy (repeat) | 95 | 59 | 36 | 0.621 | 9.61 / s | 3.64 / s |
| adaptive (repeat) | 57 | 28 | 29 | 0.491 | 6.46 / s | 3.29 / s |

Arms were interleaved, with the bucket allowed to refill between them.

What this establishes, which the local harness cannot:

1. The real API refuses with **`RequestLimitExceeded`** — the exact code in botocore's
   throttling list. The mechanism is not an artefact of our stand-in.
2. The bucket behaves as the quota says. The unpaced arms sustained 3.01 and 3.64
   accepted calls/sec over ~10 seconds, consistent with 2/sec refill plus the initial
   five-token burst.
3. Adaptive mode paces against real responses: 35% fewer calls, and the throttled
   fraction falls from 0.67/0.62 to 0.51/0.49.
4. **It gave up no useful throughput.** Accepted calls went 30/36 to 31/29 — unchanged
   within noise. The quota grants what it grants; pacing only stops us spending
   attempts on refusals.

An honest caveat about magnitude: the dev probe isolates **adaptive mode alone at fixed
offered load**, because a DryRun probe issues a fixed number of calls and the batching
knob only matters when Ray's autoscaler is deciding how many calls a fleet needs. The
large effect in §5.1 comes from the combination. Do not quote the two together.

A useful coincidence worth recording: the legacy arm's throttled fraction (0.67) matches
the production window's (402 refused of 600 attempts, 0.67) almost exactly, which is the
best evidence available that the harness reproduces the right regime.

### 5.5 The escalation, walked

Target 250, batch 50, a region that will place 5 nodes and no more, walked over the
whole first-actor wait (21600 s / 300 s = 72 placement intervals):

| Arm | Final requested | Nodes placed | Furthest ahead of placement |
|---|---|---|---|
| historical | 250 | 5 | 245 |
| headroom = 25 | 30 | 5 | 25 |

And with the shortage lifting at the fourth interval, both arms reach 250 requested and
250 placed. The bound does not strand a fleet; it stops one manufacturing throttle while
it waits.

One field observation, not a controlled result and not claimed as one: a fill dispatched
by hand asking for 47 actors instead of 250 got its first actor within about fifteen
minutes, on a roster that had failed twice at 250. One data point, consistent with the
mechanism. It is the reason "ask for what you will actually be granted" is stated as a
principle rather than derived from it.

---

## 6. What shipped, and how it is turned on

Everything defaults to today's behaviour. Merging changes nothing about a campaign in
flight; each lever is a deliberate act.

| Lever | Where | Default | Turned on by |
|---|---|---|---|
| `actor_request_headroom` | `InferenceConfig`, both fill flows, the campaign | `None` | Passing `ACTOR_REQUEST_HEADROOM` |
| `launch_pacing` | `ray_cluster`, both fill flows, the campaign | `False` | Passing `True` |
| `actor_request_batch_size` | `InferenceConfig`, both fill flows, the campaign | 50 | Passing a smaller number |

The headroom is the primary change and stands alone. Launch pacing is complementary and
secondary: it makes the requests that remain cost less quota.

### On `actor_request_batch_size` as immediate relief

Config, not code, so it can be applied as a dispatch parameter without waiting for a
release. The arithmetic, against today's unmodified Ray defaults:

A batch of B actors is B instances, which the autoscaler splits into `ceil(B / 5)`
`RunInstances` calls, drained by 2 launcher threads per cluster. At the current
B = 50 that is **10 calls per cluster per placement round** — 50 simultaneous calls
across five clusters, against a bucket holding five tokens and refilling two per second.
The burst is an order of magnitude past what the bucket can absorb.

**Recommended: 10.** That is 2 calls per cluster per round, 10 across five clusters — a
burst the bucket drains in about 2.5 seconds, inside one refill window. It is the
largest batch whose five-cluster burst the quota can actually absorb. Smaller buys
little more and slows the ramp; larger puts the burst back.

This is arithmetic from the quota, not a measurement, and it is stated as such. Note
also that it interacts with the other two levers: once `launch_pacing` raises Ray's own
per-call instance count, a batch of 50 costs only 2 calls and the config can go back up.
And once `actor_request_headroom` is on, a batch larger than the headroom is inert,
because the headroom is the tighter bound.
