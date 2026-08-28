# The 516 signature refusals, and the one credential mechanism both accounts share

**2026-08-28.** Four zone-years lost their assemblies to `SignatureDoesNotMatch` refusals from
S3. This records what is established, what is ruled out, what changed as a result, and — because
three of my own intermediate claims here were wrong — which of them to distrust if you find them
quoted elsewhere.

**The root cause is not proven.** Read the last section before acting on this.

## What happened

Between 07:04:28 and 08:48:21Z the fleet took **516 `SignatureDoesNotMatch` refusals**. Cells
`47N-2018`, `35N-2017`, `35N-2018` and `47N-2019` lost their assemblies. Staged tiles survived, so
the cost was wall clock, not data.

**Every refusal was an icechunk chunk write.** 227 events name `icechunk`, `write_chunk` and
`store::set` — all three counts identical — and the remainder are their continuation lines. In the
same window, across clients that were all busy throughout:

| client | auth failures |
|---|---:|
| icechunk | **516** |
| boto3 | 0 |
| s3fs / fsspec | 0 |
| GDAL | 1 (incidental) |

That asymmetry is what rules out a service-side signing fault. An AWS problem in us-west-2 would
not spare boto3, s3fs and GDAL while hitting one client's writes.

**And these absences rule out expiry, throttling and clock skew**, all checked over the same
window: `SlowDown` 0, `ThrottlingException` 0, `ExpiredToken` 0, `TokenRefreshRequired` 0,
`InvalidClientTokenId` 0, `RequestTimeTooSkewed` 0, `InvalidAccessKeyId` 0.

So: valid unexpired credentials, correct key ids, no clock complaint, no throttling, one single
error type, 272 distinct Fargate tasks, self-resolving.

## Why two accounts failed at once

This was the most puzzling part of the incident — the refusals span **two buckets in two AWS
accounts**, one written with the task role and one with an assumed cross-account role. A
coincidence across two independent credential sources would be remarkable.

It is not a coincidence, because **both destinations are written by the same code path.**
`GlobalAssembler.assemble_global` opens one store — ours or the partner-owned published one,
decided only by `store_path` — and hands the session to `write_year_shards`, which forks
`n_workers` spawned children. The credential *source* differs per destination; the credential
*delivery mechanism* is one piece of icechunk code shared by both.

So a defect in that mechanism produces simultaneous failures in two accounts with no AWS-side
event required. That is the strongest structural fact the incident yields, and it is what makes
the next section worth reading rather than a curiosity.

## The mechanism: the callback is invoked per request, forever

icechunk issue [#2077](https://github.com/earth-mover/icechunk/issues/2077), open since
2026-04-15: `PythonCredentialsFetcher<S3StaticCredentials>` holds `initial: Option<CredType>` as a
one-shot cache, and its `get()` takes `&self` rather than `&mut self`, so **it can never be
refilled once the first credential expires.** From that moment every S3 operation falls through to
the Python callback. The reporter measured **301,000 credential requests in four hours** from one
workload. An icechunk maintainer replied that this "sounds like intended behavior", so it is
disputed as a bug and any mitigation is ours.

`scatter_initial_credentials=True` asks icechunk to populate that `initial` cache eagerly, so a
deserialised child starts with a credential instead of calling back from its first request.

## What was actually wrong in our code, and what changed

**Our two fork paths disagreed.** Both pickle an icechunk session to spawned children:

| path | store | flag before | flag after |
|---|---|---|---|
| `assemble()` → `open_or_create_repo` | standalone zone-year store | **True** | True |
| `assemble_global()` → `open_global_repo` | **the global / published store** | *unset → False* | **True** |

The campaign runs the second one. `open_global_repo` did not accept the parameter at all, so the
sixteen children of every global assembly deserialised with an empty credential cache.

The fix adds the parameter to `open_global_repo` and sets it at that single call site. **It is set
unconditionally, not guarded on whether the caller supplied a credential callback** — `assemble()`
carried such a guard and it was wrong in the same direction: `_create_storage` falls back to
`_default_credentials_provider` when the argument is None, so the guard disabled the scatter on
exactly the fallback path, while `_create_storage` already omits the option when no provider exists
at all. Both call sites now state True. It is
**opt-in per call site, not defaulted**, because the flag has a cost (next section) and the ten
other `open_global_repo` callers read or commit in-process and never pickle anything — for them an
eagerly cached credential buys nothing.

> **Distrust this if you see it quoted:** I stated during the investigation that "no caller ever
> sets this flag." That was wrong. I had audited `open_global_repo`'s callers and missed that
> assembly's sibling path uses `open_or_create_repo`, where it was already set and already
> correct. The real defect was an inconsistency between two paths, not a blanket omission.

## The cost of the flag, and why it is set per call site

From icechunk's own docstring: "credentials obtained are stored, and **they can be sent over the
network if you pickle the session/repo**." With the flag set, the live secret access key and
session token sit inside the pickled `Storage`. Where that pickle travels is therefore the whole
question:

| transport | exposure |
|---|---|
| `run_forked` — `multiprocessing.get_context("spawn")` + `ProcessPoolExecutor` | a local OS pipe to a child of this process, on this host, whose parent already holds the credential in memory. **No new exposure.** |
| a Dask or Ray worker over the network | a live secret in flight, and those object stores can spill to disk under memory pressure — a credential at rest |

Both changed paths are the first kind. The blanket default is the tempting version and it is the
one that would put a secret on the wire for paths that gain nothing.

## Two more corrections to my own reasoning

**1. It cannot cause credentials to expire.** This was the specific worry that prompted the
research. icechunk's docstring: "After the initial set of credentials has expired, the cached
value is no longer used." The fallback to the live callback is unconditional and expiry-aware, so
a long session cannot be pinned to a stale credential.

**2. But it therefore only covers one TTL.** Our `expires_after` is 15 minutes, so a three-hour
assembly gets roughly 15 minutes of relief and 2h45m of the original per-request behaviour.
Because `initial` can never be refilled (#2077), a longer TTL delays the onset without preventing
it. **This is a consistency fix and a reduction in callback volume — it is not a fix for a
long-running session,** and I described it as one before reading the docstring properly.

## Why the stampede is NOT the leading explanation

Ranking this hypothesis first was an overreach, for a reason worth recording.

The reporter's 301,000 figure came from a callback that made a network request per call. **Ours
does not.** `_resolve_iam_credentials` is `lru_cache(maxsize=1)` and `get_frozen_credentials()` is
a lock and a return on an already-resolved botocore credential, with refresh happening in the
background. Our per-request cost is a GIL attach, not an HTTP round trip — and **GIL contention
produces latency, not malformed signatures.**

Three candidates remain, none proven:

1. **A defect in icechunk's Rust S3 signing under this concurrency.** icechunk's internal async
   concurrency defaults to 256 per repo, across sixteen forks per cell, across ten cells. This
   campaign applies more pressure to that code than the reporter's workload did.
2. **A race in the credential handoff**, which the instrumentation below discriminates directly.
3. **A service-side fault** — weakest, for the client-asymmetry reason above.

## A note on the instrumentation's own concurrency

Review found three defects in the first version of the credential boundary, all one shape: **it
tried to reason about ORDER using values that do not carry it.**

The boundary keeps per-process state so it can log a credential the first time it is used and stay
quiet afterwards. The first version compared the incoming access key against the last one recorded;
the second compared expiries as a generation marker. Both are wrong, because **acquisition happens
before the boundary is reached and outside its lock.** A thread can freeze the old credential, be
descheduled past a refresh, and arrive after a faster thread recorded the new one — carrying an old
key with a *later* expiry, since the expiry is computed after the freeze. Either version then
announced a rotation backwards and a second one forwards: three rotation records for one real
rotation, in the exact signal an incident has to be lined up against.

The resolution was to stop deciding order. The state is now the set of keys this process has
already announced, so a late callback carrying a superseded key is simply not novel and says
nothing. Ordering of the emitted records — which can still interleave, because the logging is
deliberately outside the lock — is carried by a sequence number stamped while the lock is held.

Serialising acquisition under the lock was the suggested alternative and was declined:
`get_frozen_credentials()` takes botocore's lock and can block on a refresh that makes an STS call,
so it would put every icechunk request in the process behind a network round trip — the same
serialisation the design exists to avoid.

## What would settle it

The instrumentation shipped alongside this change (`_serve_icechunk_credential`) records the
access-key id and expiry at the single point where every icechunk credential is constructed:
first serve at INFO, rotations at INFO, steady re-serves at DEBUG. On the next occurrence it
answers the one question this evidence cannot:

* **One steady key across the failure** → the credential never changed, no retry could ever have
  helped, and the cause is upstream of our credential path entirely. Candidate 1.
* **A rotation seconds before the first refusal** → candidate 2, and a retry on a *fresh* session
  becomes arguable — with jitter, because sixteen forks sharing a fixed backoff synchronise their
  bursts.

Until one of those is observed, do not add retries here. A retry premised on "assembly writes have
no retry" was drafted during this investigation and **withdrawn**: `StorageRetriesSettings(max_tries=10,
initial_backoff_ms=200, max_backoff_ms=30_000)` plus `operation_attempt_timeout_ms=180_000` already
wrap every chunk write, and the refusals exhausted that ladder rather than arriving unprotected.

## Version notes

We run icechunk **2.1.1**. The only later release is **2.1.2** (2026-07-29) and its notes carry no
credential-fetcher change, so upgrading does not address #2077, which has no linked fix PR.
