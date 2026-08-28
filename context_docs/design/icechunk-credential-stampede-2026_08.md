# The 28 August 2026 storage-credential incident

**Audience:** anyone who wants to understand this incident, including people who have never worked
on this pipeline. Terms are defined where they first appear. Code locations are collected in an
appendix at the end so the explanation does not depend on them.

**One-paragraph summary.** For about an hour and three quarters on the morning of 28 August 2026,
Amazon's storage service rejected a large number of our write requests, saying the requests were not
correctly signed. Four units of work lost their output and had to be redone; no data was corrupted
and nothing was permanently lost, so the cost was time rather than data. We can prove the problem
was confined to one of the several software libraries we use to talk to storage, which rules out a
general fault at Amazon. **We have not established the root cause.** We did find and fix a related
configuration mistake of our own, and we added logging that will identify the cause if it recurs.

---

## 1. Background: what has to go right for one write to succeed

The pipeline builds a global dataset. It runs on the order of four thousand small machines at once,
and they all write their results into cloud storage — Amazon S3.

Three things in that sentence matter for this incident.

**Credentials.** Nothing writes to S3 without proving who it is. The proof is a set of three short
strings: an **access key** (a public identifier, safe to write in a log), a **secret key**, and a
**session token**. They are issued together, they expire after a set period, and they must be used
as a matching set. Our processes fetch fresh ones automatically as the old ones lapse.

**Request signing.** Every request to S3 is signed: the sender combines the request with its secret
key to produce a signature, and Amazon recomputes that signature to check it. If the two do not
match, Amazon rejects the request with the error **`SignatureDoesNotMatch`**. Importantly, that
error does not mean "you lack permission" — it means the arithmetic did not agree, which for
automatically-issued credentials usually means the three strings were not a matching set.

**Several libraries, one destination.** Different parts of the pipeline reach S3 through different
software libraries. Four are relevant here: **Icechunk** (the library that stores the actual
dataset), and **boto3**, **s3fs** and **GDAL** (used for reading source imagery and other
supporting files). They all talk to the same storage service, in the same processes, at the same
time — which turns out to be the key to interpreting what happened.

---

## 2. What happened

Between 07:04 and 08:49 UTC, the fleet took **227 refused writes**, each reported as
`SignatureDoesNotMatch`. Four
units of work — each one a region-and-year combination — lost their output and will be redone
automatically. Their inputs were untouched, so the loss was compute time.

### The decisive observation

**Every one came from Icechunk, and specifically from Icechunk writing data.**
None came from the other three libraries, which were all working hard throughout the same period.

| library | rejections in the window |
|---|---:|
| **Icechunk** (dataset writes) | **227** |
| boto3 | 0 |
| s3fs | 0 |
| GDAL | 0 |

This is what rules out a general fault at Amazon. A signing or validation problem in Amazon's
storage service would not single out one library's writes while sparing three other libraries
running in the same processes against the same service.

> **How the count was reached, and two numbers withdrawn.** 227 is the number of log lines naming a
> refused chunk write (`write_chunk`, and `store::set`, which agree exactly), and those came from
> **210 separate machines**. Counting more broadly, 805 lines mention the error at all — each
> failure emits several lines of context, and this log group sometimes writes a line twice — and
> those span 272 machines. **The two figures pair with different populations**, so quoting 227
> against 272, as an earlier version did, is arithmetically impossible.
> **An earlier version of this document said 516 rejections. That figure was the non-event
> remainder — the context lines — not a count of failures.**
> It also credited GDAL with one incidental rejection. **There were none:** the line matched a
> search for "gdal" only because that substring appears by chance inside an S3 extended request id
> (`...HoEH7GtwggdalSnjlC4Q...`). The line was itself an Icechunk chunk write.

### What else was ruled out

Four other explanations were checked directly and all came back at zero for the same period.

| explanation | how it would appear | count |
|---|---|---:|
| Our credentials had expired | `ExpiredToken`, `TokenRefreshRequired` | 0 |
| We used a credential Amazon did not recognise | `InvalidAccessKeyId`, `InvalidClientTokenId` | 0 |
| Our machines' clocks had drifted | `RequestTimeTooSkewed` | 0 |
| Amazon was rate-limiting us | `SlowDown`, `ThrottlingException` | 0 |
| — | **`SignatureDoesNotMatch`** | **227** |

So: valid, unexpired credentials; identifiers Amazon recognised; no complaint about our clocks; no
rate-limiting; a single error type; spread across 272 separate machines; and it stopped on its own.

---

## 3. Why two separate accounts failed at the same moment

This was the most puzzling feature of the incident, and the part most likely to mislead.

The failures spanned **two storage buckets in two different AWS accounts** — one of ours, and one
belonging to a delivery partner. Those two destinations use *completely different credentials*, from
different sources. Two independent credential systems failing in the same window would be a
remarkable coincidence.

It is not a coincidence, because **both destinations are written through the same layer.** They are
reached by different application code — the standalone store by one assembly entry point, the
published store by another — but both hand the work to the same storage library, and both obtain
their credentials through the same single function in our code, which serves whichever role the
destination requires. The credential *source* differs per destination; the credential *delivery
mechanism* does not.

That means a defect in the shared mechanism explains simultaneous failures in two accounts with no
Amazon-side event required. This is the strongest structural conclusion the incident supports, and
it is why the next section is worth reading.

---

## 4. What we found in the storage library

Icechunk lets an application supply a small function that hands over a fresh credential whenever one
is needed. Icechunk is supposed to hold on to the credential and reuse it until it expires.

There is an open bug report against Icechunk — [issue #2077][2077], filed 15 April 2026 — showing
that this holding-on works **only once**: because of how the caching code is written, the stored
credential can never be replaced after the first one expires. The report describes Icechunk then
asking the application for a credential on **every storage request** for the remaining life of the
process, and its author measured **301,000 credential requests in four hours** from a single
workload.

> **We have not verified that frequency for our own workload, and it is disputed.** Review here
> argued that Icechunk's S3 client caches the credential between requests, which would make the
> re-fetching happen per *deserialisation* of a session — often, in a workload that ships a session
> per task, but not once per request inside one long-lived worker like ours. Settling it needs a
> reading from inside a running assembly, which the instrumentation shipped alongside this change
> can now provide. **What the flag does is not in doubt** — it removes the credential fetch a freshly
> deserialised session would otherwise make — only how much that saves.

An Icechunk maintainer replied that this "sounds like intended behavior," so it is disputed as a bug
and any mitigation has to be ours.

Icechunk does offer a setting that reduces the damage: it can be told to fetch a credential
immediately and keep it, so that copies of the connection sent to worker processes start with a
credential in hand rather than asking for one on their first request.

[2077]: https://github.com/earth-mover/icechunk/issues/2077

---

## 5. What was wrong on our side, and what we changed

The final stage of the pipeline assembles finished results and writes them into the dataset. It does
this by splitting the work across sixteen separate worker processes, each of which receives a copy
of the storage connection.

**We have two code paths that do this**, one for each of two kinds of destination. They should have
been configured identically. They were not:

| code path | destination | setting before | setting after |
|---|---|---|---|
| assembling into a standalone store | an interim dataset | **on** | on |
| assembling into the main dataset | **the published dataset** | **off** | **on** |

The second path is the one the production campaign actually uses, and it was the one with the
setting switched off — so each of its sixteen worker processes started with no cached credential and
had to fetch one before it could write. **How often they then kept asking is the open question of
section 4**, not something measured here; what is certain is that the fetch a freshly deserialised
session makes was not being avoided.

The setting is now switched on for both paths. It is switched on **explicitly at each of these two
places rather than made the default**, because it has a real cost, explained next, and about ten
other places in the code open the same dataset without ever copying the connection to a worker; for
those, switching it on would pay the cost for no benefit.

> **A correction, in case you see the earlier version quoted.** While investigating I stated that
> *no* part of our code switched this setting on. That was wrong — the first of the two paths above
> always had. I had checked the callers of one function and missed that a sibling path reaches the
> same setting through a differently-named one. The real defect was an inconsistency between two
> nearly identical paths, which is a sharper finding than the one I reported.

### The cost of the setting, and why it is applied per location

Icechunk's own documentation warns that with this setting on, the credential is stored inside the
connection object — and "they can be sent over the network if you pickle the session/repo."
("Pickling" is Python's term for turning an object into bytes so it can be sent somewhere else.) In
other words, switching this on puts a live secret key and session token inside something that gets
copied elsewhere. Where that copy travels is the whole question:

| how the copy travels | exposure |
|---|---|
| **to a worker process on the same machine** (what both changed paths do) | The copy goes through an operating-system pipe to a child of the same process, on the same machine, which already holds the credential in memory. **No new exposure.** |
| to a worker on another machine, over the network | A live secret in transit — and those systems can also spill data to disk under memory pressure, leaving a credential written down. |

Both changed paths are the first kind. Making the setting a global default is the tempting shortcut
and it is the one that would put secrets on the network for paths that gain nothing from it.

---

## 6. What this change does and does not do

Stated plainly, because the fix is easy to over-read.

- **It does not risk credentials going stale.** This was the specific concern raised. Icechunk's
  documentation is explicit that once the stored credential expires, the stored copy is no longer
  used and the live fetch takes over. The fallback is automatic and expiry-aware.
- **It only helps for one credential lifetime, and that is shorter than it sounds.** The task-role
  path promises fifteen minutes; **the published-store path promises five** (its interval is
  deliberately short so that a callback lands inside the credential library's mandatory-refresh
  window). Against an assembly of about three and a half hours, that is roughly 2% of the run for
  the destination this incident actually concerned. Because of the Icechunk bug above, giving credentials a longer life
  would delay this rather than prevent it. **This is a consistency fix and a reduction in wasted
  work — it is not a cure for a long-running job**, and I described it as one before reading the
  documentation carefully.
- **It is not the established cause of the incident.** See the next section.

---

## 7. Why the "too many credential requests" theory is not the leading explanation

Ranking this first was an overreach on my part, and the reason is worth recording.

The 301,000 figure in the Icechunk bug report came from an application whose credential function
made a network call every time it was asked. **Ours usually does not** — it reads an
already-fetched credential out of a small in-memory cache, with refreshes happening quietly in the
background, so the cost of being asked repeatedly is normally acquiring a lock inside the process
rather than a network round trip. **Two exceptions matter and an earlier version of this document
missed them:** the very first call in a process performs the role assumption, and a call landing in
the credential library's mandatory-refresh window blocks on a real token-service request. Those are
a handful of calls per process, not one per storage operation. **Lock contention makes things slow;
it does not produce incorrectly signed requests**, which is the reason this is not the leading
explanation.

Three candidate explanations remain, and none is proven:

1. **A defect in Icechunk's own request-signing code under heavy parallel use.** Icechunk allows a
   large number of storage operations in flight at once, multiplied by sixteen worker processes,
   multiplied by ten simultaneous regions. This campaign puts far more pressure on that code than
   the workload in the bug report did.
2. **A race in the handoff of credentials** between the part that refreshes them and the part that
   attaches them to requests. The new logging (next section) tests this one directly.
3. **A transient fault on Amazon's side.** Weakest, for the reason in section 2 — but see the
   follow-up reading below, which has since strengthened it.

### A later reading that shifts the ranking

At 16:20 UTC the same day, the campaign was running **five** simultaneous assemblies rather than the
four it had during the incident, and — because of a separate performance fix — each was using
sixteen worker processes rather than five. Storage write concurrency was therefore **higher than
during the incident**. Across the following hour and 1,041,266 log lines there were **zero**
`SignatureDoesNotMatch` rejections.

That is real evidence against explanation 1 and for explanation 3. One hour is only one hour, so it
is not conclusive, but it is the first independent measurement that separates the candidates.

---

## 8. A note on the new logging's own concurrency

Worth recording because the same mistake was made three times over, in a small piece of code, and
review caught every one of them. The pattern is a useful one to recognise: **each version tried to
say something about the order of events using information that did not record it.**

The new logging keeps a small note of which credentials a process has already reported, so that a
credential is announced once rather than on every request. The first version compared the incoming
credential against the last one recorded. The second compared expiry times, on the theory that a
later expiry meant a newer credential.

**Both were wrong, for the same reason: nothing available at that point in the code records when a
credential was obtained.** A thread can pick up the old credential, be paused by the operating
system while another thread fetches and records a newer one, and then arrive carrying the old
credential *with a later expiry stamped on it* — because the expiry is calculated after the
credential is picked up. Either version would then report a credential change backwards and another
one forwards: three change records for one real change, corrupting exactly the signal the logging
exists to provide.

A third version kept the "have I mentioned this one yet" test but still reported each new
credential as a change *from* the previous one. Review found that this breaks in the one case the
first two did not cover: at the very start of a process there is no history to match against, so a
straggler carrying the old credential is treated as new, the record reads "changed from the new
credential to the old one", and the superseded credential is then left standing as the apparent
current one — mislabelling every later use of the real credential. This case is not hypothetical on
the path we changed: a worker process that inherits a credential only calls the credential function
once that credential expires, which is exactly the moment a refresh can interleave.

The fix was to stop making ordering claims of any kind. Each record now says only "this process
began using credential X", never "X replaced Y". Read in sequence order for one process, those
records *are* its credential history — which is what can honestly be established. The worst a
straggler can now cost is one extra line naming a credential the process genuinely did use, and no
record can describe the current credential wrongly.

Ordering of the log lines themselves is handled separately by a counter stamped at the moment the
note is updated. That counter orders the moments the code *decided* to hand over a credential — it
recovers the sequence when two lines reach the log out of order. It is not proof of the order the
credentials were *used*, because a thread can be paused between taking its number and returning.

The reviewer's alternative — hold a lock across the credential fetch — was declined: that fetch can
block on a network call, so holding a lock across it would put every storage request in the process
behind a network round trip. That is a worse problem than the one being fixed.

---

## 9. What would settle the root cause

Alongside the fix, every process now records **which credential it is using**: the access key (a
public identifier, and the one Amazon's own audit log indexes on), the expiry, the process id, and a
sequence number. The first credential a process uses and every subsequent change are recorded at
normal log level; routine repeats are recorded at debug level so they do not flood the logs. The
secret key and session token are **never** logged.

If this recurs, that record answers the one question the current evidence cannot — whether the
credential in use changed at the moment of failure:

- **One unchanged credential across the failure** rules out the handoff — candidate 2 — and means no
  retry could have helped. It does **not** choose between the other two: a signing or
  canonicalisation defect in the library and a transient fault on Amazon's side both survive an
  unchanged key, and separating those needs the request ids taken to AWS support.
- **A different credential appearing moments before the first rejection** implicates the handoff.
  Candidate 2 — and only
  then does adding a retry make sense, with randomised delays, because sixteen workers sharing a
  fixed delay would retry in synchronised bursts.

Until one of those is observed, **do not add retries here.** A retry was drafted during this
investigation on the belief that these writes had none, and **withdrawn**: they already retry up to
ten times with increasing delays, and these rejections exhausted that allowance rather than arriving
unprotected. A second layer of retries on top would have added delay and hidden the signal.

---

## Appendix A: where this lives in the code

| thing described above | where |
|---|---|
| the single place every storage credential is built and logged | `providers/aws/credentials.py`, `_serve_icechunk_credential` |
| the two credential sources (our account; the partner's) | same file: `iam_icechunk_credentials`, `AssumedRoleIcechunkCredentials` |
| the setting discussed in section 5 | `scatter_initial_credentials`, threaded through `storage/zarr_store.py` |
| the two assembly paths that copy a connection to workers | `inference/assembly.py`, `assemble` and `assemble_global` |
| the sixteen-worker split | `storage/shard_writer.py`, `run_forked` and `write_year_shards` |
| the existing retry allowance | `storage/zarr_store.py`, `StorageRetriesSettings` |

## Appendix B: versions

We run Icechunk **2.1.1**. The only later release is **2.1.2** (29 July 2026), and its release notes
contain no change to the credential-caching code, so upgrading does not address issue #2077. That
issue has no linked fix.
