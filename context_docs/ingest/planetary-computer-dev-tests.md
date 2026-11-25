# Reading Sentinel-2 from Planetary Computer: dev tests, November 2026

Tests of the Planetary Computer read path (PR #279) on global-tessera-dev, 24–25 November 2026,
before any year is routed to it. Every run used the S2 ROI ingest flow with `provider` and
`collection` named, on ARM Fargate workers, against an Earth Search control run at the same time
where one applies. All test data has been deleted.

## Summary

- **Pixels are right.** Planetary Computer's mosaics keep the same dates as Earth Search where
  both hold the same images, with no pixel empty in one mosaic and filled in the other. Values
  differ slightly because the two catalogues hold different ESA processing runs (§1).
- **A single ingest is 3.7–4.6 times slower than Collection 1**, with no failures, once the NAT
  instances are large enough (§2). On the standing `t4g.micro` NAT instances it is 40–50 times
  slower, because their connection-tracking table fills (§3).
- **The read token survives a long run**: a two-hour ingest renewed it several times with no
  failure and no slowdown (§4).
- **Several ingests behind one NAT address were refused by the token service** while each worker
  fetched its own token (§5). Sharing one token per Dask fleet removed that: six 60-worker
  ingests behind one address all started and read (§6).
- **At that load, some reads fail with no response at all**, and two of the six ingests failed
  when such a failure hit a file's header, which GDAL reports as an unrecognised format (§6). The
  failures are packets the NAT instance drops for exceeding EC2's packets-per-second allowance.
  With each source read retried, all six completed (§7).

## 1. Pixel agreement

Tiny ROI `tiny_epsg5070.zarr`, 4 workers. Ratio is the median of Planetary Computer ÷ control
over pixels whose scene classification is vegetation, bare soil or water in both mosaics.

| Window | Control | Dates kept (control / PC) | Median ratio, per date | SCL agreement |
|---|---|---|---|---|
| July 2023 | Collection 1 | 12 / 12, same dates | 1.000–1.006 on dates at the same baseline (05.09); up to 1.015, and 0.925 in blue on one hazy date, where PC holds ESA's 2024 reprocessing (05.10) | 98.0–99.9% |
| July 2018 | legacy | 9 / 10 | 0.86–1.08; PC is ESA baseline 02.12 processed by Esri, the legacy collection is Earth Search's own Sen2Cor runs | 88–100% |
| February 2018 | legacy | 4 / 7 (snow: too few clear pixels for ratios) | — | 92–100% |

In every pair, no pixel is empty in one mosaic while filled in the other, including on the 2018
files that declare no nodata of their own. The re-keying in `stac._rekey_assets` declares it.

## 2. Speed of one ingest, on `c7gn.8xlarge` NAT instances

Paired runs at 60 workers each, started together.

| Area and window | Collection 1 | Planetary Computer | Ratio |
|---|---|---|---|
| Iowa, July 2023 (21 of 25 dates kept by both) | 3.4 min | 15.7 min | 4.6× |
| Zone 35N, 2024-01-01..07 (6 of 7 dates kept by both) | 12.5 min | 45.8 min | 3.7× |

Planetary Computer's write stage took 290–530 s per 35N date against 85–120 s, and its
coverage gate (the scene-classification read) 250–390 s against 80–95 s. Earth Search reads
stay inside AWS on the S3 gateway endpoint; every Planetary Computer byte crosses one NAT
instance to Azure West Europe. One ingest's traffic held at 330–430 MB/s (about 3 Gbps) through
its NAT instance, whose CPU stayed under 3%, so the NAT instance was not the limit at this size.
Two ingests behind two different addresses each reached that rate at the same time.

## 3. The standing NAT instances

The dev account's NAT instances are `t4g.micro`, as is production's (arbol-infra
`isolated-vpc-infra`). On them the Iowa run kept 2 dates in 40 minutes, at 410–560 s per date
with repeated warp failures, and was cancelled. The instance carrying it held 7,678 tracked
connections against a maximum of 7,680, so new connections were being dropped. That maximum
scales with memory. On `c7gn.8xlarge` with the limit raised by hand, a single ingest peaked at
20,894 tracked connections in Iowa and 28,764 in 35N. Reading from Planetary Computer at any
scale needs larger NAT instances for the duration (arbol-infra PR #88 holds the change).

## 4. A long run

Iowa, all of 2018, 60 workers: 280 dates found, 188 kept, 2 h 10 min, no failures or retries.
First hour 40 s per date, the rest 37 s per date. The read token, which lasts about 45 minutes,
was renewed several times under load.

## 5. Several ingests behind one address

Each Dask cluster runs in one subnet, so all its traffic leaves through one NAT instance and
one public address.

- Two 60-worker ingests behind one address: both refused at start with HTTP 429 from the token
  service after five plain exponential retries; two others, alone behind their addresses,
  finished.
- Three 60-worker ingests behind each of two addresses, with ten retries jittered over about
  ten minutes: all six refused.

At the time each worker asked for its own token when it first read and every ~35 minutes after,
so a fleet's start was a burst of one request per worker. The service tolerated one 60-worker
fleet per address and not 120 or 180 workers, and its refusals outlasted any wait an ingest could
afford. A campaign puts about 400 workers behind each address, so a fleet now shares one token
(`auth.sign_azure_href`; [authentication](../../docs/ingest/authentication.md)).

## 6. Six ingests behind one address, sharing a token per fleet

25 November 2026, 16:43 UTC. Six 60-worker ingests (360 workers), each one month of Iowa in 2023,
every Dask cluster pinned to the us-west-2b subnet by overriding `PRIVATE_SUBNETS` per run, so
all traffic left through one `c7gn.8xlarge` NAT instance and one address. The other two zones'
NAT instances stayed `t4g.micro`. The control is §5: two or three fleets per address, each worker
with its own token, were all refused.

| Month | Outcome | Duration | Dates kept |
|---|---|---|---|
| February | completed | 22.1 min | 13 / 21 |
| March | **failed**, 16 min in | 15.8 min | — |
| April | completed | 25.6 min | 19 / 23 |
| May | completed (one batch retried date by date) | 29.9 min | 19 / 25 |
| November | **failed**, 21 min in | 21.0 min | — |
| December | completed | 21.8 min | 14 / 23 |

- **No token was refused.** Every ingest started and read, with no token-service error logged.
- **Each ingest took about 1.3–1.65 times as long per date found as one alone, or 1.6–2.0 times
  per date kept.** Durations are end to end, cluster start included; on that basis the single Iowa
  run of §2 (15.7 minutes of ingest) took 18.1 minutes for 25 dates found and 21 kept. The NAT
  instance carried 1.0–1.5 GB/s in five-minute averages, about 170–240 MB/s per ingest against
  330–430 alone. It tracked at most about 40,000 connections, its CPU stayed idle, and its
  connection table recorded no failed inserts or drops.
- **Some reads got no response at all.** In the 35 minutes from 16:40, GDAL logged 8,214 retries
  of range reads that got no HTTP status and 83 of reads answered HTTP 503, almost all at
  17:01–17:02. Its own
  retries covered them. The single ingests of §2–§4, about three hours in all, retried nothing.
- **Two ingests failed on a file GDAL could not open.** The March and November runs each hit a
  file reported as "not recognized as being in a supported file format" on every retry within a
  minute, with a token valid for another half hour. Read again afterwards, the March file is a
  sound 244 MB TIFF. Each of the write's three attempts failed on a different file. The cause is
  GDAL's: a connection lost while it reads a file's header is retried once, logged nowhere and
  reported as an unrecognised format ([source-read failures, Cause 13](source-read-failures.md)).
  The reader now retries each source in place.

## 7. The same six ingests, with the per-source retry

25 November 2026, 18:12 UTC, the same months and pinning as §6, with each source read retried in
place and EC2's per-instance allowance counters sampled on the NAT instance every minute.

| Month | Outcome | Duration | Dates kept |
|---|---|---|---|
| February | completed | 19.8 min | 13 / 21 |
| March | completed | 23.0 min | 16 / 24 |
| April | completed | 46.3 min, one write stalled 13 min | 19 / 23 |
| May | completed | 22.7 min | 19 / 25 |
| November | completed | 24.2 min | 19 / 24 |
| December | completed | 19.4 min | 14 / 23 |

- **All six completed, and no write was retried.** 32 source reads failed and each succeeded on its
  second try; none was given up.
- **The NAT instance dropped 15.2 million packets** for exceeding its packets-per-second
  allowance (`pps_allowance_exceeded`), up to 2.2 million a minute from 18:16 to 18:34, and none
  after the heavy reading ended. Its bandwidth and connection allowances were never exceeded, and
  its Linux connection table, which peaked at 43,844, dropped nothing. These drops are the lost
  connections of §6 ([source-read failures, Cause 13](source-read-failures.md)).
- **One April write took 13 minutes** (788 s against 220–500 s for its others), after the drops
  had stopped. All six read threads on one worker waited on a lock inside GDAL's `reproject` with
  no traffic, then the write finished. Its cause is not known.

Open before campaign scale: sizing NAT egress for packet rate (a larger instance, more NAT
instances and addresses, or a managed NAT gateway), and the April stall.
