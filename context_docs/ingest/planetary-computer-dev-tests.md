# Reading Sentinel-2 from Planetary Computer: dev tests, November 2026

Tests of the Planetary Computer read path (PR #279) on global-tessera-dev, 24–27 November 2026,
before any year is routed to it. Every run used the S2 ROI ingest flow with `provider` and
`collection` named, on ARM Fargate workers, against an Earth Search control run at the same time
where one applies. All test data has been deleted.

## Summary

- **Pixels are right.** Planetary Computer's mosaics keep the same dates as Earth Search where
  both hold the same images, with no pixel empty in one mosaic and filled in the other. Values
  differ slightly because the two catalogues hold different ESA processing runs (§1).
- **With the NAT out of the way, one ingest is at most 2.7–3.6 times slower than Collection 1**
  (§8; two fleets shared the address, so a lone ingest may be slightly faster).
  Every request crosses a 160 ms round trip to Azure West Europe, against 1 ms to S3 in Oregon.
  The 3.7–4.6 of §2 was measured before NAT packet drops could be seen. On the standing
  `t4g.micro` NAT instances it is 40–50 times slower, because their connection table fills (§3).
- **Microsoft holds each client address to about 1.3 GB/s.** Past about three ingests per address
  each slows in proportion: twelve behind one address ran 12 times slower than Collection 1 (§9).
  At 60 ingests at once, the campaign needs 30–60 NAT addresses; the cheapest design measured is
  Oregon at one or two ingests per address, about $34,000–38,000 per global year against $9,000 from
  Earth Search (§10). Reading from Frankfurt costs about the same once transfer is counted.
- **The read token survives a long run**: a two-hour ingest renewed it several times with no
  failure and no slowdown (§4).
- **Several ingests behind one NAT address were refused by the token service** while each worker
  fetched its own token (§5). Sharing one token per Dask fleet removed that: six 60-worker
  ingests behind one address all started and read (§6).
- **At that load, some reads fail with no response at all**, and two of the six ingests failed
  when such a failure hit a file's header, which GDAL reports as an unrecognised format (§6). The
  failures are packets the NAT instance drops for exceeding EC2's packets-per-second allowance.
  With each source read retried, all six completed (§7); on a `c7gn.16xlarge` the drops stop (§8).

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

Paired runs at 60 workers each, started together. **Superseded by §8**: these ran before the NAT
instance's packet drops were counted, and §7 shows that size dropping packets at higher load.

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
  of range reads that got no HTTP status and 83 of range reads answered HTTP 503, almost all at
  17:01–17:02. Its own retries covered them. The single ingests of §2–§4, about three hours in all, retried nothing.
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

## 8. Speed with the NAT out of the way, on `c7gn.16xlarge`

25 November 2026, 19:20–21:00 UTC. The us-west-2b NAT instance, and for the last round also
us-west-2c's, raised to `c7gn.16xlarge`, with EC2's allowance counters sampled every minute. **No
round dropped a packet** (`pps_allowance_exceeded` stayed 0; peak 47,782 tracked connections). Times
are ingest only, from the first batch to the coverage summary; cluster start is excluded.

**One ingest each, paired** (Iowa and 35N ran at the same time, two Planetary Computer fleets on
one NAT, so a lone ingest may be slightly faster):

| Area and window | Collection 1 | Planetary Computer | Ratio | Dates kept, both |
|---|---|---|---|---|
| Iowa, July 2023 | 3.6 min | 13.0 min | 3.6× | 21 / 25 |
| Zone 35N, 2024-01-01..07 | 14.1 min | 38.3 min | 2.7× | 6 / 7 |

The coverage gate and the write slow by the same factor (Iowa 3.7× and 3.9×), so the cost is in
each read. From the NAT instance, a request to Azure West Europe takes a 160 ms round trip and a
new connection 0.65 s to its first byte, against 1 ms and 0.15 s to S3 in us-west-2; 8 MB arrives
in 2.9 s against 0.1 s. Planetary Computer's 512-pixel blocks, against Earth Search's 1,024, also
mean more requests per read.

**Six at once**, each one month of Iowa in 2023, with six Collection 1 controls for the same months
alongside (Earth Search reads do not cross the NAT):

| Planetary Computer ingests per address | Ratio to Collection 1, per month | Through the NAT instances |
|---|---|---|
| six, behind one address | 5.2–6.8× | 1.25 GB/s |
| three, behind each of two addresses | 3.9–4.6× | 1.8 GB/s in total |

All 18 runs completed with the same dates kept as their controls (within one date, where the
catalogues differ) and no write retried; 26 source reads in the one-address round succeeded on a
second try. With no packet dropped and no NAT allowance near its limit, the extra slowdown is
beyond our NAT: something on Microsoft's side, or the path to it, caps each client address.
Spreading fleets over more addresses recovers most of it.

Open before campaign scale: the April stall of §7. §9 settles how many fleets each address should
carry; a managed gateway is ruled out on price (§10).

## 9. How many ingests per address, and reading from Frankfurt

27 November 2026. All three NAT instances `c7gn.16xlarge`, no packet dropped in any round. One-month
Iowa 2023 ingests, each Dask cluster pinned to a subnet, with a Collection 1 control for the same
month in each round except the last, which uses the two-per-address round's controls. Ingest time is
from the catalogue query to the coverage summary.

| Ingests per address | Ratio to Collection 1 (February, April, November) | Mean over every month run |
|---|---|---|
| one, over three addresses | 3.2–4.0× | 3.5× |
| two, over three addresses | 3.4–4.6× | 4.0× (six months) |
| three, over two addresses (§8) | 4.1–4.2× | 4.3× |
| six, behind one address (§8) | 5.3–5.6× | 5.9× |
| twelve, behind one address | 11.1–13.3× | 12.4× (six months) |

With twelve behind one address the NAT instance carried 1.2–1.4 GB/s throughout, the same total as
with six. **Microsoft holds each client address to about 1.3 GB/s**, so past about three ingests
each one gets an equal share of that and slows in proportion. Below that, the round trip sets the
pace: one ingest alone is still 3.5 times slower than Collection 1.

**Read benchmark.** One `c7g.4xlarge` in each region, with its own public address, reading random
1,024-pixel windows of 10 m red-band files from Iowa, July 2023
(`temp/pc-s2-dev/bench/bench.py`, not committed):

| | Oregon | Frankfurt |
|---|---|---|
| Round trip to Planetary Computer (TCP connect) | 155 ms | 9 ms |
| Planetary Computer, 1 thread | 2.4 windows/s | 10.1 windows/s |
| Planetary Computer, 96 threads | 158 windows/s, 0.53 GB/s | 382 windows/s, 1.28 GB/s |
| Collection 1 (S3 in Oregon), 1 thread | 11.3 windows/s | 3.6 windows/s |

From Frankfurt one thread reads Planetary Computer about as fast as Oregon reads Collection 1, and
96 threads reach the same 1.3 GB/s per-address ceiling. Multi-threaded Collection 1 reads hung on a
read with no timeout and were not measured.

## 10. What each design costs

Per global year of imagery read from Planetary Computer, with 60 ingests running at once. The basis
is the campaign cost model's Graviton Fargate line ([campaign cost model](../campaign/campaign-cost-model.md)
§13): about $9,000 of Sentinel-2 ingest per global year at Earth Search speed, which is about 780
fleet-hours. Fargate time scales with the slowdown; NAT instance hours are fleet-hours ÷ ingests per
address. Prices are us-west-2 and eu-central-1 on-demand from the AWS pricing API.

| Design | Addresses | Slowdown | Fargate | NAT instances | Transfer | Total |
|---|---|---|---|---|---|---|
| Earth Search (reference) | — | 1× | $9,000 | — | — | $9,000 |
| Oregon, today's three addresses (20 per address) | 3 | ~20× (from the ceiling) | ~$180,000 | ~$3,000 | — | ~$183,000 |
| Oregon, six per address | 10 | 5.9× | $53,000 | $3,000 | — | $56,000 |
| Oregon, two per address | 30 | 4.0× | $36,000 | $1,600 | — | $37,600 |
| **Oregon, one per address** | **60** | **3.5×** | **$31,500** | **$2,700** | — | **$34,200** |
| Frankfurt, one per address, mosaics written to Oregon | 60 | ~1.2× (estimated) | $12,400 (Frankfurt Fargate is 15% dearer) | $900 | $25,600 (1,280 TB at $20/TB) | ~$39,000 |
| Managed NAT gateway, Oregon | — | as above | as above | — | ~$45/TB processed | ruled out |

- **Addresses, not region, decide the cost.** Going from three addresses to thirty or sixty saves
  about $145,000 per global year for about $2,000 of NAT instances.
- **Oregon at one or two per address is the cheapest design measured.** Frankfurt removes most of the
  Fargate time but pays it back in transfer, needs the same number of addresses, and needs an ingest
  deployment in another region. Its estimated slowdown and its cross-region mosaic writes are not
  measured. Frankfurt finishes ingest about three times sooner, which matters only if ingest, not
  the graphics-card fleet, sets the campaign's pace.
- **NAT instance sizes are assumed**: `c7gn.4xlarge` ($1.00/h) for one or two ingests, `c7gn.8xlarge`
  for three, `c7gn.16xlarge` for six or more. Six ingests dropped packets on a `c7gn.8xlarge` (§7),
  so a smaller size needs its packet counters checked before a campaign relies on it.
- **Giving each worker its own public address** instead of a NAT would remove the shared address for
  about $0.005 per worker-hour, but takes workers out of the isolated network design.
