# A fresh worker's first minutes: the disk, not the compiler

Measured on 2026-10-20 and 21 in global-tessera-dev, on fresh `g5.2xlarge` boxes booted from the
`dev/global-tessera-pytorch-2-14` worker AMI, running v2 on PyTorch 2.14.1. Outcome: a new worker's first v2
forward waits about six minutes, nearly all of it its EBS volume loading the AMI's snapshot on first read,
and an ext4 image of the venv mounted from the instance store cuts that to under a minute at any fleet size.

## 1. Where a fresh worker's time goes

One process times each phase of a new machine's first v2 forward. Five more processes on the same box isolate
the causes: without bytecode written, with the compiler's cache wiped, and with the page cache dropped.

| phase | fresh disk | disk blocks already fetched | + bytecode present | + compiler cache warm |
|---|---:|---:|---:|---:|
| import torch | 104 s | 1.3 | 1.3 | 1.3 |
| import its compiler | 52 s | 6.2 | 2.0 | 2.0 |
| build the model | 64 s | 3.0 | 1.6 | 1.6 |
| first library calls | 23 s | 0.1 | 0.1 | 0.1 |
| first forward, which compiles a kernel | 109 s | 3.6 | 3.1 | 2.5 |
| **total** | **354 s** | **14.7 s** | **8.5 s** | **7.9 s** |

339 of the 354 s are the volume fetching blocks from the snapshot the first time each file is read: the
PyTorch libraries, the CUDA libraries and the compiler toolchain. The kernel compile itself is about 3 s.
Baking bytecode and a compiler cache into the AMI therefore saves about 7 s, and a compiler cache would have
to be built on a GPU of each type per AMI, so neither is worth doing. The same cost explains most of the 5% of
billed GPU-hours the October v2 full-stack run (eight `g6e.xlarge`, 394 tiles) spent outside its tiles,
by its `CHUNK_SUMMARY` lines and billing in `s3://global-tessera-inputs-dev/perf/_v2_fullstack/`: roughly
six minutes per new worker.

## 2. The remedies

| fresh worker, first v2 forward | getting the venv | first forward | total | at fleet scale |
|---|---:|---:|---:|---|
| today's AMI | — | 354 s | 354 s | — |
| read the 6.4 GB venv in parallel at boot | 202 s | 20 s | 222 s | yes |
| EBS provisioned volume-initialization rate, 300 MiB/s | — | 179 s | 179 s | yes, charged per GiB initialized |
| the venv as a tar on S3, unpacked onto the instance store | 11 s + 43 s unpack | 31 s | 85 s | yes |
| **the venv as an ext4 image with bytecode on S3, loop-mounted read-only** | **24 s, instant mount** | **25 s** | **~49 s** | **yes** |
| Fast Snapshot Restore on the AMI's snapshot | — | 36 s | 36 s | no |

**Fast Snapshot Restore is fastest but rationed.** AWS grants an 80 GiB snapshot 10 fast-restored volumes per
AZ, refilling at 10 per hour, at $0.75 per snapshot per AZ-hour, and creates any further volume the slow way
without an error. It suits dev testing, where the yield-embeddings `dev-test` skill switches it on and off,
not a fleet.

**The provisioned initialization rate halves the wait,** but the volume is still loading while the first
process runs: its progress counter had not moved when the process finished.

**The mounted image scales with S3.** The 8.5 GB image downloaded in 24 s, and mounting it is instant, where
unpacking a tar of the same venv took 43 s. Bytecode compiled into the image cannot go stale, because the
mount is read-only. What remains of its 25 s first forward is mostly the kernel's first compile on a cold
cache (12.7 s, against 2.4 s warm). The base interpreter and the driver libraries stay on the EBS volume.

## 3. How it is wired

The AMI build packs the venv into an image, uploads it under a unique name and writes that URI into the AMI
at `/opt/tessera/env-image.uri`, so every AMI names exactly the venv it was built with. `cluster.yaml.template`
then copies the image to the instance store and mounts it over `/opt/tessera/venv` before `ray start`. Any
miss leaves the AMI's own venv in place: an AMI without the file, an instance without an instance store at
`/opt/dlami/nvme`, a venv already mounted, or a copy that fails.

## 4. End to end on a built AMI

On 2026-10-21 the yield-embeddings AMI build with the packing step baked a `dev/global-tessera-env-image` AMI
in global-tessera-dev. It has PyTorch 2.5.1 and ran TE main's code, which compiles no kernel. Two fresh
`g5.2xlarge` boxes booted from it, under the Ray worker's instance profile:

| fresh worker, first v2 forward | getting the venv | first forward | total |
|---|---:|---:|---:|
| the volume's venv (control) | — | 194 s | 194 s |
| this template's `worker_setup_commands` entry, run as Ray's `ubuntu` user | 26 s | 14 s | **40 s** |

The command found the AMI's URI file, downloaded the 7.6 GiB image and mounted it read-only. Run a second time,
it returned at once and left one mount. The control is faster than §1's 354 s because PyTorch 2.5.1 without a
compile reads fewer files on first use than PyTorch 2.14.1 with one.

Harness and raw results: `s3://global-tessera-inputs-dev/perf/g7e/` (`warm_probe.py` for the phase timings,
`warm.sh` for the decomposition, `image_build.sh` and `image_test.sh` for the mounted image, `env_e2e.sh` and
`e2emount/`, `e2econtrol/` for §4).
