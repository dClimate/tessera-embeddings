# Storage

Everything that reads or writes the datasets on S3, so the rules about what a valid write
looks like live in one place rather than in each pipeline stage.

The data sits in Icechunk, which puts version control over Zarr arrays: a write opens a
session, changes things, and commits. Nothing is visible until the commit lands, so a job
that dies part way leaves nothing half-written.

## Start here if you are contributing

**`zone_grid.py`** defines the 120 groups the global dataset is divided into — 60 longitude
bands, north and south — and each one's map projection, extent and name. Changing the world's
coverage means changing this.

**`conventions.py`** writes the labels that travel with the data, so someone opening it can
tell what the numbers mean without asking us. A new metadata standard goes here.

**`registry.py`** builds the summary table published beside the data: one row per tile per
year saying whether it was filled and how good the coverage was. New columns that help
someone judge whether an area is usable belong here. Each fill writes one raw part per zone-year
to `parts/`; `registry_master.py` merges it into the zone-year's file in `master/`, the one
registry consumers read, tile by tile, so `master/` always describes what the store holds now.
`registry_rebuild.py` re-derives a row's measurements from the store when a fill left them null.
Both sit outside the inference code identity, so changing them never invalidates staged work. The
design is in
[`context_docs/storage/registry-master.md`](../../../context_docs/storage/registry-master.md).

## The rest

`zarr_store.py` is the largest file and holds the general read and write operations. It is
mostly rules about what makes a write safe: matching grids, no duplicate dates, no two
writers on one dataset. `manifest.py` refuses a write whose settings do not match the
dataset's.

`empty_store.py` creates a dataset with the right shape but no values in it, and
`global_store.py` does that for all 120 groups at once, so filling never resizes anything. It also
owns the store's minimum optical depth rule: stamped on the root at seeding, loosened (never
tightened) for the store or one zone by `loosen_optical_min_obs`, and read per zone through
`effective_optical_min_obs` ([ADR 028](../../../context_docs/decisions/028-embed-every-pixel-with-an-optical-observation.md)).
`shard_writer.py` then fills one group-year in parallel and `region_writes.py` works out where a
block of data belongs. What stops two writers overwriting each other is the conflict check at
commit time; `session_catch_up.py` only shortens the work that check has to do.

**`update.py`** is how a group-year that is already filled gets some of its tiles redone: a named,
write-once plan of which tiles, under which land mask, either filling only the store's holes or
rewriting tiles whole. It records each update beside the original fill in the group's `updates`
attr, never over it, and `shard_writer.py` merges a holes update into the tiles it touches
([ADR 027](../../../context_docs/decisions/027-targeted-updates.md)). It sits outside both code
identities, like `campaign.py`.

`campaign.py` tracks what is finished and clears out old versions, `published_store.py`
checks a finished dataset from outside, and `object_store.py` deletes what we no longer need.
`time_axis.py` and `icechunk_logging.py` are small shared helpers.
