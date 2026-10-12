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
someone judge whether an area is usable belong here. `registry_rebuild.py` re-derives a row's
measurements from the store when a fill left them null, and `registry_master.py` merges the
campaign's parts and those rebuilt rows into the `master/` view consumers read. Both sit outside
the inference code identity, so changing them never invalidates staged work.

## The rest

`zarr_store.py` is the largest file and holds the general read and write operations. It is
mostly rules about what makes a write safe: matching grids, no duplicate dates, no two
writers on one dataset. `manifest.py` refuses a write whose settings do not match the
dataset's.

`empty_store.py` creates a dataset with the right shape but no values in it, and
`global_store.py` does that for all 120 groups at once, so filling never resizes anything.
`shard_writer.py` then fills one group-year in parallel and `region_writes.py` works out where a
block of data belongs. What stops two writers overwriting each other is the conflict check at
commit time; `session_catch_up.py` only shortens the work that check has to do.

`campaign.py` tracks what is finished and clears out old versions, `published_store.py`
checks a finished dataset from outside, and `object_store.py` deletes what we no longer need.
`time_axis.py` and `icechunk_logging.py` are small shared helpers.
