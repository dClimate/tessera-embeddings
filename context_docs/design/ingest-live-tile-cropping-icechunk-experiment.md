# Icechunk multi-write-per-commit experiment

Question: can multiple `icechunk.xarray.to_icechunk` calls share ONE `writable_session` + ONE `commit` (N disjoint spatial windows of one date -> one snapshot), with dask-backed xarray data?

## Versions (repo venv `/Users/rbanick/dev/tessera-embeddings/.venv/bin/python`)

- icechunk 2.0.4
- zarr 3.2.1
- xarray 2026.4.0
- dask 2026.3.0
- numpy 2.4.4
- scheduler: default threaded local dask scheduler (no distributed cluster)

## Setup (all tests)

Local filesystem icechunk repo per test. Store: dims (time=3, northing=64, easting=64), chunks (1, 16, 16), data vars `emb` float32 (fill -999.0) + `count` uint16 (fill 0), datetime64[ns] time coord. Seeded all-fill via `to_icechunk(ds, session, mode="w")` + commit (ancestry = 2 snapshots: repo-init + seed). All window datasets are dask-backed (`ds.chunk({"time":1,"northing":16,"easting":16})`, verified `dask.array.core.Array`) with coords dropped, mirroring the repo's `_drop_region_coords`. Scripts: `common.py`, `test_a.py` ... `test_f_attrs.py` in this directory.

## TEST A — two r+ region writes, one session, one commit: **PASS**

```python
session = repo.writable_session("main")
to_icechunk(win1, session, mode="r+",
            region={"time": slice(1,2), "northing": slice(0,32), "easting": slice(0,64)},
            align_chunks=True, split_every=8)
to_icechunk(win2, session, mode="r+",
            region={"time": slice(1,2), "northing": slice(32,64), "easting": slice(0,64)},
            align_chunks=True, split_every=8)
session.commit("...")
```

- Snapshots: 2 before -> 3 after (exactly 1 new).
- Read-back: window 1 = 1.0/11, window 2 = 2.0/22, time slices 0 and 2 still all-fill. Both vars correct.

## TEST B — same spatial chunk row, adjacent chunk-aligned windows: **PASS**

Regions `northing 0:16` and `16:32` (same easting chunk columns 0:64), same call sequence as A. Snapshots 2 -> 3. No interference: window 1 = 3.0/33, window 2 = 4.0/44, `northing 32:64` of that date and the other two dates still fill.

## TEST C — 20 disjoint chunk-aligned 16x16 windows, one session, one commit: **PASS**

Loop of 20 `to_icechunk(..., mode="r+", region=..., align_chunks=True, split_every=8)` on one session, then one commit. Snapshots 2 -> 3. All 20 windows read back with their distinct values; untouched chunks still fill.

- Timing: 20 writes + 1 commit = **0.20 s total (~0.010 s/write)** on local FS with tiny chunks. No per-call slowdown observed.

## TEST D — append (mode="a") THEN region write into the new date, SAME session, one commit: **PASS (same-session works; fallback not needed)**

```python
session = repo.writable_session("main")
to_icechunk(new_date_allfill_with_coords, session, mode="a", append_dim="time", align_chunks=True)
# session sees its own uncommitted append:
#   zarr.open_group(session.store)["emb"].shape == (4, 64, 64)  <- already 4
to_icechunk(win, session, mode="r+",
            region={"time": slice(3,4), "northing": slice(0,32), "easting": slice(0,64)},
            align_chunks=True, split_every=8)
session.commit("...")
```

- Snapshots: 2 before -> 3 after (append + region = ONE snapshot).
- Known wrinkle answered: **the same session DOES see its own uncommitted append** — `zarr.open_group(session.store)` showed shape (4, 64, 64) and time indices [0 1 2 3] immediately after the append, before commit, so `region={"time": slice(3,4)}` validated and wrote fine.
- Read-back: 4 dates, appended time == 2024-01-04, window 7.0/77 correct, rest of new date fill, original 3 dates untouched.
- Two-session fallback (append+commit, then region+commit) was coded but never triggered. Pre-seeding dates first is NOT mandatory.

## TEST E — uncommitted r+ writes invisible to a separate readonly reader: **PASS**

After one r+ region write on an uncommitted session: a `readonly_session(branch="main")` from a **separate** `Repository.open` handle read the entire store as all-fill, and ancestry count was unchanged. After `session.commit(...)` the same read path saw the written window (guards against a false pass). Snapshots 2 -> 3.

## TEST F (bonus) — root attrs across multiple r+ calls: preserved

Root attrs stamped in a prior commit (`{"geoemb:probe": "keep-me", "spatial:thing": 42}`) survived two `mode="r+"` region writes + commit in one session, unchanged. So the attrs-clobbering that motivated the repo's `_commit_preserving_attrs` did **not** reproduce for pure `mode="r+"` region writes on icechunk 2.0.4 / xarray 2026.4.0 (it may still apply to `mode="a"`/`"w"` or older versions — this is a data point, not a recommendation to drop the guard).

## Verdict

**Multiple to_icechunk region writes per session+commit: YES.** On icechunk 2.0.4 / zarr 3.2.1 / xarray 2026.4.0 with dask-backed data on the threaded scheduler, N sequential `to_icechunk(..., mode="r+", region=..., align_chunks=True, split_every=8)` calls against one `writable_session("main")` followed by one `session.commit()` produce exactly one snapshot, with all windows correct and no cross-window interference even when windows share a chunk row (tested up to 20 windows; ~10 ms/write locally, no degradation). The session never refused a second write, i.e. to_icechunk's internal fork/merge for dask tolerates a session that already carries uncommitted changes. **Append+region same session: YES** — `mode="a", append_dim="time"` followed by `mode="r+"` region writes into the just-appended date commits as one snapshot, because the session exposes its own uncommitted append (shape already reflects the new date before commit), so region indices for the new date resolve without an intermediate commit; the two-session pre-seed-dates-first fallback is available but not required. Uncommitted session writes are invisible to readonly readers on separate repo handles (commit atomicity holds). Caveats: all windows here were exactly chunk-aligned, so `align_chunks=True` never had to read-modify-write a boundary chunk — two writes straddling the SAME chunk within one session were not tested and should still be treated as forbidden (disjoint chunk-aligned windows remain the contract); root attrs were not clobbered by multiple r+ calls in this stack; local-FS icechunk warns it is unsafe for concurrent commits (irrelevant here — one commit).
