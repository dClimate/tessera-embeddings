# Configuration

Settings the pipeline reads at startup. Nothing here does work; it describes what the
working code should do, so changing a value here changes behaviour everywhere.

## Start here if you are contributing

Most changes land in two small files.

**`satellites.py`** lists the satellites we read: which bands to fetch from each one, and
the corrections their pixel values need. Adding a band, or supporting a new satellite,
starts here.

**`providers.py`** lists the catalogues we ask for imagery — where each one lives, which
collections it serves, and how it behaves when a request is too large. Adding a data source
means adding an entry here.

The two are separate on purpose. A satellite is one thing; the several catalogues that
publish it are another, and they disagree about naming and about whether pixel corrections
have already been applied.

## The rest

`paths.py` holds the bucket and prefix locations. These always come from the caller, so the
same code runs against a test bucket or the real one without a flag.

`store_layout.py` defines the shape of the data on disk — how arrays are split into chunks,
what type each holds, how they are compressed. The code that creates stores and the code
that writes to them both read it, so the two cannot disagree.

`ingest.py`, `inference.py`, `assembly.py` and `dask.py` hold the tunable numbers for each
pipeline stage: batch sizes, worker counts, retry limits, quality thresholds.

`environment.py` sets options for the image-reading library, and must run before that
library is imported. `time_windows.py` works out which twelve months a run covers.
`code_identity.py` fingerprints the source code so a half-finished job can tell whether it
was produced by the version now running. `fault_injection.py` triggers failures deliberately,
for testing recovery.
