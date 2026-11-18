"""Choose the precision gate's probe regularization (C) by spatial cross-validation on the bf16 reference.

The probe should behave like a classifier a user would tune on these embeddings, so C is chosen the way a user
would choose it: by held-out accuracy, with no FP8 data involved. Each tile uses the gate's own folds (three window
rows, 48k training pixels from the other two rows; see ``calibrate.py``), so the score is spatially held out.

The score is balanced accuracy (mean per-class recall over classes present in the held-out row), because the
gate judges every class rather than the pixel majority. One C serves every tile. It is the smallest C whose mean
score is within one standard error of the best, the SE taken over the paired (tile, fold) differences from the
best C. For each C the report also gives the share of the weights' squared norm on the trailing 64 dimensions,
which shows how far the probe leans on v2's low-signal Matryoshka tail.

usage: select_probe_c.py <out.json> <labels_dir> <embed_dir> <tile>:<label_key>:<labels_file> [...]
  embed_dir holds the record's mixed_g6e_<tile>.npz (its bf16_q/bf16_s arrays are the reference).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

from tessera_embeddings.profiling.inference import precision_gate as pg

GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)
NTRAIN = 48000
PROBE_TOL = 1e-6
PROBE_MAX_ITER = 5000
TAIL = slice(64, 128)


def load(labels: Path, embed_dir: Path, tile: str, key: str) -> tuple[np.ndarray, np.ndarray, list]:
    """The tile's bf16 reference, its class labels, and the gate's three folds."""
    lab_f = np.load(labels)
    emb = np.load(embed_dir / f"mixed_g6e_{tile}.npz")
    assert (emb["Y"] == lab_f["Y"]).all() and (emb["X"] == lab_f["X"]).all()
    x = emb["bf16_q"].astype(np.float32) * emb["bf16_s"][:, None]
    lab, _ = pg.class_labels(lab_f[key], key)
    row = np.searchsorted(np.unique(lab_f["Y"]), lab_f["Y"]) // 160
    ok = lab >= 0
    rng = np.random.default_rng(1)
    folds = [
        (np.flatnonzero(ok & (row == f)), rng.choice(np.flatnonzero(ok & (row != f)), NTRAIN, replace=False))
        for f in range(3)
    ]
    return x, lab, folds


def score(x: np.ndarray, lab: np.ndarray, held: np.ndarray, train: np.ndarray, c: float) -> dict:
    """Held-out balanced accuracy, iterations and tail weight share for one fold at one C."""
    mu, sd = x[train].mean(0), x[train].std(0) + 1e-9
    clf = LogisticRegression(C=c, max_iter=PROBE_MAX_ITER, tol=PROBE_TOL).fit((x[train] - mu) / sd, lab[train])
    pred = clf.predict((x[held] - mu) / sd)
    truth = lab[held]
    recalls = [float((pred[truth == k] == k).mean()) for k in np.unique(truth)]
    w2 = clf.coef_**2
    return {
        "bacc": float(np.mean(recalls)),
        "acc": float((pred == truth).mean()),
        "n_iter": int(np.max(clf.n_iter_)),
        "tail_share": float(w2[:, TAIL].sum() / w2.sum()),
    }


def main(argv: list[str]) -> int:
    """Entry point."""
    out, labels_dir, embed_dir, *specs = argv
    tiles = [s.split(":") for s in specs]
    data = {t: load(Path(labels_dir) / lf, Path(embed_dir), t, k) for t, k, lf in tiles}
    jobs = [(t, c, f) for t, *_ in tiles for c in GRID for f in range(3)]
    res = Parallel(n_jobs=-1, verbose=5)(delayed(score)(data[t][0], data[t][1], *data[t][2][f], c) for t, c, f in jobs)
    table = {(t, c, f): r for (t, c, f), r in zip(jobs, res, strict=True)}
    names = [t for t, *_ in tiles]
    mean = {c: np.mean([table[t, c, f]["bacc"] for t in names for f in range(3)]) for c in GRID}
    best = max(GRID, key=lambda c: mean[c])
    se = {}
    for c in GRID:
        d = [table[t, best, f]["bacc"] - table[t, c, f]["bacc"] for t in names for f in range(3)]
        se[c] = float(np.std(d, ddof=1) / np.sqrt(len(d)))
    chosen = min(c for c in GRID if mean[best] - mean[c] <= se[c])
    report = {
        "grid": list(GRID),
        "best": best,
        "chosen": chosen,
        "mean_bacc": {str(c): float(mean[c]) for c in GRID},
        "se_vs_best": {str(c): se[c] for c in GRID},
        "per_tile": {
            t: {
                str(c): {
                    k: float(np.mean([table[t, c, f][k] for f in range(3)]))
                    for k in ("bacc", "acc", "n_iter", "tail_share")
                }
                for c in GRID
            }
            for t in names
        },
    }
    Path(out).write_text(json.dumps(report, indent=1))
    print(f"{'C':>8} {'bacc':>7} {'-best':>7} {'se':>6}  " + "  ".join(f"{t[-8:]:>15}" for t in names))
    for c in GRID:
        cells = "  ".join(
            f"{report['per_tile'][t][str(c)]['bacc']:.3f}/{report['per_tile'][t][str(c)]['tail_share']:.2f}/"
            f"{report['per_tile'][t][str(c)]['n_iter']:>4.0f}"
            for t in names
        )
        print(f"{c:>8g} {mean[c]:.4f} {mean[best] - mean[c]:+.4f} {se[c]:.4f}  {cells}")
    print(f"best C {best}; chosen (one-SE) C {chosen}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
