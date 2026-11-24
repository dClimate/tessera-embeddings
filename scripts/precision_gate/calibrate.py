"""Per-pixel predictions for every variant the precision gate scores on one tile (ADR 043).

For one tile and one shard of the variant list, writes one ``.npz`` per variant: a linear probe's and a
kNN-20 classifier's per-pixel predictions in both directions across the change, and the gap-closure
matrices. ``apply_gate.py`` applies the gate from these files, so its rule can change without
recomputing predictions. Shard 0 also writes ``envelope.json``: ADR 012's envelope and coherence for
every embedded comparison.

Embeddings come from ``embed_lattice.py``: ``lattice_<side>_<tile>.npz`` for sides ``l40s``, ``a10g``,
``l4`` (today's code) and ``cand`` (the candidate, on an L40S). The reference is ``l40s``. Variants:

  cards_A10G, cards_L4    two cards on today's code (known good)
  candidate               the candidate against the reference
  mixed                   reference A10G, test the candidate (a store both cards write to)
  noise_<d>               isotropic noise matched to the candidate's mean cosine after int8, 40 draws
  ctrl_<i>_<j>_<cos>      class i shifted towards class j's centroid, cosine <cos> inside class i
                          (j = i's most-confused class under the reference probe)
  ctrl2_<i>_<j>_<cos>     the largest class shifted towards the smallest control-eligible class
  refit_<r>               the reference only, 48k training sample redrawn (seed r): the refit spread, reported

Per variant (int8 arrays over the lattice; -1 = unlabelled):
  p_rx, k_rx   fitted on the reference, applied to the test side  (baseline: p_rr, k_rr in base.npz)
  p_xx, k_xx   fitted on the test side, applied to itself
  p_xr, k_xr   fitted on the test side, applied to the reference
  gc_cls[i,j]  class i's mean unit-vector error projected on (centroid j - centroid i), over |gap|^2, with
               the error and the centroids taken from the comparison's own reference side
  gc_clu[a,b]  the same over 64 k-means clusters (fitted on L40S bf16, assigned on the reference side)
  (mixed variants also carry p_rr, k_rr for their own reference; refit variants carry only p_rr, k_rr)

usage: calibrate.py <labels.npz> <label_key> <embed_dir> <tile> <shard> <nshards> <out_dir>
env:   SMOKE=1 for a fast local check (small training sample, a few variants)
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from tessera_embeddings.profiling.inference import precision_gate as pg

labels_path, label_key, embed_dir, TILE = sys.argv[1:5]
SHARD, NSHARDS, OUTDIR = int(sys.argv[5]), int(sys.argv[6]), Path(sys.argv[7])
SMOKE = os.environ.get("SMOKE") == "1"
NTRAIN = 4800 if SMOKE else 48000
NDRAWS = 4 if SMOKE else 40
DOSES = (0.9999, 0.99995, 0.99998)
NREFIT = 2 if SMOKE else 10
MIN_CTRL = 2000  # px; classes smaller than this cannot reach the gating SE, so get no control
PROBE_C = 0.003  # spatial CV on the reference, one-SE rule (select_probe_c.py, removed in this file's history)
PROBE_TOL = 1e-6
PROBE_MAX_ITER = 5000
OUTDIR.mkdir(parents=True, exist_ok=True)

L = np.load(labels_path)
F = {side: np.load(Path(embed_dir) / f"lattice_{side}_{TILE}.npz") for side in ("l40s", "a10g", "l4", "cand")}
for f in F.values():
    assert (f["Y"] == L["Y"]).all() and (f["X"] == L["X"]).all()
N = L["Y"].size


def deq(side: str) -> np.ndarray:
    """A side's embedding, dequantised to float32."""
    return F[side]["q"].astype(np.float32) * F[side]["s"][:, None]


def environment() -> dict:
    """What the predictions depend on beyond the data: library versions, BLAS, the probe's stopping rule."""
    import platform

    import scipy
    import sklearn
    from threadpoolctl import threadpool_info

    blas = [
        {k: i.get(k) for k in ("internal_api", "version", "architecture", "num_threads")} for i in threadpool_info()
    ]
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "blas": blas,
        "probe": {"C": PROBE_C, "tol": PROBE_TOL, "max_iter": PROBE_MAX_ITER},
    }


def unit(x: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length."""
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def requant(x: np.ndarray) -> np.ndarray:
    """Round-trip through the store's int8 per-row quantisation."""
    s = np.maximum(np.abs(x).max(axis=1) / 127.0, 1e-8).astype(np.float32)
    return (np.clip(np.rint(x / s[:, None]), -127, 127) * s[:, None]).astype(np.float32)


lab, names = pg.class_labels(L[label_key], label_key)
lab = lab.astype(np.int8)
K = len(names)
uy, ux = np.unique(L["Y"]), np.unique(L["X"])
assert uy.size == 480 and ux.size == 480, (uy.size, ux.size)
yi, xi = np.searchsorted(uy, L["Y"]), np.searchsorted(ux, L["X"])
win = 3 * (yi // 160) + xi // 160
ok = lab >= 0
cnt = np.bincount(lab[ok], minlength=K)

ref = deq("l40s")
u_ref = unit(ref)
cent = np.stack([u_ref[lab == i].mean(0) if cnt[i] else np.zeros(128, np.float32) for i in range(K)])
km_rng = np.random.default_rng(0)
km = KMeans(n_clusters=64, n_init=1, random_state=0).fit(u_ref[km_rng.choice(N, 30000, replace=False)])
clus = km.predict(u_ref)


def folds_for(seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Three folds by window row: (held-out pixels, a training sample from the other rows)."""
    frs = np.random.default_rng(seed)
    return [
        (np.flatnonzero(ok & (win // 3 == f)), frs.choice(np.flatnonzero(ok & (win // 3 != f)), NTRAIN, replace=False))
        for f in range(3)
    ]


FOLDS = folds_for(1)  # the scorer's own folds


def fit(a: np.ndarray, folds: list = FOLDS) -> list:
    """A standardised, L2-regularised logistic-regression probe per fold, fitted on ``a`` to convergence.

    ``PROBE_C`` is the regularization a user tuning a classifier on these embeddings would pick: it maximises
    held-out balanced accuracy on the reference (within one SE). The fit runs to convergence because a fit
    stopped short is not a function of its data alone: where it stops depends on the floating-point path, which
    moves verdicts between environments.
    """
    out = []
    for _, train in folds:
        mu, sd = a[train].mean(0), a[train].std(0) + 1e-9
        clf = LogisticRegression(C=PROBE_C, max_iter=PROBE_MAX_ITER, tol=PROBE_TOL)
        with warnings.catch_warnings():  # bootstrap runs with -W ignore; a probe that stops short must still fail
            warnings.simplefilter("error", ConvergenceWarning)
            clf.fit((a[train] - mu) / sd, lab[train])
        out.append((clf, mu, sd))
    return out


def probe(fits: list, x: np.ndarray, folds: list = FOLDS) -> np.ndarray:
    """Per-pixel probe predictions on ``x``'s held-out pixels; -1 elsewhere."""
    p = np.full(N, -1, dtype=np.int8)
    for (held, _), (clf, mu, sd) in zip(folds, fits, strict=True):
        p[held] = clf.predict((x[held] - mu) / sd)
    return p


def knn(a: np.ndarray, x: np.ndarray, folds: list = FOLDS) -> np.ndarray:
    """Per-pixel kNN-20 predictions for ``x``'s held-out pixels against ``a``'s training pixels; -1 elsewhere."""
    p = np.full(N, -1, dtype=np.int8)
    for held, train in folds:
        ut, xu, lt = unit(a[train]), unit(x[held]), lab[train]
        for s in range(0, held.size, 1000):
            nb = np.argpartition(-(xu[s : s + 1000] @ ut.T), 20, axis=1)[:, :20]
            p[held[s : s + 1000]] = (lt[nb][:, :, None] == np.arange(K)).sum(1).argmax(1)
    return p


def gap_closure(x: np.ndarray, r: np.ndarray, groups_of: np.ndarray, n: int) -> np.ndarray:
    """[i, j]: group i's mean unit-vector error from ``r`` to ``x``, on the gap between ``r``'s centroids i and j.

    As a share of that gap: the reference side sets both the error and the centroids.
    """
    u_r = unit(r)

    def means(v: np.ndarray) -> np.ndarray:
        return np.stack(
            [v[groups_of == g].mean(0) if (groups_of == g).any() else np.zeros(128, np.float32) for g in range(n)]
        )

    me, cents = means(unit(x) - u_r), means(u_r)
    gap = cents[None, :, :] - cents[:, None, :]  # [i, j] = cent_j - cent_i
    return ((me[:, None, :] * gap).sum(-1) / np.maximum((gap * gap).sum(-1), 1e-12)).astype(np.float32)


def noisy(seed: int, cn: float) -> np.ndarray:
    """The reference with isotropic noise at mean cosine ``cn``, re-quantised."""
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(ref.shape).astype(np.float32)
    n -= (n * u_ref).sum(1, keepdims=True) * u_ref  # orthogonal to each pixel, so the angle is exact
    scale = np.tan(np.arccos(cn)) * np.linalg.norm(ref, axis=1, keepdims=True)
    n *= scale / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    return requant(ref + n)


def directed(src: int, dst: int, c: float) -> np.ndarray:
    """The reference with class ``src`` shifted towards class ``dst``'s centroid at cosine ``c``, re-quantised."""
    x = ref.copy()
    m = lab == src
    d = np.repeat(cent[dst][None].astype(np.float32), int(m.sum()), 0)
    xn = np.linalg.norm(ref[m], axis=1, keepdims=True)
    u = ref[m] / xn
    p = d - (d * u).sum(1, keepdims=True) * u
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-12
    x[m] = ref[m] + np.tan(np.arccos(c)) * xn * p
    return requant(x)


t0 = time.time()


def log(m: str) -> None:
    """A progress line tagged with the tile, shard and elapsed time."""
    print(f"[{TILE} {SHARD}/{NSHARDS} {time.time() - t0:5.0f}s] {m}", flush=True)


def save(name: str, **arrs: np.ndarray) -> None:
    """Write one variant's predictions."""
    np.savez_compressed(OUTDIR / f"{name}.npz", names=np.array(names), cnt=cnt, **arrs)
    log(f"done {name}")


# (reference side, test side) per embedded comparison; the envelope is scored for each
PAIRS = {
    "cards_A10G": ("l40s", "a10g"),
    "cards_L4": ("l40s", "l4"),
    "candidate": ("l40s", "cand"),
    "mixed": ("a10g", "cand"),
}

log("classes: " + ", ".join(f"{n} {c}" for n, c in zip(names, cnt, strict=True)))
fits_ref = fit(ref)
p_rr = probe(fits_ref, ref)
eligible = [i for i in range(K) if cnt[i] >= MIN_CTRL]
conf = np.zeros((K, K), int)
np.add.at(conf, (lab[ok], p_rr[ok]), 1)
np.fill_diagonal(conf, -1)
confused = {i: int(np.argmax(np.where(np.arange(K) != i, conf[i], -1))) for i in eligible}
big = int(np.argmax(cnt))
small = min((i for i in eligible if i != big), key=lambda i: cnt[i])
log(
    f"base probe acc {100 * (p_rr[ok] == lab[ok]).mean():.2f}; most confused: "
    + ", ".join(f"{names[i]}->{names[j]}" for i, j in confused.items())
    + f"; ctrl2 {names[big]}->{names[small]}"
)

c = float((u_ref * unit(deq("cand"))).sum(1).mean())
c_try = float((u_ref * unit(noisy(999, c))).sum(1).mean())
cn = 1 - (2 * (1 - c) - (1 - c_try))  # squared angles add: aim the noise at what is left after re-quantising

V = list(PAIRS)
V += [f"noise_{d}" for d in range(NDRAWS)]
V += [f"ctrl_{i}_{j}_{c:g}" for i, j in confused.items() for c in DOSES]
V += [f"ctrl2_{big}_{small}_{c:g}" for c in DOSES]
V += [f"refit_{r}" for r in range(NREFIT)]
if SMOKE:
    V = [V[k] for k in (0, 2, 3, *range(len(PAIRS), len(PAIRS) + NDRAWS), len(PAIRS) + NDRAWS, len(V) - 2, len(V) - 1)]
mine = [name for k, name in enumerate(V) if k % NSHARDS == SHARD]
log(f"{len(V)} variants, {len(mine)} on this shard")
if SHARD == 0:
    save("base", p_rr=p_rr, k_rr=knn(ref, ref), clus=clus.astype(np.int8), variants=np.array(V))
    env = {}
    for name, (r, t) in PAIRS.items():
        env[name] = pg.envelope(F[r]["q"], F[r]["s"], F[t]["q"], F[t]["s"])
        env[name]["coherence"] = pg.coherence(deq(r), deq(t))
    env["candidate"]["noise_cosine_target"] = cn
    (OUTDIR / "envelope.json").write_text(json.dumps(env, indent=1))
    (OUTDIR / "environment.json").write_text(json.dumps(environment(), indent=1))


def test_side(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(reference side, changed side) for a variant."""
    kind, *a = name.split("_")
    if name in PAIRS:
        r, t = PAIRS[name]
        return (ref if r == "l40s" else deq(r)), deq(t)
    if kind == "noise":
        return ref, noisy(100 + int(a[0]), cn)
    if kind in ("ctrl", "ctrl2"):
        return ref, directed(int(a[0]), int(a[1]), float(a[2]))
    raise ValueError(name)


for name in mine:
    if name.startswith("refit_"):
        folds = folds_for(1000 + int(name.split("_")[1]))
        save(name, p_rr=probe(fit(ref, folds), ref, folds), k_rr=knn(ref, ref, folds))
        continue
    r, x = test_side(name)
    extra = {}
    if r is ref:
        fr = fits_ref
    else:
        fr = fit(r)
        extra = {"p_rr": probe(fr, r), "k_rr": knn(r, r)}
    fx = fit(x)
    save(
        name,
        p_rx=probe(fr, x),
        k_rx=knn(r, x),
        p_xx=probe(fx, x),
        p_xr=probe(fx, r),
        k_xx=knn(x, x),
        k_xr=knn(x, r),
        gc_cls=gap_closure(x, r, lab, K),
        gc_clu=gap_closure(x, r, clus if r is ref else km.predict(unit(r)), 64),
        cos=np.float32((unit(r) * unit(x)).sum(1).mean()),
        **extra,
    )
log("SHARD COMPLETE")
