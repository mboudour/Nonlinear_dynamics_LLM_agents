"""E1: response centralities across the gain (Fig. 1)."""
from _common import *
import numpy as np
from scipy.stats import spearmanr
from llmbvp.core import *
RATIOS = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.05, 1.2, 1.5, 2, 3, 5, 10, 30]
out = {}
for name, W in benchmark_networks().items():
    n = len(W); rows = []
    for p in range(NPROF[name]):
        s = susceptibility(n, p); g0 = gstar(W, s); gsa = gstar_sources(W, s)
        Bfj, Qfj = scores(fj_response(W, s))
        for r in RATIOS:
            g = r * g0
            U, c1, _ = source_experiments(W, s, g, 1.0); B, Q = scores(U)
            Us, c2, _ = source_experiments(W, s, g, 0.05); R = Us / 0.05
            off = R[~np.eye(n, dtype=bool)]
            rows.append(dict(p=p, ratio=r, gstar=g0, frac_super=float(np.mean(g > gsa)),
                             gsa_over_g_max=float(gsa.max() / g0),
                             rhoB_fj=spearmanr(B, Bfj)[0], rhoB_stub=spearmanr(B, -s)[0],
                             rhoQ_s=spearmanr(Q, s)[0], rhoQ_fj=spearmanr(Q, Qfj)[0],
                             rhoQfj_s=spearmanr(Qfj, s)[0],
                             cvB=float(B.std() / B.mean()), rangeB=float(B.max() - B.min()),
                             maxR=float(off.max()), fracR1=float(np.mean(off > 1)),
                             converged=bool(c1.all() and c2.all())))
        print(name, p, flush=True)
    out[name] = rows
save('E1_rankings.json', out)
