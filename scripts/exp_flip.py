"""E2: single-source flip thresholds over all profiles, and hysteresis loops (Fig. 2)."""
from _common import *
import numpy as np
from llmbvp.core import *
RATIO = 1.5
nets = benchmark_networks(); out = {}
for name in ['Karate club', 'Les Misérables']:
    W = nets[name]; n = len(W); prof = []
    for p in range(NPROF[name]):
        s = susceptibility(n, p); g = RATIO * gstar(W, s)
        qc = flip_thresholds(W, s, g)
        prof.append(dict(p=p, g=g, min_ratio_a=float((g / gstar_sources(W, s)).min()), qc=qc.tolist()))
        print(name, p, np.flatnonzero(np.isfinite(qc)), flush=True)
    out[name] = prof
# loops: karate, profile 0
W = nets['Karate club']; n = len(W); s = susceptibility(n, 0); g = RATIO * gstar(W, s)
qs = np.concatenate([np.linspace(-1, 1, 161), np.linspace(1, -1, 161)])
loops = {}
for a in [33, 0, 11]:
    C = np.zeros((n, 1), bool); C[a] = True
    V, _, _ = iterate(W, s, g, C, -1.0, -np.ones((n, 1))); ms = []
    for q in qs:
        V, _, _ = iterate(W, s, g, C, q, V); ms.append(float(np.mean(np.delete(V[:, 0], a))))
    loops[str(a)] = ms
out['loops'] = dict(q=qs.tolist(), **loops)
save('E2_flip.json', out)
