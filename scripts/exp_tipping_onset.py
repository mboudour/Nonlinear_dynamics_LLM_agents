"""E3b: tipping with the onset (neutral-state) left Perron vector of S(W o K) as targeting rule."""
from _common import *
import numpy as np
from llmbvp.core import *
RATIOS = [1.1, 1.25, 1.5, 2, 3, 5]
out = {}
for name, W in benchmark_networks().items():
    n = len(W); rows = []
    for p in range(NPROF[name]):
        s = susceptibility(n, p); g0 = gstar(W, s)
        o = np.argsort(-perron_left(s[:, None] * W), kind='stable')
        for r in RATIOS:
            g = r * g0; Vstart = committed_state(W, s, g, -1.0)[0][:, 0]
            rows.append(dict(p=p, ratio=r, rule='Onset Perron', pc=tipping_fraction(W, s, g, o, Vstart=Vstart)))
    out[name] = rows; print(name, flush=True)
save('E3b_tipping_onset.json', out)
