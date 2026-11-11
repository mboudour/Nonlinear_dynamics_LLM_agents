"""E3: committed-minority tipping fraction vs targeting rule (Fig. 3)."""
from _common import *
import numpy as np, sys
from llmbvp.core import *
RATIOS = [1.1, 1.25, 1.5, 2, 3, 5]; NRAND = 20
nets = benchmark_networks()
names = sys.argv[1:] or list(nets)
for name in names:
    W = nets[name]; n = len(W); rows = []
    for p in range(NPROF[name]):
        s = susceptibility(n, p); g0 = gstar(W, s)
        for r in RATIOS:
            g = r * g0
            cents = centralities(W, s, g)
            qc = flip_thresholds(W, s, g, steps=16, tol=1e-8)
            Bfj = cents['FJ broadcasting']
            orders = {k: np.argsort(-v, kind='stable') for k, v in cents.items()}
            orders['Flip threshold'] = np.lexsort((-Bfj, qc))
            Vstart = committed_state(W, s, g, -1.0)[0][:, 0]
            for k, o in orders.items():
                rows.append(dict(p=p, ratio=r, rule=k, pc=tipping_fraction(W, s, g, o, Vstart=Vstart)))
            rng = np.random.default_rng(50 + 100 * p + int(10 * r))
            for d in range(NRAND):
                rows.append(dict(p=p, ratio=r, rule='Random', draw=d,
                                 pc=tipping_fraction(W, s, g, rng.permutation(n), Vstart=Vstart)))
            print(name, p, r, flush=True)
    save(f'E3_tipping_{name.replace(" ", "_").replace("é", "e")}.json', rows)
