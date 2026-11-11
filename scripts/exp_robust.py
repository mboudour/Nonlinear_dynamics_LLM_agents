"""E8: robustness of the targeting comparison to network density and susceptibility distribution."""
from _common import *
import numpy as np, networkx as nx
from llmbvp.core import *
n = 300; rows = []
SUSC = {'Beta(4,2)': lambda m, p: susceptibility(m, p),
        'Beta(2,2)': lambda m, p: 0.15 + 0.75 * np.random.default_rng(2000 + p).beta(2, 2, m)}
for deg, seed in [(5, 31), (20, 32)]:
    A = nx.to_numpy_array(nx.gnp_random_graph(n, deg / n, seed=seed, directed=True)).T
    G = nx.from_numpy_array(A.T, create_using=nx.DiGraph)
    keep = sorted(max(nx.strongly_connected_components(G), key=len))
    W = rowstoch(A[np.ix_(keep, keep)]); m = len(keep)
    for sl, sf in SUSC.items():
        for p in range(4):
            s = sf(m, p); g0 = gstar(W, s)
            for r in [2, 5]:
                g = r * g0; Vs = committed_state(W, s, g, -1.0)[0][:, 0]
                c = centralities(W, s, g)
                rules = {'Out-strength': c['Out-strength'], 'FJ broadcasting': c['FJ broadcasting'],
                         'Onset Perron': perron_left(s[:, None] * W), 'Jacobian Perron': c['Jacobian Perron'],
                         'Ordered Green': ordered_green(W, s, g, Vs)}
                for k, v in rules.items():
                    rows.append(dict(deg=deg, n=m, susc=sl, p=p, ratio=r, rule=k,
                                     **tipping_scan(W, s, g, np.argsort(-v, kind='stable'), Vstart=Vs)))
                rng = np.random.default_rng(80 + p)
                for d in range(10):
                    rows.append(dict(deg=deg, n=m, susc=sl, p=p, ratio=r, rule='Random', draw=d,
                                     **tipping_scan(W, s, g, rng.permutation(m), Vstart=Vs)))
        print(deg, sl, flush=True)
save('E8_robust.json', rows)
