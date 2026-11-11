"""E7: full integer scans (exact continuation) for every targeting sequence of Figs. 3 and 5 and the
random-regular part of Fig. 4; adds the ordered-state Green rule. Compares with stored bisections."""
from _common import *
import numpy as np, json, networkx as nx, scipy.sparse as sp
from llmbvp.core import *
RATIOS = [1.1, 1.25, 1.5, 2, 3, 5]; NRAND = 20
nets = benchmark_networks(); onset = load('E3b_tipping_onset.json'); out = {}
for name, W in nets.items():
    n = len(W); rows = []
    stored = load('E3_tipping_%s.json' % name.replace(' ', '_').replace('é', 'e')) + onset[name]
    for p in range(NPROF[name]):
        s = susceptibility(n, p); g0 = gstar(W, s)
        for r in RATIOS:
            g = r * g0; Vs = committed_state(W, s, g, -1.0)[0][:, 0]
            cents = centralities(W, s, g); Bfj = cents['FJ broadcasting']
            cents['Ordered Green'] = ordered_green(W, s, g, Vs)
            cents['Onset Perron'] = perron_left(s[:, None] * W)
            orders = {k: np.argsort(-v, kind='stable') for k, v in cents.items()}
            if n < 100:
                orders['Flip threshold'] = np.lexsort((-Bfj, flip_thresholds(W, s, g, steps=16, tol=1e-8)))
            for k, o in orders.items():
                res = tipping_scan(W, s, g, o, Vstart=Vs)
                old = [x['pc'] for x in stored if x['p'] == p and x['ratio'] == r and x['rule'] == k]
                rows.append(dict(p=p, ratio=r, rule=k, **res, bisection=(old[0] if old else None)))
            rng = np.random.default_rng(50 + 100 * p + int(10 * r))
            olds = [x['pc'] for x in stored if x['p'] == p and x['ratio'] == r and x['rule'] == 'Random']
            for d in range(NRAND):
                res = tipping_scan(W, s, g, rng.permutation(n), Vstart=Vs)
                rows.append(dict(p=p, ratio=r, rule='Random', draw=d, **res, bisection=olds[d]))
    out[name] = rows; print(name, flush=True)
# biased channels (Fig. 5)
bias = {}
for name in ['Karate club', 'Directed ER']:
    W = nets[name]; n = len(W); rows = []; stored = load('E5_bias.json')[name]
    for p in range(4):
        s = susceptibility(n, p); g = 2.0 * gstar(W, s)
        Bfj, _ = scores(fj_response(W, s)); o = np.argsort(-Bfj)
        for e in [0.0, 0.02, 0.05, 0.1, 0.2]:
            for sign in [+1.0, -1.0]:
                tw = 'favoured' if sign > 0 else 'disfavoured'
                old = [x['pc'] for x in stored if x['p'] == p and x['eps'] == e and x['toward'] == tw]
                rng = np.random.default_rng(70 + p)
                res = tipping_scan(W, s, g, o, sign=sign, eps=e)
                rows.append(dict(p=p, eps=e, toward=tw, rule='FJ broadcasting', **res, bisection=old[0]))
                for d in range(10):
                    res = tipping_scan(W, s, g, rng.permutation(n), sign=sign, eps=e)
                    rows.append(dict(p=p, eps=e, toward=tw, rule='Random', draw=d, **res, bisection=old[1 + d]))
    bias[name] = rows; print('bias', name, flush=True)
save('E7_scans.json', dict(tipping=out, bias=bias, rr=[]))
# random-regular graph (Fig. 4)
M = load('E4_meanfield.json'); n = 2000
W = sp.csr_matrix(rowstoch(nx.to_numpy_array(nx.random_regular_graph(10, n, seed=21)))); s = np.full(n, 0.65)
Wd = W.toarray(); g0 = gstar(Wd, s); rr = []
for i, b in enumerate(M['beta']):
    rng = np.random.default_rng(22)
    for d in range(5):
        res = tipping_scan(W, s, b * g0, rng.permutation(n))
        rr.append(dict(beta=b, draw=d, **res, bisection=M['rr'][i][d]))
    print('rr', b, flush=True)
    save('E7_scans.json', dict(tipping=out, bias=bias, rr=rr))
