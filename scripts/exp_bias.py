"""E5: biased channels f(x) = tanh(g x + eps): asymmetric tipping (Fig. 5)."""
from _common import *
import numpy as np
from llmbvp.core import *
EPS = [0.0, 0.02, 0.05, 0.1, 0.2]; RATIO = 2.0; NRAND = 10
nets = benchmark_networks(); out = {}
for name in ['Karate club', 'Directed ER']:
    W = nets[name]; n = len(W); rows = []
    for p in range(4):
        s = susceptibility(n, p); g = RATIO * gstar(W, s)
        Bfj, _ = scores(fj_response(W, s)); o = np.argsort(-Bfj)
        for e in EPS:
            for sign in [+1.0, -1.0]:          # +1: committed push toward the favoured side
                rng = np.random.default_rng(70 + p)
                rows.append(dict(p=p, eps=e, toward=('favoured' if sign > 0 else 'disfavoured'), rule='FJ broadcasting',
                                 pc=tipping_fraction(W, s, g, o, sign=sign, eps=e)))
                for d in range(NRAND):
                    rows.append(dict(p=p, eps=e, toward=('favoured' if sign > 0 else 'disfavoured'), rule='Random',
                                     pc=tipping_fraction(W, s, g, rng.permutation(n), sign=sign, eps=e)))
        print(name, p, flush=True)
    out[name] = rows
save('E5_bias.json', out)
