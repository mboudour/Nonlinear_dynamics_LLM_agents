"""E4: mean-field tipping vs complete and random-regular graphs, homogeneous s (Fig. 4)."""
from _common import *
import numpy as np, networkx as nx
from scipy.optimize import brentq
from llmbvp.core import *
S = 0.65
BETAS = [1.02, 1.05, 1.1, 1.2, 1.35, 1.5, 2, 3, 5]
def Hc(b):
    if b <= 1: return 0.0
    t = np.sqrt(1 - 1 / b); return b * t - np.arctanh(t)
def pc_mf(beta):
    g = beta / S
    return brentq(lambda p: beta * p * np.tanh(g) - Hc(beta * (1 - p)), 1e-15, 1 - 1e-12)
out = dict(beta=BETAS, mf=[pc_mf(b) for b in BETAS], complete=[], rr=[])
# complete graph: by symmetry the committed set is irrelevant
n = 1000; W = (np.ones((n, n)) - np.eye(n)) / (n - 1); s = np.full(n, S)
for b in BETAS:
    out['complete'].append(tipping_fraction(W, s, b * gstar(W, s), np.arange(n)))
# random 10-regular graph, random placement
n = 2000; W = rowstoch(nx.to_numpy_array(nx.random_regular_graph(10, n, seed=21))); s = np.full(n, S)
g0 = gstar(W, s)
for b in BETAS:
    rng = np.random.default_rng(22)
    out['rr'].append([tipping_fraction(W, s, b * g0, rng.permutation(n)) for _ in range(5)])
    print(b, out['complete'][-1], np.mean(out['rr'][-1]), out['mf'][BETAS.index(b)], flush=True)
save('E4_meanfield.json', out)
