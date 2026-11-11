"""Checks the hypotheses of Theorem 4 (v^- < 0, rho(DT(v^-)) < 1) for every flip experiment,
and monotonicity in k of the tipping criterion by full scans."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, networkx as nx
from llmbvp.core import *
nets = benchmark_networks()
for name in ['Karate club', 'Les Misérables']:
    W = nets[name]; n = len(W); worst_rho, worst_v, reducible = 0, -1, 0
    for p in range(NPROF[name]):
        s = susceptibility(n, p); g = 1.5 * gstar(W, s)
        Cmask = np.eye(n, dtype=bool)
        V, conv, _ = iterate(W, s, g, Cmask, 0.0, -np.ones((n, n)), tol=1e-12)
        for a in range(n):
            keep = np.setdiff1d(np.arange(n), [a]); v = V[keep, a]
            J = s[keep, None] * W[np.ix_(keep, keep)] * (g * (1 - np.tanh(g * v) ** 2))[None, :]
            worst_rho = max(worst_rho, spectral_radius(J)); worst_v = max(worst_v, v.max())
            G = nx.from_numpy_array((W[np.ix_(keep, keep)] > 0).astype(int).T, create_using=nx.DiGraph)
            reducible += not nx.is_strongly_connected(G)
    print(f"{name}: max rho(DT(v-)) = {worst_rho:.4f}, max entry of v- = {worst_v:.4f}, reducible interiors = {reducible} of {n*NPROF[name]}")
# monotonicity scans of the tipping criterion
for name in ['Karate club', 'Les Misérables']:
    W = nets[name]; n = len(W); viol = 0; total = 0
    for p in range(4):
        s = susceptibility(n, p)
        for r in [1.5, 2, 3, 5]:
            g = r * gstar(W, s); Vs = committed_state(W, s, g, -1.0)[0][:, 0]
            orders = [np.argsort(-W.sum(0)), np.random.default_rng(p).permutation(n)]
            for o in orders:
                res = []
                for k in range(1, n):
                    C = np.zeros((n, 1), bool); C[o[:k], 0] = True
                    Vk, _, _ = iterate(W, s, g, C, 1.0, Vs[:, None], tol=1e-9)
                    res.append(np.mean(Vk[~C[:, 0], 0]) > 0)
                res = np.array(res); total += 1
                viol += np.any(res[:-1] & ~res[1:])
    print(f"{name}: tipping criterion non-monotone in {viol} of {total} full scans")
# agent 1 loop: branch difference
W = nets['Karate club']; n = len(W); s = susceptibility(n, 0); g = 1.5 * gstar(W, s); a = 0
C = np.zeros((n, 1), bool); C[a] = True
up, _, _ = iterate(W, s, g, C, 0.0, -np.ones((n, 1)), tol=1e-12)
V, _, _ = iterate(W, s, g, C, 1.0, -np.ones((n, 1)), tol=1e-12)
dn, _, _ = iterate(W, s, g, C, 0.0, V, tol=1e-12)
print("agent 1 at q=0: mean of others from below %.3f, from above %.3f" % (np.mean(np.delete(up[:, 0], a)), np.mean(np.delete(dn[:, 0], a))))
