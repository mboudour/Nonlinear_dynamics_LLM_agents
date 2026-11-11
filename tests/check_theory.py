"""Numerical checks of the analytical results (run: python tests/check_theory.py)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import brentq
from llmbvp.core import *

nets = benchmark_networks()
W = nets['Karate club']; n = len(W); s = susceptibility(n, 0)
g0 = gstar(W, s); gs = gstar_sources(W, s)
print("[1] karate profile 0: g* =", round(g0, 4), " g*_a range:", round(gs.min(), 4), round(gs.max(), 4))
a = int(np.argmax(gs)); 
for lab, g in [("between g* and g*_a", 0.5 * (g0 + gs[a])), ("above g*_a", 1.05 * gs[a])]:
    vals = []
    for q in [1e-2, 1e-4, 1e-6]:
        C = np.zeros((n, 1), bool); C[a] = True
        V, conv, _ = iterate(W, s, g, C, q, np.zeros((n, 1)), tol=1e-14)
        vals.append(np.max(np.delete(V[:, 0], a)) / q)
    print(f"    source {a}, {lab}: max_b R_ab(q) for q=1e-2,1e-4,1e-6:", np.round(vals, 3))

print("[2] uniqueness/stability of the positive committed state (no committed agents)")
for name in ['Karate club', 'Directed ER', 'Directed heavy-tailed']:
    W2 = nets[name]; m = len(W2); s2 = susceptibility(m, 0); g = 1.5 * gstar(W2, s2)
    rng = np.random.default_rng(1)
    X0 = rng.random((m, 20)) * 0.9 + 0.05
    V, conv, _ = iterate(W2, s2, g, np.zeros((m, 20), bool), 0.0, X0, tol=1e-13)
    spread = np.max(V.max(1) - V.min(1))
    v = V[:, 0]; J = s2[:, None] * W2 * (g * (1 - np.tanh(g * v) ** 2))[None, :]
    print(f"    {name}: max spread across 20 positive starts = {spread:.2e}, rho(J(v+)) = {spectral_radius(J):.4f}, min v+ = {v.min():.3f}")

print("[3] period-2 orbit, two agents, s=0.8, tanh(2x)")
W2 = np.array([[0., 1.], [1., 0.]]); s2 = np.array([0.8, 0.8])
mstar = brentq(lambda x: 0.8 * np.tanh(2 * x) - x, 0.1, 1)
x = np.array([mstar, -mstar]); y = s2 * (W2 @ np.tanh(2 * x))
d = 0.8 * 2 * (1 - np.tanh(2 * mstar) ** 2)
print(f"    m={mstar:.4f}; image of (m,-m) = {np.round(y, 4)}; two-step multiplier d^2 = {d**2:.4f}")

print("[4] rho(M) nondecreasing in h (two families), 10^4 random draws")
rng = np.random.default_rng(2); bad = 0
for _ in range(10000):
    dA, dB = rng.random(2) * 2; fA = rng.random(); f = np.array([fA, 1 - fA]); dd = np.array([dA, dB])
    hs = np.linspace(0, 0.999, 50)
    r = [max(abs(np.linalg.eigvals(np.diag(dd) @ ((1 - h) * np.outer(np.ones(2), f) + h * np.eye(2))))) for h in hs]
    bad += np.any(np.diff(r) < -1e-12)
print("    violations:", bad)

print("[5] mean-field tipping: exact root vs asymptotic (2/(3 tanh(1/s)))(beta-1)^{3/2}, s=0.65")
def Hc(b):
    if b <= 1: return 0.0
    t = np.sqrt(1 - 1 / b); return b * t - np.arctanh(t)
sbar = 0.65
for beta in [1.01, 1.03, 1.1]:
    g = beta / sbar
    pc = brentq(lambda p: beta * p * np.tanh(g) - Hc(beta * (1 - p)), 1e-14, 1 - 1e-9)
    print(f"    beta={beta}: p_c={pc:.3e}, asymptotic={2 / (3 * np.tanh(1 / sbar)) * (beta - 1) ** 1.5:.3e}")

print("[6] small-signal formula, karate profile 0, g = 0.95 g*: |R(q) - R0| (relative tol 1e-15)")
g = 0.95 * g0; R0 = fj_response(W, s, K=g * np.ones((n, n)))
for q in [1e-2, 1e-3, 1e-4]:
    U, conv, _ = source_experiments(W, s, g, q, tol=1e-15 * q)
    print(f"    q={q}: max |R(q)-R0| = {np.max(np.abs(U / q - R0)):.2e}")

print("[7] maximum principle fails without instability: single edge, s=0.8, tanh(2x), q=0.05:",
      round(0.8 * np.tanh(2 * 0.05) / 0.05, 4))
