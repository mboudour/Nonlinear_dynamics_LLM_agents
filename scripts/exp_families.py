"""E6: two model families, critical gain vs homophily (Fig. 6)."""
from _common import *
import numpy as np
from llmbvp.core import *
aA, aB, N, K = 2.0, 0.5, 1000, 60
def family_net(f, h, seed=7):
    rng = np.random.default_rng(seed); nA = int(round(f * N)); fam = np.array([0] * nA + [1] * (N - nA))
    A = np.zeros((N, N))
    for i in range(N):
        fown = (nA if fam[i] == 0 else N - nA) / N
        kin = int(round((fown + (1 - fown) * h) * K))
        own = np.flatnonzero((fam == fam[i]) & (np.arange(N) != i)); oth = np.flatnonzero(fam != fam[i])
        A[i, rng.choice(own, kin, replace=False)] = 1; A[i, rng.choice(oth, K - kin, replace=False)] = 1
    return rowstoch(A), fam
rows = []
for f in [0.1, 0.3, 0.5]:
    for h in np.round(np.linspace(0, 0.9, 10), 2):
        W, fam = family_net(f, h); alpha = np.where(fam == 0, aA, aB)
        qAA = W[fam == 0][:, fam == 0].sum(1).mean(); qBB = W[fam == 1][:, fam == 1].sum(1).mean()
        rec = dict(f=f, h=float(h), qAA=float(qAA), qBB=float(qBB))
        for lab, s in [('hetero', susceptibility(N, 8)), ('homog', np.full(N, 0.65))]:
            g = gstar(W, s, alpha=alpha)
            sA, sB = s[fam == 0].mean(), s[fam == 1].mean()
            M = np.array([[sA * aA * qAA, sA * aA * (1 - qAA)], [sB * aB * (1 - qBB), sB * aB * qBB]])
            rec[lab] = dict(gstar=g, gstar_quot=1 / max(abs(np.linalg.eigvals(M))))
        # committed-state profile by family at g = 1.2 g* (heterogeneous s)
        s = susceptibility(N, 8); g = 1.2 * rec['hetero']['gstar']
        V, conv = committed_state(W, s, g, +1.0, alpha=alpha)
        rec['mean_A'] = float(V[fam == 0, 0].mean()); rec['mean_B'] = float(V[fam == 1, 0].mean())
        rows.append(rec); print(f, h, rec['hetero'], rec['mean_A'], rec['mean_B'], flush=True)
save('E6_families.json', rows)
