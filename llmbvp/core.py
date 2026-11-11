"""Channel-mediated boundary-value Friedkin-Johnsen dynamics.

Convention (receive-from): W[i, j] > 0 means the opinion of j enters the update of i.
Update for susceptible agent i:
    v_i <- s_i * sum_j W[i, j] * f(g * kappa_ij * v_j + eps) + (1 - s_i) * mu_i
with f = tanh and kappa_ij = alpha_i * eta_j. Committed agents keep a fixed value.
All experiments are vectorised: column k of a state matrix is one experiment.
"""
import numpy as np
import networkx as nx

TOL = 1e-10
MAXIT = 200_000


# ----------------------------------------------------------------- networks
def rowstoch(A):
    W = np.asarray(A, float).copy()
    rs = W.sum(1)
    if np.any(rs == 0):
        raise ValueError("every agent must have positive in-weight")
    return W / rs[:, None]


def chung_lu_scc(n=400, gamma=2.5, mean_deg=8, seed=14):
    """Directed Chung-Lu graph with power-law expected in/out degrees; largest SCC."""
    rng = np.random.default_rng(seed)
    w_out = (1 - rng.random(n)) ** (-1 / (gamma - 1))
    w_in = (1 - rng.random(n)) ** (-1 / (gamma - 1))
    w_out *= mean_deg * n / w_out.sum()
    w_in *= mean_deg * n / w_in.sum()
    P = np.minimum(1.0, np.outer(w_out, w_in) / (mean_deg * n))
    np.fill_diagonal(P, 0)
    adj = rng.random((n, n)) < P              # adj[u, v]: arc u -> v
    G = nx.from_numpy_array(adj.astype(int), create_using=nx.DiGraph)
    scc = sorted(max(nx.strongly_connected_components(G), key=len))
    return adj[np.ix_(scc, scc)].T.astype(float)   # receive-from matrix


def benchmark_networks():
    """Returns dict name -> row-stochastic receive-from matrix W."""
    nets = {}
    nets['Karate club'] = nx.to_numpy_array(nx.karate_club_graph(), weight=None)
    nets['Les Misérables'] = nx.to_numpy_array(nx.les_miserables_graph(), weight='weight')
    n = 300
    G = nx.gnp_random_graph(n, 10 / n, seed=11, directed=True)
    nets['Directed ER'] = nx.to_numpy_array(G).T
    P = np.full((4, 4), 1.5 / n); np.fill_diagonal(P, 30 / n)
    G = nx.stochastic_block_model([75] * 4, P, seed=12, directed=True)
    nets['Directed SBM'] = nx.to_numpy_array(G).T
    nets['Directed heavy-tailed'] = chung_lu_scc()
    return {k: rowstoch(v) for k, v in nets.items()}


NPROF = {'Karate club': 12, 'Les Misérables': 12, 'Directed ER': 4,
         'Directed SBM': 4, 'Directed heavy-tailed': 4}


def susceptibility(n, p):
    """Profile p: 0.15 + 0.75 * Beta(4, 2), seed 1000 + p."""
    return 0.15 + 0.75 * np.random.default_rng(1000 + p).beta(4, 2, n)


# ----------------------------------------------------------------- operators
def gain_matrix(W, s, g=1.0, alpha=None, eta=None, keep=None):
    n = len(s)
    alpha = np.ones(n) if alpha is None else alpha
    eta = np.ones(n) if eta is None else eta
    M = s[:, None] * W * (g * np.outer(alpha, eta))
    if keep is not None:
        M = M[np.ix_(keep, keep)]
    return M


def spectral_radius(M):
    return float(np.max(np.abs(np.linalg.eigvals(M)))) if M.size else 0.0


def gstar(W, s, alpha=None, eta=None, committed=()):
    """Critical gain of the interior obtained by committing `committed`."""
    keep = np.setdiff1d(np.arange(len(s)), np.asarray(committed, int))
    r = spectral_radius(gain_matrix(W, s, 1.0, alpha, eta, keep))
    return np.inf if r == 0 else 1.0 / r


def gstar_sources(W, s):
    return np.array([gstar(W, s, committed=[a]) for a in range(len(s))])


def fj_response(W, s, K=None):
    """R[a, b] = G[b, a] / G[a, a], G = (I - S (W o K))^{-1}  (K=None: identity channels)."""
    n = len(s)
    M = s[:, None] * W * (1.0 if K is None else K)
    G = np.linalg.inv(np.eye(n) - M)
    return (G / np.diag(G)[None, :]).T


def scores(R):
    n = R.shape[0]
    off = R - np.diag(np.diag(R))
    return off.sum(1) / (n - 1), off.sum(0) / (n - 1)    # B_a, Q_b


# ----------------------------------------------------------------- dynamics
def _phi(W, V, g, alpha, eta, eps):
    if alpha is None:
        return W @ np.tanh(g * (V if eta is None else eta[:, None] * V) + eps)
    out = np.empty_like(V)
    for a in np.unique(alpha):
        rows = alpha == a
        X = V if eta is None else eta[:, None] * V
        out[rows] = W[rows] @ np.tanh(g * a * X + eps)
    return out


def iterate(W, s, g, Cmask, Cval, V0, alpha=None, eta=None, eps=0.0, mu=0.0,
            tol=TOL, maxit=MAXIT):
    """Iterate experiments (columns). Cmask: (n, m) bool, committed agents; Cval: their values.
    Returns (V, converged: bool array, iterations)."""
    S = np.where(Cmask, 0.0, s[:, None])
    U = np.where(Cmask, Cval, mu)
    V = np.where(Cmask, Cval, V0)
    for it in range(1, maxit + 1):
        Vn = S * _phi(W, V, g, alpha, eta, eps) + (1 - S) * U
        d = np.max(np.abs(Vn - V), axis=0)
        V = Vn
        if np.all(d < tol):
            return V, np.ones(V.shape[1], bool), it
    return V, d < tol, maxit


def source_experiments(W, s, g, q, alpha=None, eta=None, V0=None, tol=TOL):
    n = len(s)
    Cmask = np.eye(n, dtype=bool)
    V0 = np.zeros((n, n)) if V0 is None else V0
    V, conv, it = iterate(W, s, g, Cmask, q, V0, alpha, eta, tol=tol)
    return V.T, conv, it      # V.T[a, b] = U_ab(q)


def committed_state(W, s, g, sign=-1.0, committed=None, alpha=None, eta=None, eps=0.0):
    """Ordered state of the population with `committed` (list of index arrays, one per column)
    held at 0, reached from the uniform state `sign`."""
    n = len(s)
    if committed is None:
        Cmask = np.zeros((n, 1), bool)
    else:
        Cmask = np.zeros((n, len(committed)), bool)
        for k, c in enumerate(committed):
            Cmask[c, k] = True
    V0 = np.full(Cmask.shape, sign)
    V, conv, _ = iterate(W, s, g, Cmask, 0.0, V0, alpha, eta, eps)
    return V, conv


def flip_thresholds(W, s, g, steps=24, tol=1e-9):
    """Single-source flip threshold q_c(a) (inf if q = 1 does not reverse the population).
    Population: interior V \\ {a}, ordered negative state with a held at 0; source a then set to q."""
    n = len(s)
    Cmask = np.eye(n, dtype=bool)
    Vneg, _, _ = iterate(W, s, g, Cmask, 0.0, -np.ones((n, n)), tol=tol)
    def flipped(q):
        V, _, _ = iterate(W, s, g, Cmask, q[None, :], Vneg, tol=tol)
        m = (V.sum(0) - np.diag(V)) / (n - 1)
        return m > 0
    lo, hi = np.zeros(n), np.ones(n)
    ok = flipped(hi)
    for _ in range(steps):
        mid = (lo + hi) / 2
        f = flipped(mid)
        hi = np.where(f, mid, hi); lo = np.where(f, lo, mid)
    return np.where(ok, hi, np.inf)


def tipping_fraction(W, s, g, order, sign=+1.0, eps=0.0, Vstart=None, tol=1e-9):
    """Smallest k/n such that committing order[:k] at value `sign` reverses the population,
    which starts in the ordered state of opposite sign (no committed agents). nan if k=n-1 fails."""
    n = len(s)
    if Vstart is None:
        Vstart, _ = committed_state(W, s, g, -sign, eps=eps)
        Vstart = Vstart[:, 0]
    if np.mean(Vstart) * sign > 0:          # no state of opposite sign exists
        return 0.0
    def rev(k):
        C = np.zeros((n, 1), bool); C[order[:k], 0] = True
        V, _, _ = iterate(W, s, g, C, sign, Vstart[:, None], eps=eps, tol=tol)
        return sign * np.mean(V[~C[:, 0], 0]) > 0
    if not rev(n - 1):
        return np.nan
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if rev(mid): hi = mid
        else: lo = mid
    return hi / n


# ----------------------------------------------------------------- centralities
def perron_left(M):
    w, V = np.linalg.eig(M.T)
    v = np.abs(np.real(V[:, np.argmax(np.real(w))]))
    return v / v.sum()


def centralities(W, s, g):
    n = len(s)
    out = {}
    out['Out-strength'] = W.sum(0)
    out['Eigenvector'] = perron_left(W)
    out['Katz'] = np.linalg.inv(np.eye(n) - 0.85 * W).sum(0)
    B, _ = scores(fj_response(W, s))
    out['FJ broadcasting'] = B
    Vneg, _ = committed_state(W, s, g, -1.0)
    v = Vneg[:, 0]
    J = s[:, None] * W * (g * (1 - np.tanh(g * v) ** 2))[None, :]
    out['Jacobian Perron'] = perron_left(J)     # sensitivity to infinitesimal inputs at the ordered state
    return out


# ----------------------------------------------------------------- full scans and local response
def ordered_green(W, s, g, Vneg=None):
    """Ordered-state local response to clamping: score_a = mean_{i != a} G_ia / G_aa,
    G = (I - J)^{-1}, J = DT at the negative ordered state of the whole network."""
    n = len(s)
    if Vneg is None:
        Vneg = committed_state(W, s, g, -1.0)[0][:, 0]
    Wd = W.toarray() if hasattr(W, 'toarray') else W
    J = s[:, None] * Wd * (g * (1 - np.tanh(g * Vneg) ** 2))[None, :]
    G = np.linalg.inv(np.eye(n) - J)
    R = G / np.diag(G)[None, :]          # R[i, a] = G_ia / G_aa
    return (R.sum(0) - 1.0) / (n - 1)


def tipping_scan(W, s, g, order, sign=+1.0, eps=0.0, Vstart=None, tol=1e-10, maxit=MAXIT):
    """Full scan over k = 1..n-1 by exact continuation (nested committed sets, order-preserving map).
    Returns dict: pc (first k/n with reversed mean, nan if none), interval (no recrossing),
    all_converged."""
    n = len(s)
    if Vstart is None:
        Vstart = committed_state(W, s, g, -sign, eps=eps)[0][:, 0]
    V = Vstart.copy(); C = np.zeros((n, 1), bool); pred = []; conv_all = True
    for k in range(1, n):
        C[order[k - 1], 0] = True
        Vk, conv, _ = iterate(W, s, g, C, sign, V[:, None], eps=eps, tol=tol, maxit=maxit)
        V = Vk[:, 0]; conv_all &= bool(conv.all())
        pred.append(sign * np.mean(V[~C[:, 0]]) > 0)
    pred = np.array(pred)
    first = int(np.argmax(pred)) + 1 if pred.any() else None
    interval = (not pred.any()) or bool(pred[first - 1:].all())
    return dict(pc=(first / n if first else np.nan), interval=interval, all_converged=conv_all)
