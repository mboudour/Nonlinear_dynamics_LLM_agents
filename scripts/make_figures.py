"""All manuscript figures from results/*.json."""
from _common import *
import numpy as np, pandas as pd, glob, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.linewidth': 0.6, 'lines.linewidth': 1.3,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'axes.edgecolor': '#555555', 'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
MK = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
NETS = ['Karate club', 'Les Misérables', 'Directed ER', 'Directed SBM', 'Directed heavy-tailed']
def lab(ax, k, x=-0.2): ax.text(x, 1.04, '(%s)' % 'abcdef'[k], transform=ax.transAxes, fontweight='bold')

# Fig 1 ---------------------------------------------------------------
E1 = load('E1_rankings.json')
fig, ax = plt.subplots(2, 2, figsize=(6.8, 4.4)); ax = ax.ravel()
keys = [('rhoB_fj', r'Spearman$(B,\,B^{\mathrm{FJ}})$', 'linear'), ('rhoB_stub', r'Spearman$(B,\,-s)$', 'linear'),
        ('cvB', r'coefficient of variation of $B$', 'log'), ('rhoQ_s', r'Spearman$(Q,\,s)$', 'linear')]
for k, (key, yl, sc) in enumerate(keys):
    for i, net in enumerate(NETS):
        d = pd.DataFrame(E1[net]).groupby('ratio')[key].mean()
        ax[k].plot(d.index, d.values, color=C[i], marker=MK[i], ms=3, label=net)
    ax[k].set_xscale('log'); ax[k].set_yscale(sc); ax[k].axvline(1, color='#999999', lw=0.7, ls=':')
    ax[k].set_xlabel(r'$g/g^*$'); ax[k].set_ylabel(yl); lab(ax[k], k)
ax[1].legend(fontsize=6.3, loc='center left', bbox_to_anchor=(0.0, 0.62))
fig.tight_layout(); fig.savefig(FIG / 'fig1_rankings.pdf'); plt.close()

# Fig 2 ---------------------------------------------------------------
L = load('E2_flip.json')['loops']; q = np.array(L['q']); h = len(q) // 2
fig, ax = plt.subplots(figsize=(3.4, 2.4))
names = {'33': 'agent 34', '0': 'agent 1', '11': 'agent 12'}
for i, a in enumerate(['33', '0', '11']):
    y = np.array(L[a])
    ax.plot(q[:h], y[:h], color=C[i], marker=MK[i], ms=2.2, markevery=16, label=names[a] + r', $q$ increasing')
    ax.plot(q[h:], y[h:], color=C[i], ls='--', lw=1.0, label=names[a] + r', $q$ decreasing')
ax.axhline(0, color='#999999', lw=0.5, ls=':'); ax.axvline(0, color='#999999', lw=0.5, ls=':')
ax.set_xlabel('value $q$ of the committed agent'); ax.set_ylabel(r'mean opinion of the others')
ax.legend(fontsize=5.6, loc='upper left'); fig.tight_layout(); fig.savefig(FIG / 'fig2_hysteresis.pdf'); plt.close()

# Fig 3 ---------------------------------------------------------------
SC = load('E7_scans.json')
RULES = ['Out-strength', 'Katz', 'FJ broadcasting', 'Onset Perron', 'Jacobian Perron', 'Ordered Green', 'Random']
fig, ax = plt.subplots(1, 5, figsize=(7.0, 2.45), sharey=True)
for k, net in enumerate(NETS):
    d = pd.DataFrame(SC['tipping'][net])
    for j, rule in enumerate(RULES):
        dd = d[d.rule == rule]
        if rule == 'Random':
            dd = dd.groupby(['p', 'ratio']).pc.mean().reset_index()
        st = dd.groupby('ratio').pc.agg(['mean', 'sem'])
        ax[k].errorbar(st.index, st['mean'], yerr=st['sem'].fillna(0), color=C[j] if rule != 'Random' else '#555555',
                       marker=MK[j], ms=2.6, lw=1.0, capsize=1.5, label=rule)
    ax[k].set_title(net, fontsize=7); ax[k].set_xlabel(r'$g/g^*$')
ax[0].set_ylabel(r'tipping fraction $p_c$')
h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, loc='lower center', ncol=7, fontsize=6.0)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(FIG / 'fig3_tipping.pdf'); plt.close()

# Fig 4 ---------------------------------------------------------------
M = load('E4_meanfield.json'); b = np.array(M['beta'])
from scipy.optimize import brentq
def Hc(x):
    if x <= 1: return 0.0
    t = np.sqrt(1 - 1 / x); return x * t - np.arctanh(t)
bb = np.linspace(1.002, 5, 400); S = 0.65
mf = [brentq(lambda p: B * p * np.tanh(B / S) - Hc(B * (1 - p)), 1e-15, 1 - 1e-12) for B in bb]
rr = np.array(M['rr']); fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.4))
for k in range(2):
    x = (bb - 1) if k else bb
    ax[k].plot(x, mf, color='#555555', lw=1.0, label='mean field')
    xs = (b - 1) if k else b
    ax[k].plot(xs, M['complete'], ls='none', marker='o', ms=3.5, color=C[0], label='complete graph, $n=1000$')
    ax[k].errorbar(xs, rr.mean(1), yerr=rr.std(1), ls='none', marker='s', ms=3.2, color=C[1], capsize=1.5,
                   label='random 10-regular, $n=2000$')
ax[0].set_xlabel(r'$\beta=g/g^*$'); ax[0].set_ylabel(r'tipping fraction $p_c$'); ax[0].legend(fontsize=6.2)
xg = np.array([0.02, 0.3]); ax[1].plot(xg, 0.75 * 2 / (3 * np.tanh(1 / S)) * xg ** 1.5, color=C[2], ls=':', label='slope 3/2')
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel(r'$\beta-1$'); ax[1].set_ylabel(r'$p_c$')
ax[1].legend(fontsize=6.2, loc='upper left'); lab(ax[0], 0); lab(ax[1], 1)
fig.tight_layout(); fig.savefig(FIG / 'fig4_meanfield.pdf'); plt.close()

# Fig 5 ---------------------------------------------------------------
Bi = load('E7_scans.json')['bias']; fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
for k, net in enumerate(['Karate club', 'Directed ER']):
    d = pd.DataFrame(Bi[net]); j = 0
    for rule in ['Random', 'FJ broadcasting']:
        for tw, ls in [('favoured', '-'), ('disfavoured', '--')]:
            dd = d[(d.rule == rule) & (d.toward == tw)]
            if rule == 'Random': dd = dd.groupby(['p', 'eps']).pc.mean().reset_index()
            st = dd.groupby('eps').pc.agg(['mean', 'sem'])
            ax[k].errorbar(st.index, st['mean'], yerr=st['sem'], color=C[0] if rule == 'Random' else C[1], ls=ls,
                           marker=MK[j], ms=3, capsize=1.5, label=f"{rule}, toward {tw.replace('favoured', 'favored')} side"); j += 1
    ax[k].set_title(net, fontsize=7.5); ax[k].set_xlabel(r'channel bias $\varepsilon$')
ax[0].set_ylabel(r'tipping fraction $p_c$')
h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, loc='lower center', ncol=2, fontsize=6.3)
lab(ax[0], 0); lab(ax[1], 1, -0.08); fig.tight_layout(rect=(0, 0.16, 1, 1)); fig.savefig(FIG / 'fig5_bias.pdf'); plt.close()

# Fig 6 ---------------------------------------------------------------
F = pd.DataFrame(load('E6_families.json')); fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.4))
for i, f in enumerate([0.1, 0.3, 0.5]):
    d = F[F.f == f]
    ax[0].plot(d.h, [x['gstar_quot'] for x in d.hetero], color=C[i], lw=1.0, label=rf'$\phi_\mathcal{{A}}={f}$, quotient')
    ax[0].plot(d.h, [x['gstar'] for x in d.hetero], color=C[i], ls='none', marker=MK[i], ms=3.2, label=rf'$\phi_\mathcal{{A}}={f}$, network')
d = F[F.f == 0.1]
ax[1].plot(d.h, d.mean_A, color=C[0], marker='o', ms=3, label=r'amplifying family $\mathcal{A}$')
ax[1].plot(d.h, d.mean_B, color=C[1], marker='s', ms=3, label=r'attenuating family $\mathcal{B}$')
ax[0].set_xlabel(r'homophily index $h$'); ax[0].set_ylabel(r'critical gain $g^*$'); ax[0].legend(fontsize=5.8, ncol=1, loc='upper right')
ax[1].set_xlabel(r'homophily index $h$'); ax[1].set_ylabel(r'mean opinion, committed state'); ax[1].legend(fontsize=6.2)
lab(ax[0], 0); lab(ax[1], 1); fig.tight_layout(); fig.savefig(FIG / 'fig6_families.pdf'); plt.close()
print('figures done')
