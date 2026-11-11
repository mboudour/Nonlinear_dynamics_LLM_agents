# Nonlinear boundary-value dynamics of LLM agents

Code and numerical results for the manuscript *Nonlinear boundary-value dynamics of LLM agents:
committed sources, channel distortion, and response centralities* (M. Boudourides).

## Contents

- `llmbvp/core.py` – the model (Eq. 1–2 of the paper), benchmark networks, critical gains,
  source-indexed experiments, flip thresholds, tipping fractions and targeting centralities.
- `scripts/exp_*.py` – one script per experiment; each writes a JSON file to `results/`.
  - `exp_rankings.py` – response centralities across the gain (Fig. 1)
  - `exp_flip.py` – flip thresholds over all profiles and hysteresis loops (Fig. 2)
  - `exp_tipping.py`, `exp_tipping_onset.py` – committed-minority tipping and targeting rules (Fig. 3).
    `exp_tipping_part.py` runs one network and one susceptibility profile with identical logic and seeds
    (used to parallelize the heavy-tailed network); `merge_tipping.py` merges the parts.
  - `exp_meanfield.py` – mean-field test on complete and random-regular graphs (Fig. 4)
  - `exp_bias.py` – biased channels (Fig. 5)
  - `exp_scan.py` – full integer scans (exact continuation) of every tipping sequence in Figs. 3–5, adding the
    ordered-state Green rule; Figs. 3 and 5 report these first crossings (`results/E7_scans.json`)
  - `exp_robust.py` – robustness of the targeting comparison to density and susceptibility law
  - `exp_families.py` – two model families (Fig. 6)
- `scripts/make_figures.py` – builds all figures in `figures/` from `results/`.
- `tests/check_theory.py` – numerical checks of the analytical results.
- `tests/check_hypotheses.py` – checks the hypotheses of Theorem 4 in every flip experiment.
- `results/` – the JSON outputs used in the paper.

## Reproduce

```
pip install -r requirements.txt
python scripts/run_all.py          # all experiments and figures
python tests/check_theory.py       # checks of the theorems
```

Seeds are fixed in the code (networks: 11, 12, 14; susceptibility profile p: 1000 + p;
random orders: see `exp_tipping.py`), so results are reproducible exactly with the pinned versions.
Tested with Python 3.13.
