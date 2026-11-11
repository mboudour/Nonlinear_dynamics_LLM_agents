"""Run every experiment and build all figures (about 1.5 h on one core)."""
import subprocess, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
for s in ['exp_rankings', 'exp_flip', 'exp_tipping', 'exp_tipping_onset', 'exp_meanfield',
          'exp_bias', 'exp_families', 'exp_scan', 'exp_robust', 'make_figures']:
    print('running', s, flush=True)
    subprocess.run([sys.executable, str(HERE / f'{s}.py')], check=True)
