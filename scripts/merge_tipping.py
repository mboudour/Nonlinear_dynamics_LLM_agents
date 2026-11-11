"""Merge per-profile tipping runs (exp_tipping_part.py) into E3_tipping_<network>.json."""
from _common import *
import glob, json
for net in ['Directed_heavy-tailed']:
    rows = []
    for f in sorted(glob.glob(str(RES / f'part_{net}_*.json'))):
        rows += json.load(open(f))
    save(f'E3_tipping_{net}.json', rows); print(net, len(rows))
