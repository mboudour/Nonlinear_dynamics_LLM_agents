import sys, json, pathlib, os
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / 'results'; FIG = ROOT / 'figures'
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
def save(name, obj):
    with open(RES / name, 'w') as f: json.dump(obj, f, indent=1, default=float)
def load(name):
    with open(RES / name) as f: return json.load(f)
