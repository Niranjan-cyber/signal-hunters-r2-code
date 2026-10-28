# Paired bootstrap B vs A on held-out sim (oofsimt), PKS-sim (oofsim) and clean (oof), plus held-out per-class F1.
# usage: python src/paired.py <stackA> <stackB>
import sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score
A, B = sys.argv[1], sys.argv[2]
tr = pd.read_csv('train_meta.csv'); L = sorted(tr.label.unique()); y = tr.label.map({l: i for i, l in enumerate(L)}).values
rng = np.random.default_rng(0); BS = [rng.integers(0, len(y), len(y)) for _ in range(200)]
def ll(p, ix): return -np.log(p[ix, y[ix]]).mean()
def f1(p, ix): return f1_score(y[ix], p[ix].argmax(1), average='macro')
for part in ['oofsimt', 'oofsim', 'oof']:
    pa, pb = [np.clip(np.load(f'oof/{n}__{part}.npy'), 1e-7, 1) for n in (A, B)]
    out = []
    for nm, fn in [('LL', ll), ('F1', f1)]:
        d = np.array([fn(pb, b) - fn(pa, b) for b in BS]); full = fn(pb, np.arange(len(y))) - fn(pa, np.arange(len(y)))
        out.append(f'{nm} {fn(pa, np.arange(len(y))):.4f}->{fn(pb, np.arange(len(y))):.4f} d {full:+.4f} ({full / d.std():+.1f}sd)')
    print(f'{part:8s} ' + ' | '.join(out))
pa, pb = np.load(f'oof/{A}__oofsimt.npy'), np.load(f'oof/{B}__oofsimt.npy')
ca, cb = f1_score(y, pa.argmax(1), average=None), f1_score(y, pb.argmax(1), average=None)
print('held-out per-class dF1', {l[:6]: round(float(b - a), 4) for l, a, b in zip(L, ca, cb)})
ta, tb = np.load(f'oof/{A}__test.npy'), np.load(f'oof/{B}__test.npy')
print('test argmax changes', int((ta.argmax(1) != tb.argmax(1)).sum()))
