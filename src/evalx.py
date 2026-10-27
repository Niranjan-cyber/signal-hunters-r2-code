# Paired bootstrap of fold/OOF comparisons: python src/evalx.py nameA nameB [fold]
import numpy as np, pandas as pd, sys
from sklearn.metrics import f1_score, log_loss
tr = pd.read_csv('train_meta.csv'); LABELS = sorted(tr.label.unique()); y = tr.label.map({l: i for i, l in enumerate(LABELS)}).values
folds = np.load('oof/folds.npy'); fold = int(sys.argv[3]) if len(sys.argv) > 3 else None
idx = np.where(folds == fold)[0] if fold is not None else np.arange(len(y))
P = [np.clip(np.load(f'oof/{n}__oof.npy')[idx], 1e-7, 1) for n in sys.argv[1:3]]; yy = y[idx]
def ll(p, ix): return -np.log(p[ix, yy[ix]]).mean()
def f1(p, ix): return f1_score(yy[ix], p[ix].argmax(1), average='macro')
rng = np.random.default_rng(0); B = [rng.integers(0, len(yy), len(yy)) for _ in range(200)]
for nm, fn in [('LL', ll), ('F1', f1)]:
    a = np.array([fn(P[0], b) for b in B]); b_ = np.array([fn(P[1], b) for b in B])
    print(f'{nm}: A {fn(P[0], np.arange(len(yy))):.4f} B {fn(P[1], np.arange(len(yy))):.4f} | sd A {a.std():.4f} | paired diff B-A {np.mean(b_ - a):+.4f} sd {np.std(b_ - a):.4f}')
