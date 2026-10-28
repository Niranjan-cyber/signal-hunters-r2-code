# Class-specific log-probability biases for macro-F1, fitted on half A of the held-out-simulated OOF and scored on
# half B (and swapped); also checked on PKS-sim and clean OOF. usage: python src/class_bias.py <stack name>
import sys, json, numpy as np, pandas as pd
from sklearn.metrics import f1_score
name = sys.argv[1]
tr = pd.read_csv('train_meta.csv'); L = sorted(tr.label.unique()); y = tr.label.map({l: i for i, l in enumerate(L)}).values
P = {k: np.log(np.clip(np.load(f'oof/{name}__{k}.npy'), 1e-9, 1)) for k in ['oofsimt', 'oofsim', 'oof']}
GRID = np.array([-0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6])
def f1(lp, b, ix): return f1_score(y[ix], (lp[ix] + b).argmax(1), average='macro')
def fit(lp, ix):
    b = np.zeros(9)
    for _ in range(3):
        for c in range(9):
            best = (f1(lp, b, ix), b[c])
            for g in GRID:
                b2 = b.copy(); b2[c] = g; s = f1(lp, b2, ix)
                if s > best[0] + 2e-4: best = (s, g)
            b[c] = best[1]
    return b
rng = np.random.default_rng(0); A = np.zeros(len(y), bool)
for c in range(9):
    ix = np.where(y == c)[0]; A[rng.choice(ix, len(ix) // 2, replace=False)] = True
bA, bB = fit(P['oofsimt'], np.where(A)[0]), fit(P['oofsimt'], np.where(~A)[0])
z = np.zeros(9); iA, iB = np.where(A)[0], np.where(~A)[0]
print('held-out honest: base %.4f -> %.4f' % ((f1(P['oofsimt'], z, iB) + f1(P['oofsimt'], z, iA)) / 2, (f1(P['oofsimt'], bA, iB) + f1(P['oofsimt'], bB, iA)) / 2))
b = (bA + bB) / 2
print('biases', dict(zip([l[:6] for l in L], b.round(2))))
for k in ['oofsim', 'oof']:
    al = np.arange(len(y)); print(f'{k}: base {f1(P[k], z, al):.4f} -> with biases {f1(P[k], b, al):.4f}')
json.dump({'bias': b.tolist()}, open(f'oof/{name}__bias.json', 'w'))
