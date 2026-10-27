# Stacker: LightGBM on physics features + log CNN probabilities (OOF), same folds. Then split-half class multipliers.
# usage: python src/stack.py <name> <phys_version> <cnn_name>[,<cnn_name>...]
import sys, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score, log_loss
name, pv, cnns = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
tr = pd.read_csv('train_meta.csv'); te = pd.read_csv('test_meta.csv')
L = sorted(tr.label.unique()); y = tr.label.map({l: i for i, l in enumerate(L)}).values
folds = np.load('oof/folds.npy')
Ftr = pd.read_parquet(f'cache/train_phys{pv}.parquet'); Fte = pd.read_parquet(f'cache/test_phys{pv}.parquet')[Ftr.columns]
for c in cnns:
    o = np.load(f'oof/{c}__oof.npy'); assert (o.sum(1) > 0).all(), f'{c} OOF incomplete'
    t = np.mean([np.load(f'oof/{c}__test_f{k}.npy') for k in range(5)], 0)
    for k in range(9): Ftr[f'{c}_{L[k]}'] = np.log(o[:, k] + 1e-6); Fte[f'{c}_{L[k]}'] = np.log(t[:, k] + 1e-6)
P = dict(objective='multiclass', num_class=9, learning_rate=0.03, num_leaves=15, min_child_samples=30, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=3, verbose=-1)
oof = np.zeros((len(y), 9)); tst = np.zeros((len(te), 9))
for k in range(5):
    a, b = folds != k, folds == k
    m = lgb.train(P, lgb.Dataset(Ftr[a], y[a]), 3000, valid_sets=[lgb.Dataset(Ftr[b], y[b])], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[b] = m.predict(Ftr[b], num_iteration=m.best_iteration); tst += m.predict(Fte, num_iteration=m.best_iteration) / 5
np.save(f'oof/{name}__oof.npy', oof); np.save(f'oof/{name}__test.npy', tst)
print(name, 'OOF F1 %.4f LL %.4f' % (f1_score(y, oof.argmax(1), average='macro'), log_loss(y, oof)))
print(dict(zip(L, np.round(f1_score(y, oof.argmax(1), average=None), 3))))

# split-half class multipliers (coarse grid, coordinate ascent), honest score = fit on A, score on B and swap
GRID = np.array([0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0])
def fit_w(p, yy):
    w = np.ones(9)
    for _ in range(3):
        for c in range(9):
            best = (f1_score(yy, (p * w).argmax(1), average='macro'), w[c])
            for g in GRID:
                w2 = w.copy(); w2[c] = g; s = f1_score(yy, (p * w2).argmax(1), average='macro')
                if s > best[0] + 1e-4: best = (s, g)
            w[c] = best[1]
    return w
rng = np.random.default_rng(0); half = np.zeros(len(y), bool)
for c in range(9):
    ix = np.where(y == c)[0]; half[rng.choice(ix, len(ix) // 2, replace=False)] = True
wA = fit_w(oof[half], y[half]); wB = fit_w(oof[~half], y[~half])
base = [f1_score(y[h], oof[h].argmax(1), average='macro') for h in (~half, half)]
hon = [f1_score(y[~half], (oof[~half] * wA).argmax(1), average='macro'), f1_score(y[half], (oof[half] * wB).argmax(1), average='macro')]
print('multipliers honest: base %.4f -> %.4f' % (np.mean(base), np.mean(hon)), 'wA', wA.round(2).tolist(), 'wB', wB.round(2).tolist())
w = np.sqrt(wA * wB)                                                      # geometric mean of the two halves
json.dump({'w': w.tolist(), 'honest_gain': float(np.mean(hon) - np.mean(base))}, open(f'oof/{name}__w.json', 'w'))
for nm, ww in [('raw', np.ones(9)), ('w', w)]:
    print('test dist', nm, dict(zip(L, np.bincount((tst * ww).argmax(1), minlength=9).tolist())))
