# Physics LGB with confident test pseudo-labels added to every fold's training set (validation train-only).
import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score, log_loss
name, v, pfile, thr = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
tr = pd.read_csv('train_meta.csv'); L = sorted(tr.label.unique()); y = tr.label.map({l: i for i, l in enumerate(L)}).values
Ftr = pd.read_parquet(f'cache/train_phys{v}.parquet'); Fte = pd.read_parquet(f'cache/test_phys{v}.parquet')[Ftr.columns]
P = np.load(pfile); sel = P.max(1) >= thr; Fp, yp = Fte[sel], P[sel].argmax(1); print('pseudo rows', sel.sum())
folds = np.load('oof/folds.npy'); oof = np.zeros((len(y), 9)); tst = np.zeros((len(Fte), 9))
Pm = dict(objective='multiclass', num_class=9, learning_rate=0.05, num_leaves=31, min_child_samples=30, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1, verbose=-1)
for k in range(5):
    a, b = folds != k, folds == k
    X = pd.concat([Ftr[a], Fp]); Y = np.r_[y[a], yp]
    m = lgb.train(Pm, lgb.Dataset(X, Y), 2000, valid_sets=[lgb.Dataset(Ftr[b], y[b])], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[b] = m.predict(Ftr[b], num_iteration=m.best_iteration); tst += m.predict(Fte, num_iteration=m.best_iteration) / 5
np.save(f'oof/{name}__oof.npy', oof); np.save(f'oof/{name}__test.npy', tst)
print(name, 'OOF F1 %.4f LL %.4f' % (f1_score(y, oof.argmax(1), average='macro'), log_loss(y, oof)))
print('test dist', dict(zip([l[:6] for l in L], np.bincount(tst.argmax(1), minlength=9))))
