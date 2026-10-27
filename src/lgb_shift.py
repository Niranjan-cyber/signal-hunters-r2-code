# A2: physics LGB trained on clean vs SNR-degraded train features; each scored on clean and degraded (PKS-sim) validation.
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score, log_loss
tr = pd.read_csv('train_meta.csv'); L = sorted(tr.label.unique()); y = tr.label.map({l: i for i, l in enumerate(L)}).values
C = pd.read_parquet('cache/train_phys2f.parquet'); S = pd.read_parquet('cache/train_phys2sf.parquet')[C.columns]
Te = pd.read_parquet('cache/test_phys2f.parquet')[C.columns]
folds = np.load('oof/folds.npy')
P = dict(objective='multiclass', num_class=9, learning_rate=0.05, num_leaves=31, min_child_samples=30, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1, verbose=-1)
for tn, Xtr in [('clean', C), ('degr', S), ('mix', None)]:
    oc, od, te = np.zeros((len(y), 9)), np.zeros((len(y), 9)), np.zeros((len(Te), 9))
    for k in range(5):
        a, b = folds != k, folds == k
        X = pd.concat([C[a], S[a]]) if tn == 'mix' else Xtr[a]; Y = np.r_[y[a], y[a]] if tn == 'mix' else y[a]
        V = S[b] if tn != 'clean' else C[b]
        m = lgb.train(P, lgb.Dataset(X, Y), 2000, valid_sets=[lgb.Dataset(V, y[b])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oc[b] = m.predict(C[b], num_iteration=m.best_iteration); od[b] = m.predict(S[b], num_iteration=m.best_iteration)
        te += m.predict(Te, num_iteration=m.best_iteration) / 5
    np.save(f'oof/lgbA2_{tn}__oof.npy', oc); np.save(f'oof/lgbA2_{tn}__oofsim.npy', od); np.save(f'oof/lgbA2_{tn}__test.npy', te)
    print(f'{tn:5s} CLEAN F1 {f1_score(y, oc.argmax(1), average="macro"):.4f} LL {log_loss(y, oc):.4f} | SIM F1 {f1_score(y, od.argmax(1), average="macro"):.4f} LL {log_loss(y, od):.4f}',
          '| test dist', np.bincount(te.argmax(1), minlength=9).tolist(), flush=True)
