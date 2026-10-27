import numpy as np, pandas as pd, lightgbm as lgb, sys
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict
name = sys.argv[1] if len(sys.argv) > 1 else 'lgb_phys_v1'; v = sys.argv[2] if len(sys.argv) > 2 else ''; drop = sys.argv[3].split(',') if len(sys.argv) > 3 else []
tr = pd.read_csv('train_meta.csv'); LABELS = sorted(tr.label.unique()); y = tr.label.map({l: i for i, l in enumerate(LABELS)}).values
Ftr = pd.read_parquet(f'cache/train_phys{v}.parquet').drop(columns=drop, errors='ignore'); Fte = pd.read_parquet(f'cache/test_phys{v}.parquet')[Ftr.columns]
# adversarial
A = pd.concat([Ftr.sample(8000, random_state=0), Fte.sample(8000, random_state=0)]); ya = np.r_[np.zeros(8000), np.ones(8000)]
pa = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, verbose=-1), A, ya, cv=3, method='predict_proba')[:, 1]
imp = lgb.LGBMClassifier(n_estimators=200, verbose=-1).fit(A, ya, ).booster_.feature_importance('gain')
print('adversarial AUC %.3f' % roc_auc_score(ya, pa), 'top:', [c for _, c in sorted(zip(imp, A.columns))[-6:]])
folds = np.load('oof/folds.npy'); oof = np.zeros((len(y), 9)); tst = np.zeros((len(Fte), 9))
P = dict(objective='multiclass', num_class=9, learning_rate=0.05, num_leaves=31, min_child_samples=30, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1, verbose=-1)
for k in range(5):
    tri, vai = folds != k, folds == k
    m = lgb.train(P, lgb.Dataset(Ftr[tri], y[tri]), 2000, valid_sets=[lgb.Dataset(Ftr[vai], y[vai])], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[vai] = m.predict(Ftr[vai], num_iteration=m.best_iteration); tst += m.predict(Fte, num_iteration=m.best_iteration) / 5
np.save(f'oof/{name}__oof.npy', oof); np.save(f'oof/{name}__test.npy', tst)
print(name, 'OOF F1 %.4f LL %.4f | fold0 F1 %.4f LL %.4f' % (f1_score(y, oof.argmax(1), average='macro'), log_loss(y, oof), f1_score(y[folds == 0], oof[folds == 0].argmax(1), average='macro'), log_loss(y[folds == 0], oof[folds == 0])))
print(pd.Series(f1_score(y, oof.argmax(1), average=None), index=LABELS).round(3).to_dict())
print('test dist', dict(zip(LABELS, np.bincount(tst.argmax(1), minlength=9))))
