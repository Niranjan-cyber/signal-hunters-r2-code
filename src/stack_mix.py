# Shift-robust stacker: LGB on [physics feats + log CNN probs], trained on clean rows only ('clean') or on clean + PKS-sim rows
# ('mix': degraded physics feats + CNN probs on PKS-sim validation). Scored on clean OOF and PKS-sim OOF.
# usage: python src/stack_mix.py <name> <clean|mix> <cnn>[,<cnn>...]
import sys, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score, log_loss
name, mode, cnns = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
extra = sys.argv[4].split(',') if len(sys.argv) > 4 else []
tr = pd.read_csv('train_meta.csv'); te = pd.read_csv('test_meta.csv')
L = sorted(tr.label.unique()); y = tr.label.map({l: i for i, l in enumerate(L)}).values
folds = np.load('oof/folds.npy')
C = pd.read_parquet('cache/train_phys2f.parquet'); S = pd.read_parquet('cache/train_phys2sf.parquet')[C.columns]
T = pd.read_parquet('cache/test_phys2f.parquet')[C.columns]
C, S, T = C.copy(), S.copy(), T.copy()
import os                                                                      # H = held-out simulator (s 0.75, seed 7): scored, never trained on
H = pd.read_parquet('cache/train_phys2tf.parquet')[C.columns].copy() if os.path.exists('cache/train_phys2tf.parquet') else None
if 'lv' in extra:                                                              # panel-level (scintillation) features
    LVC = ['lv_alt', 'lv_onmin_offmax', 'lv_on_mean', 'lv_off_mean', 'lv_alt_raw']
    a_, t_ = pd.read_parquet('cache/train_lv.parquet')[LVC], pd.read_parquet('cache/test_lv.parquet')[LVC]
    for c in LVC:
        C[c] = a_[c].values; S[c] = a_[c].values / np.sqrt(1 + 0.92 ** 2); T[c] = t_[c].values
        if H is not None:                                                      # 'lvg' = held-out rows also get PKS-like panel gain offsets
            g_ = pd.read_parquet('cache/train_lv_g.parquet')[LVC] if 'lvg' in extra else a_
            H[c] = g_[c].values / np.sqrt(1 + 0.75 ** 2)
if 'wide' in extra:                                                            # 3-channel matched-filter ETI track scores
    a_, s_, t_ = pd.read_parquet('cache/train_wide2.parquet'), pd.read_parquet('cache/train_wide2s.parquet'), pd.read_parquet('cache/test_wide2.parquet')
    h_ = pd.read_parquet('cache/train_wide2t.parquet') if H is not None and os.path.exists('cache/train_wide2t.parquet') else None
    if h_ is None: H = None
    for c in a_.columns:
        C[c] = a_[c].values; S[c] = s_[c].values; T[c] = t_[c].values
        if H is not None: H[c] = h_[c].values
for c in cnns:
    o = np.load(f'oof/{c}__oof.npy'); s = np.load(f'oof/{c}__oofsim.npy')
    assert (o.sum(1) > 0.99).all() and (s.sum(1) > 0.99).all(), f'{c} OOF/oofsim incomplete'
    t = np.mean([np.load(f'oof/{c}__test_f{k}.npy') for k in range(5)], 0)
    ht = np.load(f'oof/{c}__oofsimt.npy') if H is not None and os.path.exists(f'oof/{c}__oofsimt.npy') else None
    if ht is None or not (ht.sum(1) > 0.99).all(): H = None
    for k in range(9):
        C[f'{c}_{k}'] = np.log(o[:, k] + 1e-6); S[f'{c}_{k}'] = np.log(s[:, k] + 1e-6); T[f'{c}_{k}'] = np.log(t[:, k] + 1e-6)
        if H is not None: H[f'{c}_{k}'] = np.log(ht[:, k] + 1e-6)
P = dict(objective='multiclass', num_class=9, learning_rate=0.03, num_leaves=15, min_child_samples=30, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=3, verbose=-1)
sd = [int(e[4:]) for e in extra if e.startswith('seed')]                     # 'seed<N>' in extra = LightGBM seed (bagging run)
if sd: P.update(seed=sd[0], bagging_seed=sd[0], feature_fraction_seed=sd[0], data_random_seed=sd[0])
oc, os_, oh, tst = np.zeros((len(y), 9)), np.zeros((len(y), 9)), np.zeros((len(y), 9)), np.zeros((len(T), 9))
for k in range(5):
    a, b = folds != k, folds == k
    X = pd.concat([C[a], S[a]]) if mode == 'mix' else C[a]; Y = np.r_[y[a], y[a]] if mode == 'mix' else y[a]
    V = pd.concat([C[b], S[b]]); VY = np.r_[y[b], y[b]]                       # early stop on clean + sim
    m = lgb.train(P, lgb.Dataset(X, Y), 3000, valid_sets=[lgb.Dataset(V, VY)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oc[b] = m.predict(C[b], num_iteration=m.best_iteration); os_[b] = m.predict(S[b], num_iteration=m.best_iteration)
    if H is not None: oh[b] = m.predict(H[C.columns][b], num_iteration=m.best_iteration)
    tst += m.predict(T, num_iteration=m.best_iteration) / 5
np.save(f'oof/{name}__oof.npy', oc); np.save(f'oof/{name}__oofsim.npy', os_); np.save(f'oof/{name}__test.npy', tst)
f1c, f1s = f1_score(y, oc.argmax(1), average='macro'), f1_score(y, os_.argmax(1), average='macro')
print(f'{name} [{mode}] CLEAN F1 {f1c:.4f} LL {log_loss(y, oc):.4f} | SIM F1 {f1s:.4f} LL {log_loss(y, os_):.4f}')
if H is not None:
    np.save(f'oof/{name}__oofsimt.npy', oh)
    print(f'  HELD-OUT SIM (s0.75) F1 {f1_score(y, oh.argmax(1), average="macro"):.4f} LL {log_loss(y, oh):.4f}')
print('  sim per-class F1', dict(zip([l[:6] for l in L], np.round(f1_score(y, os_.argmax(1), average=None), 3))))
print('  test dist', dict(zip([l[:6] for l in L], np.bincount(tst.argmax(1), minlength=9).tolist())))
json.dump({'mode': mode, 'cnns': cnns, 'clean_f1': f1c, 'sim_f1': f1s}, open(f'oof/{name}.json', 'w'))
