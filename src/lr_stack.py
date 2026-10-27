# Second stacker family: multinomial logistic regression on log-probabilities of the CNNs (+ physics LGB members),
# trained on clean + PKS-sim rows like stack_mix; then a geometric blend with an LGB stacker, scored on clean / SIM / held-out.
# usage: python src/lr_stack.py <name> <cnn>[,<cnn>...] <lgb stacker to blend with>
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, log_loss
name, cnns, base = sys.argv[1], sys.argv[2].split(','), sys.argv[3]
tr = pd.read_csv('train_meta.csv'); L = sorted(tr.label.unique()); y = tr.label.map({l: i for i, l in enumerate(L)}).values
folds = np.load('oof/folds.npy')
lp = lambda p: np.log(np.clip(p, 1e-6, 1))
members = cnns                                                                     # physics information enters via the blend with the LGB stacker
def stack(kind):
    if kind == 'test':
        cols = [np.mean([np.load(f'oof/{c}__test_f{k}.npy') for k in range(5)], 0) if c != 'lgbA2_mix' else np.load('oof/lgbA2_mix__test.npy') for c in members]
    else:
        cols = [np.load(f'oof/{c}__{kind}.npy') for c in members]
    return np.hstack([lp(c) for c in cols])
C, S, H, T = stack('oof'), stack('oofsim'), stack('oofsimt'), stack('test')
oc, os_, oh, tst = np.zeros((len(y), 9)), np.zeros((len(y), 9)), np.zeros((len(y), 9)), np.zeros((len(T), 9))
for k in range(5):
    a, b = folds != k, folds == k
    sc = StandardScaler().fit(np.vstack([C[a], S[a]]))
    m = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(np.vstack([C[a], S[a]])), np.r_[y[a], y[a]])
    oc[b], os_[b], oh[b] = (m.predict_proba(sc.transform(X[b])) for X in (C, S, H)); tst += m.predict_proba(sc.transform(T)) / 5
for kind, arr in [('oof', oc), ('oofsim', os_), ('oofsimt', oh), ('test', tst)]: np.save(f'oof/{name}__{kind}.npy', arr)
def rep(tag, P): return f'{tag}: ' + ' | '.join(f'{k} F1 {f1_score(y, P[k].argmax(1), average="macro"):.4f} LL {log_loss(y, P[k]):.4f}' for k in ['oof', 'oofsim', 'oofsimt'])
B = {k: np.load(f'oof/{base}__{k}.npy') for k in ['oof', 'oofsim', 'oofsimt', 'test']}
R = {'oof': oc, 'oofsim': os_, 'oofsimt': oh, 'test': tst}
print(rep(base, B)); print(rep(name, R))
for w in [0.3, 0.5]:
    G = {k: np.exp((1 - w) * lp(B[k]) + w * lp(R[k])) for k in B}; G = {k: v / v.sum(1, keepdims=True) for k, v in G.items()}
    print(rep(f'blend w_lr={w}', G))
    if w == 0.3:
        for k, v in G.items(): np.save(f'oof/{name}_blend__{k}.npy', v)
