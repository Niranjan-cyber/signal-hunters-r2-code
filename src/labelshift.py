# EM label-shift correction (Saerens et al. 2002) for the stacker, validated on held-out-simulation rows resampled
# to the test's estimated class mix. usage: python src/labelshift.py <stacker name>
import sys, json, numpy as np, pandas as pd
from sklearn.metrics import f1_score
name = sys.argv[1]
tr = pd.read_csv('train_meta.csv'); L = sorted(tr.label.unique()); y = tr.label.map({l: i for i, l in enumerate(L)}).values
oof, oofs, test = np.load(f'oof/{name}__oof.npy'), np.load(f'oof/{name}__oofsimt.npy'), np.load(f'oof/{name}__test.npy')
prior_tr = np.bincount(y, minlength=9) / len(y)

def em(p, prior_src, iters=100, damp=1.0):
    """Return estimated target priors and adjusted posteriors; damp<1 shrinks the weight towards 1."""
    pi = prior_src.copy()
    for _ in range(iters):
        w = (pi / prior_src) ** damp; q = p * w; q /= q.sum(1, keepdims=True)
        new = q.mean(0)
        if np.abs(new - pi).max() < 1e-7: break
        pi = new
    return pi, q

pi_test, _ = em(test, prior_tr)
print('train prior ', dict(zip([l[:6] for l in L], prior_tr.round(3))))
print('test EM prior', dict(zip([l[:6] for l in L], pi_test.round(3))))
print('test argmax  ', dict(zip([l[:6] for l in L], (np.bincount(test.argmax(1), minlength=9) / len(test)).round(3))))

# validation: resample held-out-sim rows (and clean rows) to the estimated test mix, 30 draws of 10k rows
rng = np.random.default_rng(0); res = {}
for src_name, P in [('heldout_sim', oofs), ('clean', oof)]:
    for damp in [0.0, 0.5, 1.0]:
        sc = []
        for r in range(30):
            idx = np.concatenate([rng.choice(np.where(y == c)[0], int(round(pi_test[c] * 10000)), replace=True) for c in range(9)])
            p, yy = P[idx], y[idx]
            if damp == 0: q = p
            else: _, q = em(p, prior_tr, damp=damp)
            sc.append(f1_score(yy, q.argmax(1), average='macro'))
        res[(src_name, damp)] = (np.mean(sc), np.std(sc))
        print(f'{src_name:12s} damp {damp:.1f}: macro-F1 {np.mean(sc):.4f} (sd over draws {np.std(sc):.4f})')
# paired gain per draw (same draws for damp 0 vs 1) on held-out sim
d = []
for r in range(30):
    idx = np.concatenate([rng.choice(np.where(y == c)[0], int(round(pi_test[c] * 10000)), replace=True) for c in range(9)])
    p, yy = oofs[idx], y[idx]; _, q = em(p, prior_tr)
    d.append(f1_score(yy, q.argmax(1), average='macro') - f1_score(yy, p.argmax(1), average='macro'))
print('held-out sim paired gain (full EM): %+.4f sd %.4f' % (np.mean(d), np.std(d)))
_, qt = em(test, prior_tr); np.save(f'oof/{name}_ls__test.npy', qt)
print('test argmax after EM', dict(zip([l[:6] for l in L], np.bincount(qt.argmax(1), minlength=9).tolist())), '| changed rows', int((qt.argmax(1) != test.argmax(1)).sum()))
json.dump({'pi_test': pi_test.tolist(), 'paired_gain': float(np.mean(d))}, open(f'oof/{name}_ls.json', 'w'))
