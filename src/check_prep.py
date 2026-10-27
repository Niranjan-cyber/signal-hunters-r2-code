import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score; import lightgbm as lgb
X = np.load('train_cadences.npy', mmap_mode='r'); W = np.load('cache/train_w.npy', mmap_mode='r'); We = np.load('cache/test_w.npy', mmap_mode='r')
tr = pd.read_csv('train_meta.csv'); rng = np.random.default_rng(1)
picks = [rng.choice(np.where(tr.label == l)[0]) for l in ['rfi_persistent', 'rfi_intermittent', 'eti_candidate', 'pulsar', 'scintillating_star', 'frb']]
fig, ax = plt.subplots(12, 6, figsize=(13, 18))
for k, i in enumerate(picks):
    for p in range(6):
        ax[2*k, p].imshow(X[i, p], aspect='auto', cmap='magma', vmin=40, vmax=170); ax[2*k+1, p].imshow(W[i, p].astype(float), aspect='auto', cmap='magma', vmin=-3, vmax=8)
        for a in (ax[2*k, p], ax[2*k+1, p]): a.set_xticks([]); a.set_yticks([])
    ax[2*k, 0].set_ylabel(tr.label[i][:12] + ' raw', fontsize=7); ax[2*k+1, 0].set_ylabel('whitened', fontsize=7)
plt.tight_layout(); plt.savefig('figs/prep_check.png', dpi=60)
# line SNR preserved? best time-mean column z per panel, raw (vs local channel median) vs whitened, for persistent RFI
idx = np.where(tr.label == 'rfi_persistent')[0][:500]
xr = X[idx].astype(np.float32); colr = xr.mean(2); colr = colr - np.median(colr, 2, keepdims=True); rs = colr.max(2) / (xr.std((2, 3)) / 4)
cw = W[idx].astype(np.float32).mean(2); ws = cw.max(2) / (1 / 4)
print('persistent RFI median best-line SNR raw %.1f whitened %.1f' % (np.median(rs), np.median(ws)))
# adversarial AUC on simple whitened stats
def st(A):
    a = A.astype(np.float32); return np.concatenate([a.std((2, 3)), np.percentile(a, [1, 50, 99], axis=(2, 3)).transpose(1, 0, 2).reshape(len(a), -1),
        a.mean(2).std(2), a.mean(3).std(2), np.abs(a.mean(2)).max(2)], 1)
S = np.concatenate([st(W[:5000]), st(We[:5000])]); yy = np.r_[np.zeros(5000), np.ones(5000)]
print('adversarial AUC whitened stats', cross_val_score(lgb.LGBMClassifier(n_estimators=200, verbose=-1), S, yy, cv=3, scoring='roc_auc').mean().round(3))
Xe = np.load('test_cadences.npy', mmap_mode='r')
def rst(A):
    a = A.astype(np.float32); return np.concatenate([a.std((2, 3)), np.percentile(a, [1, 50, 99], axis=(2, 3)).transpose(1, 0, 2).reshape(len(a), -1), a.mean(2).std(2), a.mean(3).std(2)], 1)
S = np.concatenate([rst(X[:5000]), rst(Xe[:5000])])
print('adversarial AUC raw stats', cross_val_score(lgb.LGBMClassifier(n_estimators=200, verbose=-1), S, yy, cv=3, scoring='roc_auc').mean().round(3))
m = lgb.LGBMClassifier(n_estimators=200, verbose=-1).fit(np.concatenate([st(W[:5000]), st(We[:5000])]), yy)
names = [f'std{p}' for p in range(6)] + [f'q{q}_{p}' for q in (1, 50, 99) for p in range(6)] + [f'colstd{p}' for p in range(6)] + [f'rowstd{p}' for p in range(6)] + [f'colmax{p}' for p in range(6)]
print(sorted(zip(m.feature_importances_, names))[-8:])
A = W[:5000].astype(np.float32); B = We[:5000].astype(np.float32)
for nm, f in [('std', lambda a: a.std((2, 3))), ('q99', lambda a: np.percentile(a, 99, axis=(2, 3))), ('q1', lambda a: np.percentile(a, 1, axis=(2, 3))), ('colstd', lambda a: a.mean(2).std(2)), ('rowstd', lambda a: a.mean(3).std(2))]:
    print(nm, 'GBT %.3f PKS %.3f' % (np.median(f(A)), np.median(f(B))))
