import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
Xtr = np.load('train_cadences.npy', mmap_mode='r'); Xte = np.load('test_cadences.npy', mmap_mode='r')
tr = pd.read_csv('train_meta.csv'); LABELS = sorted(tr.label.unique()); rng = np.random.default_rng(0)
def show(rows, names, fn):
    fig, ax = plt.subplots(len(rows), 6, figsize=(13, 1.5 * len(rows)))
    for r, (X, i) in enumerate(rows):
        for p in range(6):
            ax[r, p].imshow(X[i, p], aspect='auto', cmap='magma', vmin=40, vmax=170); ax[r, p].set_xticks([]); ax[r, p].set_yticks([])
        ax[r, 0].set_ylabel(names[r], fontsize=7)
    plt.tight_layout(); plt.savefig(fn, dpi=70); plt.close()
for lab in LABELS:
    idx = rng.choice(np.where(tr.label == lab)[0], 6, replace=False)
    show([(Xtr, i) for i in idx], [f'{lab[:10]} {i}' for i in idx], f'figs/tr_{lab}.png')
idx = rng.choice(len(Xte), 12, replace=False); show([(Xte, i) for i in idx], [f'test {i}' for i in idx], 'figs/test_sample.png')
# instrument stats, train vs test (first 3000 each)
for nm, X in [('GBT', Xtr), ('PKS', Xte)]:
    x = X[:3000].astype(np.float32); drop = (x == 0).all(3)
    x[x == 0] = np.nan; bp = np.nanmedian(x, axis=2)          # (n,6,128) bandpass per panel
    lvl = np.nanmedian(x, axis=(2, 3))                         # panel level
    rowm = np.nanmedian(x, axis=3); glitch = (rowm - np.nanmedian(rowm, 2, keepdims=True)) > 6
    noise = np.nanmedian(np.nanstd(x - bp[:, :, None, :], axis=2), axis=(1, 2))
    print(nm, 'drop-row rate %.4f' % drop.mean(), 'glitch-row rate %.4f' % glitch.mean(), 'panel-level sd within cadence %.2f' % np.std(lvl, 1).mean(),
          'level %.1f' % np.nanmean(lvl), 'noise sd %.2f' % np.mean(noise), 'bp ripple sd %.2f' % np.nanmean(np.nanstd(bp - np.nanmedian(bp, 2, keepdims=True), 2)))
