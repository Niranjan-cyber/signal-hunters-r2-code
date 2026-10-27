# Panel-level (pre-whitening) features: scintillation lifts whole ON panels; whitening removes panel medians.
import numpy as np, pandas as pd
def feats(X, batch=2000, gain_sd=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for s in range(0, len(X), batch):
        x = X[s:s + batch].astype(np.float32); x[x == 0] = np.nan
        x[..., 64] = np.nanmean(x[..., [63, 65]], axis=-1)
        lvl = np.nanmedian(x, axis=(2, 3))                                       # (b,6) panel level
        mad = 1.4826 * np.nanmedian(np.abs(x - lvl[:, :, None, None]), axis=(2, 3))  # per-panel noise (byte units)
        sig = np.median(mad, 1)                                                  # cadence noise scale
        z = (lvl - np.median(lvl, 1, keepdims=True)) / (sig[:, None] / np.sqrt(16 * 128) * 8)   # scaled level offsets
        if gain_sd > 0: z = z + gain_sd * rng.standard_normal(z.shape)          # stress test: extra PKS-like panel gain offsets
        on, off = z[:, [0, 2, 4]], z[:, [1, 3, 5]]
        alt = np.array([1, -1, 1, -1, 1, -1], np.float32)
        f = {'lv_alt': (z * alt).mean(1), 'lv_onmin_offmax': on.min(1) - off.max(1), 'lv_on_mean': on.mean(1), 'lv_off_mean': off.mean(1),
             'lv_alt_raw': ((lvl - lvl.mean(1, keepdims=True)) * alt).mean(1) / sig, 'lv_sd': z.std(1),
             'lv_nonalt': np.abs(np.diff(z[:, [0, 2, 4]], axis=1)).mean(1) + np.abs(np.diff(z[:, [1, 3, 5]], axis=1)).mean(1)}
        out.append(pd.DataFrame(f))
    return pd.concat(out, ignore_index=True)
def zlev(X, batch=2000):
    """Per-panel pre-whitening level offsets in noise-scaled units, (n, 6) float32 (same scaling as feats)."""
    out = []
    for s in range(0, len(X), batch):
        x = X[s:s + batch].astype(np.float32); x[x == 0] = np.nan
        x[..., 64] = np.nanmean(x[..., [63, 65]], axis=-1)
        lvl = np.nanmedian(x, axis=(2, 3)); mad = 1.4826 * np.nanmedian(np.abs(x - lvl[:, :, None, None]), axis=(2, 3))
        sig = np.median(mad, 1); out.append(((lvl - np.median(lvl, 1, keepdims=True)) / (sig[:, None] / np.sqrt(16 * 128) * 8)).astype(np.float32))
    return np.nan_to_num(np.concatenate(out))

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'z':
        for nm in ['train', 'test']:
            Z = zlev(np.load(f'{nm}_cadences.npy', mmap_mode='r')); np.save(f'cache/{nm}_lvz.npy', Z); print(nm, 'lvz', Z.shape, np.percentile(Z, [1, 50, 99]).round(2))
        sys.exit()
    for nm in ['train', 'test']:
        F = feats(np.load(f'{nm}_cadences.npy', mmap_mode='r')); F.to_parquet(f'cache/{nm}_lv.parquet'); print(nm, F.shape)
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'gain':
        F = feats(np.load('train_cadences.npy', mmap_mode='r'), gain_sd=0.34, seed=11); F.to_parquet('cache/train_lv_g.parquet'); print('train gain-stressed', F.shape)
