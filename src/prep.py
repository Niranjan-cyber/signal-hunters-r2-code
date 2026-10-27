# Whitening: per-panel level, cadence bandpass smoothed across channels (keeps narrow lines), robust noise scale.
# Output float16 (n, 6, 16, 128) with 0 at missing cells, plus uint8 missing mask. Same code for both telescopes.
import numpy as np
from scipy.ndimage import median_filter

def whiten(X, batch=1000, bp_width=5, deglitch=False):
    n = len(X); out = np.zeros((n, 6, 16, 128), np.float16); miss = np.zeros((n, 6, 16), np.uint8)
    for s in range(0, n, batch):
        x = X[s:s + batch].astype(np.float32); m = x == 0
        x[m] = np.nan
        x[..., 64] = np.nanmean(x[..., [63, 65]], axis=-1)              # GBT centre spike -> interpolate (both sets)
        lvl = np.nanmedian(x, axis=(2, 3), keepdims=True); x = x - lvl  # panel gain/level
        spec = np.nanmedian(x, axis=2)                                   # (b,6,128) time-median spectrum per panel
        bp = np.nanmedian(spec, axis=1)                                  # (b,128) cadence bandpass
        bp = np.nan_to_num(bp); bp0 = median_filter(bp, size=(1, 17), mode='nearest')
        r = bp - bp0; sd = 1.4826 * np.median(np.abs(r), 1, keepdims=True) + 1e-3
        bp = np.where(np.abs(r) > 2.5 * sd, bp0, bp)                     # mask narrow lines, then follow the ripple
        bp = median_filter(bp, size=(1, bp_width), mode='nearest')
        x = x - bp[:, None, None, :]
        mad = np.nanmedian(np.abs(x - np.nanmedian(x, axis=(2, 3), keepdims=True)), axis=(2, 3), keepdims=True)
        x = x / (1.4826 * np.maximum(mad, 1.0))
        x = x - np.nanmedian(x, axis=(2, 3), keepdims=True)
        if deglitch:                                                     # flat bright rows = broadband glitches
            med = np.nanmedian(x, 3, keepdims=True); lo = np.nanmedian(x[..., :64], 3, keepdims=True); hi = np.nanmedian(x[..., 64:], 3, keepdims=True)
            x = np.where((med > 0.6) & (np.abs(lo - hi) < 0.5 * med), x - med, x)
        x = np.clip(np.nan_to_num(x, nan=0.0), -6, 30)
        out[s:s + batch] = x; miss[s:s + batch] = m.all(3)
    return out, miss

if __name__ == '__main__':
    import time
    import sys; v = sys.argv[1] if len(sys.argv) > 1 else ''
    for nm in ['train', 'test']:
        t = time.time(); X = np.load(f'{nm}_cadences.npy', mmap_mode='r')
        w, m = whiten(X, deglitch=(v == '2')); np.save(f'cache/{nm}_w{v}.npy', w); np.save(f'cache/{nm}_miss.npy', m)
        print(nm, w.shape, round(time.time() - t, 1), 's', 'sd', float(w[:2000].astype(np.float32).std()))
