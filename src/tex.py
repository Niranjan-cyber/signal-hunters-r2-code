# Multi-scale ON-vs-OFF patchiness features (scintillation), numpy on whitened deglitched cadences.
import numpy as np, pandas as pd, sys, time
SCALES = [(2, 8), (4, 8), (4, 16), (8, 16), (8, 32), (16, 16), (16, 32), (16, 64)]
ON, OFF = [0, 2, 4], [1, 3, 5]
def feats(W, batch=2000):
    out = []
    for s in range(0, len(W), batch):
        w = W[s:s + batch].astype(np.float32); b = len(w); f = {}
        for (rt, rc) in SCALES:
            blk = w.reshape(b, 6, 16 // rt, rt, 128 // rc, rc).mean((3, 5)) * np.sqrt(rt * rc)   # z of block mean
            v = blk.reshape(b, 6, -1).var(2); mx = blk.reshape(b, 6, -1).max(2)
            n3 = (blk > 3).reshape(b, 6, -1).mean(2)
            k = f'{rt}x{rc}'
            f[f'tlr_{k}'] = np.log(v[:, ON].mean(1)) - np.log(v[:, OFF].mean(1)); f[f'tlrmin_{k}'] = np.log(v[:, ON].min(1)) - np.log(v[:, OFF].max(1))
            f[f'tv_on_{k}'] = v[:, ON].mean(1); f[f'tv_d_{k}'] = v[:, ON].mean(1) - v[:, OFF].mean(1)
            f[f'tv_onmin_d_{k}'] = v[:, ON].min(1) - v[:, OFF].max(1)
            f[f'tm_d_{k}'] = mx[:, ON].mean(1) - mx[:, OFF].mean(1); f[f'tn3_d_{k}'] = n3[:, ON].mean(1) - n3[:, OFF].mean(1)
        out.append(pd.DataFrame(f))
    return pd.concat(out, ignore_index=True)
if __name__ == '__main__':
    v = sys.argv[1] if len(sys.argv) > 1 else '2'
    for nm in ['train', 'test']:
        t = time.time(); F = feats(np.load(f'cache/{nm}_w{v}.npy', mmap_mode='r')); F.to_parquet(f'cache/{nm}_tex{"" if v == "2" else v}.parquet'); print(nm, F.shape, round(time.time() - t), 's')
