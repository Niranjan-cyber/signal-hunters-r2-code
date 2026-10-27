# Wide matched filter for ETI tracks: 3-channel boxcar across frequency (unit-noise scaled) before the global track search.
# usage: python src/phys_wide.py <whitened version: 2 | 2s>
import sys, time, numpy as np, pandas as pd, torch
sys.path.insert(0, 'src')
from phys import track_search, ON, OFF
dev = 'cuda'

def wide_feats(W, gaps, batch=250):
    tr_drifts = torch.linspace(-1.2, 1.2, 161, device=dev); c0 = torch.arange(-160, 288, device=dev)
    k = torch.ones(1, 1, 3, device=dev) / np.sqrt(3)
    rows = []
    for s in range(0, len(W), batch):
        w = torch.from_numpy(W[s:s + batch].astype(np.float32)).to(dev); g = torch.from_numpy(gaps[s:s + batch]).to(dev)
        b = w.shape[0]
        ws = torch.nn.functional.conv1d(w.reshape(-1, 1, 128), k, padding=1).reshape(b, 6, 16, 128)
        T = track_search(ws, g, tr_drifts, c0); on, off = T[..., ON], T[..., OFF]
        offmax = torch.nan_to_num(off, nan=-9).max(-1).values
        onmean, onmed, onmin = on.mean(-1), on.median(-1).values, on.min(-1).values
        sc = torch.nan_to_num(onmean - offmax.clamp(min=0), nan=-99).flatten(1); v, a = sc.max(1)
        f = {'w_eti_score': v, 'w_eti_onmed': onmed.flatten(1).gather(1, a[:, None])[:, 0], 'w_eti_onmin': onmin.flatten(1).gather(1, a[:, None])[:, 0],
             'w_eti_offmax': offmax.flatten(1).gather(1, a[:, None])[:, 0], 'w_eti_drift': tr_drifts[a // len(c0)].abs(),
             'w_eti3_score': torch.nan_to_num(onmed - offmax.clamp(min=0), nan=-99).flatten(1).max(1).values}
        rows.append(pd.DataFrame({k_: v_.float().cpu().numpy() for k_, v_ in f.items()}))
    return pd.concat(rows, ignore_index=True)

if __name__ == '__main__':
    v = sys.argv[1]
    for nm in ['train', 'test']:
        if nm == 'test' and v != '2': continue                              # test always uses the real (unsimulated) data
        t = time.time(); meta = pd.read_csv(f'{nm}_meta.csv')
        F = wide_feats(np.load(f'cache/{nm}_w{v}.npy', mmap_mode='r'), meta.slew_gap_rows.values)
        F.to_parquet(f'cache/{nm}_wide{v}.parquet'); print(nm, v, F.shape, round(time.time() - t), 's', flush=True)
