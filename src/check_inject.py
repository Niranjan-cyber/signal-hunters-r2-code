# Check synthetic ETI injection: plots + physics track scores of injected vs real ETI / real rfi_intermittent.
import re, sys, numpy as np, pandas as pd, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'src')
src = open('src/cnn.py', encoding='utf-8').read()
exec(src[src.index('PRANK = '):src.index('def augment(')])                        # PRANK + inject only
from phys import features
tr = pd.read_csv('train_meta.csv'); W = np.load('cache/train_w2.npy', mmap_mode='r'); M = np.load('cache/train_miss.npy', mmap_mode='r')
rng = np.random.default_rng(0); idx = rng.choice(np.where(tr.label == 'noise')[0], 400, replace=False)
w = torch.from_numpy(W[idx].astype(np.float32)).cuda(); m = torch.from_numpy(M[idx]).cuda().bool()
g = torch.tensor(tr.slew_gap_rows.values[idx]).cuda(); yb = torch.full((400,), 2, device='cuda')
torch.manual_seed(1); wi, yi = inject(w, m, g, yb, 0.5)
yi = yi.cpu().numpy(); diff = (wi - w).abs().amax((1, 2, 3)).cpu().numpy() > 0
kind = np.where(yi == 0, 'syn_eti', np.where(diff, 'syn_negative', 'untouched'))
F = features(wi.cpu().numpy().astype(np.float16), M[idx], tr.slew_gap_rows.values[idx])
T = pd.read_parquet('cache/train_phys2f.parquet')
cols = ['eti2_score', 'eti2_onmed', 'eti2_drift', 'conn_mis_mean', 'conn_rat_min', 'snr_off_max']
for k in ['syn_eti', 'syn_negative']: print(f'{k:14s} n={int((kind == k).sum()):3d}', F.loc[kind == k, cols].median().round(2).to_dict())
for lab in ['eti_candidate', 'rfi_intermittent']: print(f'real {lab[:9]:9s}', T.loc[tr.label == lab, cols].median().round(2).to_dict())
fig, ax = plt.subplots(1, 8, figsize=(18, 6)); show = np.r_[np.where(kind == 'syn_eti')[0][:4], np.where(kind == 'syn_negative')[0][:4]]
for a_, i in zip(ax, show):
    gp = int(g[i]); img = np.full((6 * (16 + gp), 128), np.nan)
    for p in range(6): img[p * (16 + gp):p * (16 + gp) + 16] = wi[i, p].cpu().numpy()
    a_.imshow(img, aspect='auto', cmap='magma', vmin=-2, vmax=5); a_.set_title(kind[i], fontsize=8)
plt.tight_layout(); plt.savefig('figs/inject_check.png', dpi=60); print('saved figs/inject_check.png')
