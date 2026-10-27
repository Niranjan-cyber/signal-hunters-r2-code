# Physics features on whitened cadences: per-panel drift search, global ON-track search, broadband/texture stats.
import numpy as np, pandas as pd, torch, time
dev = 'cuda'
ON, OFF = [0, 2, 4], [1, 3, 5]

def panel_dedoppler(w, drifts):
    """w (b,6,16,128) -> line SNR (b,6,D,128): sum along chan = c + d*r over 16 rows / 4."""
    b = w.shape[0]; r = torch.arange(16, device=dev).float(); out = []
    pad = torch.nn.functional.pad(w, (128, 128))                          # out-of-band = 0
    for d in drifts:
        idx = (torch.arange(128, device=dev)[None, :] + torch.round(d * r)[:, None]).long() + 128   # (16,128)
        out.append(torch.gather(pad, 3, idx.expand(b, 6, 16, 128)).sum(2) / 4)
    return torch.stack(out, 2)

def track_search(w, gap, drifts, c0):
    """Global-time straight tracks chan = c + d*t, t = p*(16+gap)+r. Returns per-panel SNR along track (b,D,C,6)."""
    b = w.shape[0]; pad = torch.nn.functional.pad(w, (256, 256)); res = []
    r = torch.arange(16, device=dev).float()
    for d in drifts:
        per = []
        for p in range(6):
            t = (p * (16 + gap))[:, None].float() + r[None, :]                         # (b,16)
            ch = (c0[None, None, :] + torch.round(d * t)[:, :, None]).long() + 256      # (b,16,C)
            ok = (ch >= 256) & (ch < 384)
            ch = ch.clamp(0, 128 + 511)
            v = torch.gather(pad[:, p], 2, ch)                                          # (b,16,C)
            n = ok.sum(1).clamp(min=1)
            per.append(torch.where(ok.sum(1) >= 8, (v * ok).sum(1) / n.sqrt(), torch.full_like(n, float('nan'), dtype=torch.float)))
        res.append(torch.stack(per, -1))                                                # (b,C,6)
    return torch.stack(res, 1)

def features(W, M, gaps, batch=250):
    pd_drifts = torch.linspace(-8, 8, 129, device=dev)
    tr_drifts = torch.linspace(-1.2, 1.2, 161, device=dev); c0 = torch.arange(-160, 288, device=dev)
    fine_drifts = torch.linspace(-1.5, 1.5, 151, device=dev)
    rows = []
    for s in range(0, len(W), batch):
        w = torch.from_numpy(W[s:s + batch].astype(np.float32)).to(dev); g = torch.from_numpy(gaps[s:s + batch]).to(dev)
        b = w.shape[0]; f = {}
        # --- per-panel narrowband
        S = panel_dedoppler(w, pd_drifts)                                   # (b,6,D,128)
        best, arg = S.flatten(2).max(2); dbest = pd_drifts[arg // 128]
        z0 = S[:, :, 64].amax(2)                                            # zero drift (index 64)
        nline = (S.amax(2) > 6).sum(2).float()
        for p in range(6):
            f[f'snr{p}'] = best[:, p]; f[f'drift{p}'] = dbest[:, p].abs(); f[f'snr0_{p}'] = z0[:, p]; f[f'nline{p}'] = nline[:, p]
        f['snr_on_min'] = best[:, ON].min(1).values; f['snr_off_max'] = best[:, OFF].max(1).values
        f['snr_on_mean'] = best[:, ON].mean(1); f['snr_off_mean'] = best[:, OFF].mean(1)
        f['n_panels_line'] = (best > 7).sum(1).float()
        f['drift_on_sd'] = dbest[:, ON].std(1); f['drift_on_absmean'] = dbest[:, ON].abs().mean(1)
        # zero-drift persistent line: same channel in all panels
        Z = S[:, :, 64]                                                     # (b,6,128)
        f['persist_min'] = Z.min(1).values.amax(1); f['persist_on_only'] = (Z[:, ON].min(1).values - Z[:, OFF].max(1).values).amax(1)
        # --- global ON track (ETI)
        T = track_search(w, g, tr_drifts, c0)                               # (b,D,C,6)
        on = T[..., ON]; off = T[..., OFF]
        onmin = on.min(-1).values; offmax = torch.nan_to_num(off, nan=-9).max(-1).values
        sc = torch.nan_to_num(onmin - offmax.clamp(min=0), nan=-99)
        sc_flat = sc.flatten(1); v, a = sc_flat.max(1)
        dI = a // len(c0); cI = a % len(c0)
        f['eti_score'] = v; f['eti_drift'] = tr_drifts[dI].abs()
        f['eti_onmin'] = onmin.flatten(1).gather(1, a[:, None])[:, 0]; f['eti_offmax'] = offmax.flatten(1).gather(1, a[:, None])[:, 0]
        f['eti_onmin_any'] = torch.nan_to_num(onmin, nan=-99).flatten(1).max(1).values
        # best track excluding near-zero drift (non-drifting trap)
        mask = (tr_drifts.abs() > 0.03)[None, :, None]
        f['eti_score_drift'] = torch.where(mask, sc, torch.full_like(sc, -99)).flatten(1).max(1).values
        # evidence summed over ON panels (one faint ON panel no longer kills the score)
        onmean = on.mean(-1); onmed = on.median(-1).values
        sc2 = torch.nan_to_num(onmean - offmax.clamp(min=0), nan=-99).flatten(1); v2, a2 = sc2.max(1)
        f['eti2_score'] = v2; f['eti2_drift'] = tr_drifts[a2 // len(c0)].abs()
        f['eti2_onmin'] = onmin.flatten(1).gather(1, a2[:, None])[:, 0]; f['eti2_onmed'] = onmed.flatten(1).gather(1, a2[:, None])[:, 0]
        f['eti2_offmax'] = offmax.flatten(1).gather(1, a2[:, None])[:, 0]
        f['eti3_score'] = torch.nan_to_num(onmed - offmax.clamp(min=0), nan=-99).flatten(1).max(1).values
        f['eti2_score_drift'] = torch.where(mask, sc2.view_as(sc), torch.full_like(sc, -99)).flatten(1).max(1).values
        # connection test: a real track has global slope == local drift inside every ON panel
        Sf = panel_dedoppler(w, fine_drifts)                                # (b,6,Df,128)
        dg = tr_drifts[a2 // len(c0)]; cg = c0[a2 % len(c0)]
        iat = (fine_drifts[None, :] - dg[:, None]).abs().argmin(1)          # fine index of the global slope
        mis, rat, pos = [], [], []
        for p in ON:
            ch = (cg + torch.round(dg * (p * (16 + g)).float())).long()     # track channel at this panel's row 0
            loc = torch.stack([Sf[torch.arange(b), p, :, (ch + o).clamp(0, 127)] for o in (-1, 0, 1)], -1).amax(-1)   # (b,Df)
            sl, il = loc.max(1); sat = loc.gather(1, iat[:, None])[:, 0]
            inb = (ch >= 0) & (ch < 128)
            mis.append(torch.where(inb, (fine_drifts[il] - dg).abs(), torch.full_like(dg, 2.0)))
            rat.append(torch.where(inb, sat / sl.clamp(min=1e-3), torch.zeros_like(dg)))
            pos.append(sl)
        mis = torch.stack(mis, 1); rat = torch.stack(rat, 1); pos = torch.stack(pos, 1)
        f['conn_mis_mean'] = mis.mean(1); f['conn_mis_max'] = mis.max(1).values
        f['conn_rat_min'] = rat.min(1).values; f['conn_rat_mean'] = rat.mean(1); f['conn_loc_min'] = pos.min(1).values
        # --- broadband
        rowm = w.mean(3)                                                    # (b,6,16)
        f['bb_on_max'] = rowm[:, ON].amax((1, 2)); f['bb_off_max'] = rowm[:, OFF].amax((1, 2))
        f['bb_on_std'] = rowm[:, ON].std(2).mean(1); f['bb_off_std'] = rowm[:, OFF].std(2).mean(1)
        lo, hi = w[..., :64].mean(3), w[..., 64:].mean(3)                   # dispersion: low band late
        f['disp_on'] = (lo[:, ON, 1:] * hi[:, ON, :-1]).mean((1, 2)); f['disp_off'] = (lo[:, OFF, 1:] * hi[:, OFF, :-1]).mean((1, 2))
        # rows per panel above threshold (pulse count) and its ON/OFF contrast
        f['bb_on_n'] = (rowm[:, ON] > 0.5).float().sum((1, 2)); f['bb_off_n'] = (rowm[:, OFF] > 0.5).float().sum((1, 2))
        f['bb_panels'] = (rowm.amax(2) > 0.8).float().sum(1)
        # --- texture (scintillation): variance of 4x16 block means
        blk = w.reshape(b, 6, 4, 4, 8, 16).mean((3, 5))
        bv = blk.flatten(2).var(2)
        f['tex_on'] = bv[:, ON].mean(1); f['tex_off'] = bv[:, OFF].mean(1); f['tex_ratio'] = f['tex_on'] - f['tex_off']
        f['tex_on_min'] = bv[:, ON].min(1).values
        # --- sweeps (satellite/radar): steep drift lines
        steep = S[:, :, (pd_drifts.abs() > 2.5)].amax((2, 3))
        f['steep_max'] = steep.max(1).values; f['steep_npan'] = (steep > 7).float().sum(1)
        f['steep_on'] = steep[:, ON].mean(1); f['steep_off'] = steep[:, OFF].mean(1)
        f['miss_frac'] = torch.from_numpy(M[s:s + batch].astype(np.float32)).to(dev).mean((1, 2))
        rows.append(pd.DataFrame({k: v.float().cpu().numpy() for k, v in f.items()}))
    return pd.concat(rows, ignore_index=True)

if __name__ == '__main__':
    import sys; v = sys.argv[1] if len(sys.argv) > 1 else ''
    for nm in ['train', 'test']:
        t = time.time(); meta = pd.read_csv(f'{nm}_meta.csv')
        Fe = features(np.load(f'cache/{nm}_w{v}.npy', mmap_mode='r'), np.load(f'cache/{nm}_miss.npy', mmap_mode='r'), meta.slew_gap_rows.values)
        Fe.to_parquet(f'cache/{nm}_phys{v}{sys.argv[2] if len(sys.argv) > 2 else ""}.parquet'); print(nm, Fe.shape, round(time.time() - t), 's', flush=True)
