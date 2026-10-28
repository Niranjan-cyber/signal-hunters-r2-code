# Global-time CNN: panels placed at their real time rows (slew gaps empty) so an ETI track is one straight line.
# Input channels: whitened value, ON indicator, valid mask. Saves OOF/test probs per fold to oof/<name>__*.npy.
import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--name', default='cnn_v1'); ap.add_argument('--folds', default='0')
ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--width', type=int, default=32)
ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--bs', type=int, default=128)
ap.add_argument('--aug', type=int, default=1); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--full', type=int, default=0)      # 1 = train on all rows (test preds only)
ap.add_argument('--wv', default='')                 # whitened cache version ('' or '2' = deglitched)
ap.add_argument('--deep', type=int, default=0)      # 1 = extra stride-2 stage: receptive field spans whole cadence
ap.add_argument('--amp', default='bf16')            # fp16 on T4/P100 (no native bf16)
ap.add_argument('--pseudo', default='')             # .npy of test probabilities for pseudo-labels
ap.add_argument('--pthr', type=float, default=0.9)  # confidence threshold for pseudo-labels
ap.add_argument('--snr_p', type=float, default=0.0)  # prob. of SNR-matching noise (PKS whitened SNR = 0.74x GBT)
ap.add_argument('--snr_max', type=float, default=1.0) # max extra noise sd before renormalising (0.92 -> 0.74x SNR)
ap.add_argument('--lvch', type=int, default=0)      # 1 = 4th input channel: pre-whitening panel level (scintillation cue)
ap.add_argument('--syn', type=float, default=0.0)  # prob. per row of a synthetic ETI positive (same prob. for a hard negative)
ap.add_argument('--arch', default='resnet')         # 'resnet' (own) or 'timm:<name>' e.g. timm:efficientnet_b0 (ImageNet-pretrained)
ap.add_argument('--dann', type=float, default=0.0)  # >0: domain-adversarial weight (GBT train vs unlabelled PKS test, gradient reversal)
ap.add_argument('--mixup', type=float, default=0.0) # >0: MixUp Beta(alpha, alpha) on the placed input with soft targets (SETI 1st/2nd place)
ap.add_argument('--simonly', type=int, default=0)   # 1 = only score saved ckpts on PKS-simulated validation
ap.add_argument('--sim_s', type=float, default=0.92); ap.add_argument('--sim_seed', type=int, default=123)
ap.add_argument('--sim_tag', default='')            # '' = training-time simulator; 't' = held-out simulator (s 0.75, seed 7)
a = ap.parse_args()
ADT = torch.float16 if a.amp == 'fp16' else torch.bfloat16
scaler = torch.amp.GradScaler('cuda', enabled=a.amp == 'fp16')
torch.manual_seed(a.seed); np.random.seed(a.seed); dev = 'cuda'
H = 6 * 22

tr = pd.read_csv('train_meta.csv'); te = pd.read_csv('test_meta.csv')
LABELS = sorted(tr.label.unique()); y = torch.tensor(tr.label.map({l: i for i, l in enumerate(LABELS)}).values)
Wtr = torch.from_numpy(np.load(f'cache/train_w{a.wv}.npy')); Wte = torch.from_numpy(np.load(f'cache/test_w{a.wv}.npy'))
Mtr = torch.from_numpy(np.load('cache/train_miss.npy')); Mte = torch.from_numpy(np.load('cache/test_miss.npy'))
Gtr = torch.tensor(tr.slew_gap_rows.values); Gte = torch.tensor(te.slew_gap_rows.values)
if a.lvch:                                                                      # G becomes (n, 7): [gap, 6 panel levels]
    Gtr = torch.cat([Gtr[:, None].float(), torch.from_numpy(np.load('cache/train_lvz.npy'))], 1)
    Gte = torch.cat([Gte[:, None].float(), torch.from_numpy(np.load('cache/test_lvz.npy'))], 1)
folds = np.load('oof/folds.npy')
# pseudo-labels: confident test cadences join every fold's training set (validation stays train-only)
Wall, Mall, Gall, yall, pidx = Wtr, Mtr, Gtr, y, np.zeros(0, int)
if a.pseudo:
    P = np.load(a.pseudo); sel = np.where(P.max(1) >= a.pthr)[0]
    Wall = torch.cat([Wtr, Wte[sel]]); Mall = torch.cat([Mtr, Mte[sel]]); Gall = torch.cat([Gtr, Gte[sel]])
    yall = torch.cat([y, torch.from_numpy(P[sel].argmax(1))]); pidx = len(y) + np.arange(len(sel))
    print('pseudo rows', len(sel), flush=True)

def place(w, miss, gap):
    """(b,6,16,128) -> (b,3|4,H,128) at global time rows p*(16+gap)+r; 4th channel = panel level if gap carries levels."""
    z = None
    if gap.dim() == 2: z, gap = gap[:, 1:], gap[:, 0].long()
    if a.arch == 'onoff':                                                         # per-panel stack for OnOffNet: (b, 6*C, 16, 128)
        chans = [w, (1.0 - miss.float())[..., None].expand_as(w)]
        if z is not None: chans.append(z[:, :, None, None].float().expand_as(w) / 2)
        return torch.stack(chans, 2).reshape(w.shape[0], 6 * len(chans), 16, 128)
    b = w.shape[0]; out = torch.zeros(b, 4 if z is not None else 3, H, 128, device=w.device)
    r = torch.arange(16, device=w.device)
    for p in range(6):
        rows = (p * (16 + gap))[:, None] + r[None, :]                       # (b,16)
        idx = rows[:, :, None].expand(b, 16, 128)
        out[:, 0].scatter_(1, idx, w[:, p])
        out[:, 1].scatter_(1, idx, torch.full_like(w[:, p], 1.0 if p % 2 == 0 else -1.0))
        out[:, 2].scatter_(1, idx, 1.0 - miss[:, p, :, None].float().expand(b, 16, 128))
        if z is not None: out[:, 3].scatter_(1, idx, z[:, p, None, None].float().expand(b, 16, 128) / 2)
    return out

PRANK = torch.tensor([0, 1, 8, 2, 5, 6, 7, 4, 3])   # label priority rank by class index (eti, frb, noise, pulsar, radar, rfi_int, rfi_pers, sat, scint)

def inject(w, miss, gz, yb, p):
    """Synthetic ETI tracks (label eti) and hard negatives (disconnected / zero-drift / also in one OFF panel; label =
    higher priority of base label and rfi_intermittent). Lines follow chan = c0 + d*t, t = panel*(16+gap) + row."""
    b, dv = w.shape[0], w.device
    u = torch.rand(b, device=dv); pos = u < p; neg = (u >= p) & (u < 2 * p); sel = pos | neg
    if not sel.any(): return w, yb
    gap = (gz[:, 0] if gz.dim() == 2 else gz).float()
    t = torch.arange(6, device=dv)[None, :, None] * (16 + gap[:, None, None]) + torch.arange(16, device=dv)[None, None, :]
    d = torch.exp(np.log(0.05) + np.log(12.0) * torch.rand(b, device=dv)) * torch.where(torch.rand(b, device=dv) < 0.5, -1.0, 1.0)   # |d| log-uniform 0.05-0.6 (real ETI median ~0.18)
    kind = torch.randint(0, 3, (b,), device=dv)                                      # negatives: 0 disconnected, 1 zero drift, 2 also in OFF
    d = torch.where(neg & (kind == 1), torch.zeros_like(d), d)
    tmax = 4 * (16 + gap) + 15
    lo = torch.where(d > 0, torch.full_like(d, 6.0), 6.0 - d * tmax); hi = torch.where(d > 0, 121.0 - d * tmax, torch.full_like(d, 121.0))
    c0 = lo + (hi - lo).clamp(min=0) * torch.rand(b, device=dv)
    ch = c0[:, None, None] + d[:, None, None] * t                                      # (b,6,16)
    off = (5 + 25 * torch.rand(b, 6, device=dv)) * torch.where(torch.rand(b, 6, device=dv) < 0.5, -1.0, 1.0)
    ch = ch + torch.where((neg & (kind == 0))[:, None, None], off[:, :, None], torch.zeros_like(ch))
    panel = torch.zeros(b, 6, device=dv); panel[:, [0, 2, 4]] = 1.0
    extra = torch.zeros(b, 6, device=dv); extra[torch.arange(b, device=dv), 1 + 2 * torch.randint(0, 3, (b,), device=dv)] = 1.0
    panel = panel + torch.where((neg & (kind == 2))[:, None], extra, torch.zeros_like(extra))
    amp = (0.6 + 2.4 * torch.rand(b, device=dv))[:, None] * (0.75 + 0.5 * torch.rand(b, 6, device=dv)) * panel   # per-panel fading
    sig = 0.6 + 0.9 * torch.rand(b, device=dv)
    f = torch.arange(128, device=dv).float()
    line = amp[:, :, None, None] * torch.exp(-0.5 * ((f[None, None, None, :] - ch[..., None]) / sig[:, None, None, None]) ** 2)
    w = w + line * (~miss)[..., None] * sel[:, None, None, None]
    rk = PRANK.to(dv)
    neg_label = torch.where(rk[yb] < rk[5], yb, torch.full_like(yb, 5))              # keep base label if it outranks rfi_intermittent
    return w, torch.where(pos, torch.zeros_like(yb), torch.where(neg, neg_label, yb))

def augment(w, miss, gz=None):
    b = w.shape[0]; sn = None
    flip = torch.rand(b, device=w.device) < 0.5; w = torch.where(flip[:, None, None, None], w.flip(-1), w)
    sh = int(np.random.randint(-12, 13)); w = torch.roll(w, sh, -1)
    if sh > 0: w[..., :sh] = 0
    elif sh < 0: w[..., sh:] = 0
    w = w * (0.8 + 0.4 * torch.rand(b, 6, 1, 1, device=w.device))            # panel gain (in z units)
    if a.snr_p > 0:                                                           # SNR matching, renormalised to unit noise
        sn = (torch.rand(b, 1, 1, 1, device=w.device) < a.snr_p) * (0.4 + (a.snr_max - 0.4) * torch.rand(b, 1, 1, 1, device=w.device))
        w = (w + sn * torch.randn_like(w) * (~miss)[..., None]) / torch.sqrt(1 + sn ** 2)
    else:
        w = w + 0.35 * torch.rand(b, 1, 1, 1, device=w.device) * torch.randn_like(w)  # extra noise -> lower SNR like PKS
    g = torch.rand(b, 6, 16, 1, device=w.device) < 0.01                       # broadband glitch rows
    w = w + g * (1 + 2.5 * torch.rand(b, 6, 16, 1, device=w.device))
    d = torch.rand(b, 6, 16, device=w.device) < 0.003                         # dropped rows
    w = w * (~d)[..., None]; miss = miss | d
    if gz is not None and gz.dim() == 2:                                       # level channel: same SNR scaling + PKS-sized gain jitter
        gz = gz.clone(); fac = 1 / torch.sqrt(1 + sn[:, :, 0, 0] ** 2) if sn is not None else 1.0
        gz[:, 1:] = gz[:, 1:] * fac + 0.34 * torch.randn_like(gz[:, 1:]) * (torch.rand(b, 1, device=w.device) < 0.5)
    return w, miss, gz

def block(i, o, s):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.GELU(), nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o))
class Res(nn.Module):
    def __init__(s, i, o, st):
        super().__init__(); s.b = block(i, o, st); s.sk = nn.Identity() if (i == o and st == 1) else nn.Sequential(nn.Conv2d(i, o, 1, st, bias=False), nn.BatchNorm2d(o))
    def forward(s, x): return F.gelu(s.b(x) + s.sk(x))
class Net(nn.Module):
    def __init__(s, w, deep=0):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(4 if a.lvch else 3, w, (3, 5), 1, (1, 2), bias=False), nn.BatchNorm2d(w), nn.GELU())
        s.body = nn.Sequential(Res(w, w, 1), Res(w, 2 * w, 2), Res(2 * w, 2 * w, 1), Res(2 * w, 4 * w, 2), Res(4 * w, 4 * w, 1), Res(4 * w, 8 * w, 2), Res(8 * w, 8 * w, 1), *([Res(8 * w, 8 * w, 2), Res(8 * w, 8 * w, 1)] if deep else []))
        s.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(16 * w, 9))
        s.dom = nn.Sequential(nn.Linear(16 * w, 64), nn.GELU(), nn.Linear(64, 1)) if a.dann > 0 else None
    def feats(s, x):
        h = s.body(s.stem(x)); return torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
    def forward(s, x): return s.head(s.feats(x))

class TimmNet(nn.Module):
    """ImageNet-pretrained timm backbone on the global-time image (input upsampled 2x), mean+max pooled."""
    def __init__(s, arch):
        super().__init__(); import timm
        s.body = timm.create_model(arch, pretrained=True, in_chans=4 if a.lvch else 3, num_classes=0, global_pool='')
        nf = s.body.num_features
        s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * nf, 9))
        s.dom = nn.Sequential(nn.Linear(2 * nf, 64), nn.GELU(), nn.Linear(64, 1)) if a.dann > 0 else None
    def feats(s, x):
        h = s.body.forward_features(F.interpolate(x, scale_factor=2, mode='nearest')); return torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
    def forward(s, x): return s.head(s.feats(x))

class OnOffNet(nn.Module):
    """Explicit ON-vs-OFF comparison: a shared encoder per panel (time pooled, frequency kept), then per-frequency
    ON-mean / ON-min / OFF-mean / OFF-max / (ON-min - OFF-max) maps, a 1D CNN along frequency and the classifier."""
    def __init__(s, w):
        super().__init__(); cin = 3 if a.lvch else 2; s.cin = cin
        s.enc = nn.Sequential(nn.Conv2d(cin, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.GELU(),
                              Res(w, w, 1), Res(w, 2 * w, (2, 1)), Res(2 * w, 2 * w, 1), Res(2 * w, 2 * w, (2, 2)))
        f = 2 * w
        s.cmp = nn.Sequential(nn.Conv1d(5 * f, 4 * w, 5, 1, 2, bias=False), nn.BatchNorm1d(4 * w), nn.GELU(),
                              nn.Conv1d(4 * w, 4 * w, 5, 2, 2, bias=False), nn.BatchNorm1d(4 * w), nn.GELU())
        s.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(8 * w, 9)); s.dom = None
    def feats(s, x):
        b = x.shape[0]; p = x.view(b * 6, s.cin, 16, 128)
        h = s.enc(p).amax(2).view(b, 6, -1, 64)                                    # (b,6,F,64): time pooled, frequency kept
        on, off = h[:, [0, 2, 4]], h[:, [1, 3, 5]]
        z = torch.cat([on.mean(1), on.amin(1), off.mean(1), off.amax(1), on.amin(1) - off.amax(1)], 1)
        g = s.cmp(z); return torch.cat([g.mean(2), g.amax(2)], 1)
    def forward(s, x): return s.head(s.feats(x))

def make_model():
    if a.arch == 'onoff': return OnOffNet(a.width).to(dev)
    return (TimmNet(a.arch[5:]) if a.arch.startswith('timm:') else Net(a.width, a.deep)).to(dev).to(memory_format=torch.channels_last)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None

def predict(model, W, M, G, tta=True):
    model.eval(); out = []
    with torch.no_grad(), torch.autocast('cuda', dtype=ADT):
        for s in range(0, len(W), 512):
            w = W[s:s + 512].to(dev).float(); m = M[s:s + 512].to(dev).bool(); g = G[s:s + 512].to(dev)
            p = F.softmax(model(place(w, m, g)).float(), 1)
            if tta: p = (p + F.softmax(model(place(w.flip(-1), m, g)).float(), 1)) / 2
            out.append(p.cpu())
    return torch.cat(out).numpy()

def gsim(G, s=0.92):
    """Level channel under the simulator: same noise-scale increase divides the levels."""
    if G.dim() < 2: return G
    G = G.clone(); G[:, 1:] /= np.sqrt(1 + s * s); return G

def pks_sim(W, M, seed=123, s=0.92):
    """Fixed PKS-like copy of validation cadences: extra noise so whitened SNR drops to 1/sqrt(1+s^2) = 0.74x."""
    g = torch.Generator().manual_seed(seed); n = torch.randn(W.shape, generator=g)
    return ((W.float() + s * n * (1 - M.float())[..., None]) / np.sqrt(1 + s * s)).half()

os.makedirs('oof', exist_ok=True); os.makedirs('ckpt', exist_ok=True)
sim_path = f'oof/{a.name}__oofsim{a.sim_tag}.npy'
oofsim = np.load(sim_path) if os.path.exists(sim_path) else np.zeros((len(y), 9))
if a.simonly:
    from sklearn.metrics import f1_score, log_loss
    for k in [int(f) for f in a.folds.split(',')]:
        vai = np.where(folds == k)[0]; model = make_model()
        model.load_state_dict(torch.load(f'ckpt/{a.name}_f{k}.pt', map_location=dev))
        oofsim[vai] = predict(model, pks_sim(Wtr[vai], Mtr[vai], a.sim_seed, a.sim_s), Mtr[vai], gsim(Gtr[vai], a.sim_s)); np.save(sim_path, oofsim)
        print(f'fold {k} SIM F1 {f1_score(y[vai], oofsim[vai].argmax(1), average="macro"):.4f} LL {log_loss(y[vai], oofsim[vai], labels=range(9)):.4f}', flush=True)
    raise SystemExit
oof_path = f'oof/{a.name}__oof.npy'; tst_path = f'oof/{a.name}__test.npy'
oof = np.load(oof_path) if os.path.exists(oof_path) else np.zeros((len(y), 9))
tst_acc = {}
run_folds = [-1] if a.full else [int(f) for f in a.folds.split(',')]
for k in run_folds:
    t0 = time.time()
    tri = np.arange(len(y)) if k < 0 else np.where(folds != k)[0]; vai = None if k < 0 else np.where(folds == k)[0]
    tri = np.concatenate([tri, pidx])
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * (len(tri) // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.15)
    Wg = Wall.to(dev); Mg = Mall.to(dev).bool(); Gg = Gall.to(dev); yg = yall.to(dev)
    if a.dann > 0: WTg, MTg, GTg = Wte.to(dev), Mte.to(dev).bool(), Gte.to(dev)        # unlabelled PKS cadences for the domain head
    step, nsteps = 0, a.epochs * (len(tri) // a.bs)
    for ep in range(a.epochs):
        model.train(); perm = torch.from_numpy(np.random.permutation(tri)).to(dev); tl = 0
        for s in range(0, len(perm) - a.bs + 1, a.bs):
            ix = perm[s:s + a.bs]; w, m = Wg[ix].float(), Mg[ix]
            gg = Gg[ix]; yb = yg[ix]
            if a.syn > 0: w, yb = inject(w, m, gg, yb, a.syn)
            if a.aug: w, m, gg = augment(w, m, gg)
            with torch.autocast('cuda', dtype=ADT):
                if a.dann > 0:                                                    # DANN: GBT batch (labelled) + PKS batch (domain only)
                    it = torch.randint(0, len(WTg), (a.bs,), device=dev)
                    wt, mt, gt = WTg[it].float(), MTg[it], GTg[it]
                    if a.aug: wt, mt, gt = augment(wt, mt, gt)
                    fs, ft = model.feats(place(w, m, gg)), model.feats(place(wt, mt, gt))
                    lam = 2 / (1 + np.exp(-10 * step / nsteps)) - 1                 # Ganin et al. schedule 0 -> 1
                    dl = model.dom(GradReverse.apply(torch.cat([fs, ft]), lam)).float().squeeze(1)
                    dy = torch.cat([torch.zeros(len(fs), device=dev), torch.ones(len(ft), device=dev)])
                    loss = F.cross_entropy(model.head(fs).float(), yb, label_smoothing=0.05) + a.dann * F.binary_cross_entropy_with_logits(dl, dy)
                elif a.mixup > 0:                                                  # MixUp on the placed input, soft targets
                    x = place(w, m, gg); lam = float(np.random.beta(a.mixup, a.mixup)); pi = torch.randperm(len(x), device=dev)
                    logits = model(lam * x + (1 - lam) * x[pi]).float()
                    loss = lam * F.cross_entropy(logits, yb, label_smoothing=0.05) + (1 - lam) * F.cross_entropy(logits, yb[pi], label_smoothing=0.05)
                else:
                    loss = F.cross_entropy(model(place(w, m, gg)).float(), yb, label_smoothing=0.05)
            step += 1
            if not torch.isfinite(loss): raise SystemExit('NaN loss')
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tl += loss.item()
        msg = f'fold {k} ep {ep} loss {tl / (len(perm) // a.bs):.4f} {time.time() - t0:.0f}s'
        if vai is not None and (ep % 4 == 3 or ep == a.epochs - 1):
            from sklearn.metrics import f1_score, log_loss
            pv = predict(model, Wtr[vai], Mtr[vai], Gtr[vai], tta=False)
            msg += f' | val F1 {f1_score(y[vai], pv.argmax(1), average="macro"):.4f} LL {log_loss(y[vai], pv, labels=range(9)):.4f}'
        print(msg, flush=True)
    torch.save(model.state_dict(), f'ckpt/{a.name}_f{k}.pt')
    if vai is not None:
        oof[vai] = predict(model, Wtr[vai], Mtr[vai], Gtr[vai]); np.save(oof_path, oof)
        oofsim[vai] = predict(model, pks_sim(Wtr[vai], Mtr[vai]), Mtr[vai], gsim(Gtr[vai])); np.save(sim_path, oofsim)
        from sklearn.metrics import f1_score, log_loss
        print(f'fold {k} CLEAN F1 {f1_score(y[vai], oof[vai].argmax(1), average="macro"):.4f} LL {log_loss(y[vai], oof[vai], labels=range(9)):.4f} | '
              f'SIM F1 {f1_score(y[vai], oofsim[vai].argmax(1), average="macro"):.4f} LL {log_loss(y[vai], oofsim[vai], labels=range(9)):.4f}', flush=True)
    tst_acc[k] = predict(model, Wte, Mte, Gte); np.save(f'oof/{a.name}__test_f{k}.npy', tst_acc[k])
    print(f'fold {k} done {time.time() - t0:.0f}s; test pred dist', np.bincount(tst_acc[k].argmax(1), minlength=9).tolist(), flush=True)
json.dump(vars(a), open(f'oof/{a.name}.json', 'w'))
