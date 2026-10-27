# AdaBN: re-estimate BatchNorm running stats on target-domain data at inference (no retraining).
# Validation: fold-k checkpoint, BN stats re-estimated on the held-out-simulated fold-k cadences (PKS-like), then
# scored on the same rows; compared with plain inference. Test: BN stats re-estimated on the real PKS test set.
# usage: python src/adabn.py <cnn name> <width> <lvch 0/1> [folds]
import sys, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import f1_score, log_loss
name, width, lvch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); folds_run = [int(f) for f in (sys.argv[4] if len(sys.argv) > 4 else '0,1,2,3,4').split(',')]
sys.argv = ['cnn.py', '--name', '_adabn_tmp', '--width', str(width), '--lvch', str(lvch), '--simonly', '1', '--folds', '0']
src = open('src/cnn.py', encoding='utf-8').read()
exec(src[:src.index("os.makedirs('oof', exist_ok=True)")])                          # data, Net, place, predict, pks_sim, gsim

def adapt(model, W, M, G, bs=256):
    """Reset BN running stats and re-estimate them (cumulative average) on W; weights untouched."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d): m.reset_running_stats(); m.momentum = None
    model.train()
    with torch.no_grad(), torch.autocast('cuda', dtype=ADT):
        for s in range(0, len(W), bs):
            model(place(W[s:s + bs].to(dev).float(), M[s:s + bs].to(dev).bool(), G[s:s + bs].to(dev)))
    model.eval(); return model

oo, oa = np.zeros((len(y), 9)), np.zeros((len(y), 9)); ta = []
for k in folds_run:
    vai = np.where(folds == k)[0]
    Ws, Gs = pks_sim(Wtr[vai], Mtr[vai], 7, 0.75), gsim(Gtr[vai], 0.75)
    model = Net(width).to(dev).to(memory_format=torch.channels_last); sd = torch.load(f'ckpt/{name}_f{k}.pt', map_location=dev)
    model.load_state_dict(sd); oo[vai] = predict(model, Ws, Mtr[vai], Gs)
    adapt(model, Ws, Mtr[vai], Gs); oa[vai] = predict(model, Ws, Mtr[vai], Gs)
    yk = y[vai].numpy()
    print(f'fold {k} held-out plain F1 {f1_score(yk, oo[vai].argmax(1), average="macro"):.4f} LL {log_loss(yk, oo[vai], labels=range(9)):.4f} | '
          f'AdaBN F1 {f1_score(yk, oa[vai].argmax(1), average="macro"):.4f} LL {log_loss(yk, oa[vai], labels=range(9)):.4f}', flush=True)
    model.load_state_dict(sd); adapt(model, Wte, Mte, Gte); ta.append(predict(model, Wte, Mte, Gte))
if len(folds_run) == 5:
    np.save(f'oof/{name}__oofsimt_adabn.npy', oa); np.save(f'oof/{name}__test_adabn.npy', np.mean(ta, 0))
    print('saved; test dist AdaBN', np.bincount(np.mean(ta, 0).argmax(1), minlength=9).tolist())
