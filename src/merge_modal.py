# Merge per-fold outputs fetched from Modal (modal_out/<name>/f<k>/oof|ckpt) into oof/ and ckpt/.
# usage: python src/merge_modal.py <name> <folds> [src dir: modal_out (default) | results]
import sys, os, shutil, numpy as np
name, fl = sys.argv[1], [int(f) for f in sys.argv[2].split(',')]
src = sys.argv[3] if len(sys.argv) > 3 else 'modal_out'
folds = np.load('oof/folds.npy')
for kind in ['oof', 'oofsim', 'oofsimt']:
    lp = f'oof/{name}__{kind}.npy'; lo = np.load(lp) if os.path.exists(lp) else np.zeros((len(folds), 9))
    got = []
    for k in fl:
        fp = f'{src}/{name}/f{k}/oof/{name}__{kind}.npy'
        if not os.path.exists(fp): continue
        ko = np.load(fp); m = folds == k
        if ko[m].sum(1).min() < 0.99: continue                                    # fold not present in this file (e.g. no held-out step)
        lo[m] = ko[m]; got.append(k)
    np.save(lp, lo); print(name, kind, 'merged', got, '| filled folds', sorted(set(folds[lo.sum(1) > 0.99])))
for k in fl:
    t = f'{src}/{name}/f{k}/oof/{name}__test_f{k}.npy'
    if os.path.exists(t): shutil.copy(t, 'oof/')
    c = f'{src}/{name}/f{k}/ckpt/{name}_f{k}.pt'
    if os.path.exists(c): os.makedirs('ckpt', exist_ok=True); shutil.copy(c, 'ckpt/')
