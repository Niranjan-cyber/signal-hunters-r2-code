# Merge a Kaggle lane's OOF/test files into local oof/: python src/merge_kaggle.py <outdir> <name> <folds e.g. 3,4>
import sys, shutil, glob, os, numpy as np
out, name, fl = sys.argv[1], sys.argv[2], [int(f) for f in sys.argv[3].split(',')]
folds = np.load('oof/folds.npy')
src = glob.glob(f'{out}/**/oof/{name}__oof.npy', recursive=True)[0].replace('\\', '/'); root = src.rsplit('/oof/', 1)[0]
ko = np.load(src); lp = f'oof/{name}__oof.npy'
lo = np.load(lp) if os.path.exists(lp) else np.zeros_like(ko)
for k in fl:
    m = folds == k; assert ko[m].sum(1).min() > 0.99, f'fold {k} missing in kaggle output'
    lo[m] = ko[m]; shutil.copy(f'{root}/oof/{name}__test_f{k}.npy', f'oof/{name}__test_f{k}.npy')
    for c in glob.glob(f'{root}/ckpt/{name}_f{k}.pt'): os.makedirs('ckpt', exist_ok=True); shutil.copy(c, 'ckpt/')
np.save(lp, lo); print(name, 'merged folds', fl, '| filled folds:', sorted(set(folds[lo.sum(1) > 0.99])))
