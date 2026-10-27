# Kaggle lane: prep + CNN folds. Edit ARGS before push.
import os, subprocess, shutil, glob, hashlib, numpy as np, pandas as pd, time
ARGS = os.environ.get('ARGS', '--name cnn_v4p --folds 4 --epochs 30 --width 48 --amp fp16 --pseudo oof/pseudo_stk_v3.npy')
HARD_STOP = time.time() + 8.5 * 3600
src = glob.glob('/kaggle/input/**/train_cadences.npy', recursive=True)[0].rsplit('/', 1)[0]
os.chdir('/kaggle/working')
for f in ['train_cadences.npy', 'test_cadences.npy', 'train_meta.csv', 'test_meta.csv', 'prep.py', 'cnn.py']:
    code = glob.glob(f'/kaggle/input/**/sh-r2-code/{f}', recursive=True) or glob.glob(f'/kaggle/input/**/{f}', recursive=True)
    if f.endswith('.py'): shutil.copy(code[0], f); continue
    if not os.path.exists(f): os.symlink(f'{src}/{f}', f) if f.endswith('.npy') else shutil.copy(f'{src}/{f}', f)
os.makedirs('cache', exist_ok=True); os.makedirs('oof', exist_ok=True); os.makedirs('src', exist_ok=True)
from sklearn.model_selection import StratifiedKFold
tr = pd.read_csv('train_meta.csv'); y = tr.label.map({l: i for i, l in enumerate(sorted(tr.label.unique()))}).values
folds = np.zeros(len(y), int)
for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)): folds[va] = k
h = hashlib.md5(folds.astype(np.int64).tobytes()).hexdigest(); print('folds hash', h, flush=True)
assert h == 'b51c718af2905ae51db487e9e65f60b9', 'fold mismatch'
np.save('oof/folds.npy', folds)
for p in glob.glob('/kaggle/input/**/sh-r2-code/pseudo_*.npy', recursive=True): shutil.copy(p, 'oof/')
wv = '2' if '--wv 2' in ARGS else ''
subprocess.run(f'python prep.py {wv}', shell=True, check=True)
subprocess.run(f'python cnn.py {ARGS}', shell=True, check=True)
for f in glob.glob('cache/*'): os.remove(f)
print('done', flush=True)
