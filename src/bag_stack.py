# Average seed-bagged stacker runs (from stack_mix.py with 'seed<N>' in extra) into one stack.
# usage: python src/bag_stack.py <out_name> <stack>[,<stack>...]
import sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score, log_loss
out, names = sys.argv[1], sys.argv[2].split(',')
tr = pd.read_csv('train_meta.csv'); L = sorted(tr.label.unique()); y = tr.label.map({l: i for i, l in enumerate(L)}).values
for part in ['oof', 'oofsim', 'oofsimt', 'test']:
    p = np.mean([np.load(f'oof/{n}__{part}.npy') for n in names], 0); np.save(f'oof/{out}__{part}.npy', p)
    if part != 'test': print(f'{out} {part:8s} F1 {f1_score(y, p.argmax(1), average="macro"):.4f} LL {log_loss(y, p):.4f}')
