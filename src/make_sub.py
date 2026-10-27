# Write a validated submission from oof/<name>__test.npy (optionally with __w.json multipliers).
# usage: python src/make_sub.py <name> <tag> [use_w 0/1]
import sys, json, hashlib, numpy as np, pandas as pd
name, tag = sys.argv[1], sys.argv[2]; use_w = len(sys.argv) > 3 and sys.argv[3] == '1'
tr = pd.read_csv('train_meta.csv'); te = pd.read_csv('test_meta.csv'); ss = pd.read_csv('sample_submission.csv')
L = sorted(tr.label.unique()); p = np.load(f'oof/{name}__test.npy')
if use_w: p = p * np.array(json.load(open(f'oof/{name}__w.json'))['w'])
sub = pd.DataFrame({'id': te.id, 'label': np.array(L)[p.argmax(1)]})
assert len(sub) == 10000 and sub.id.is_unique and set(sub.id) == set(ss.id) and sub.label.isin(L).all()
sub = sub.set_index('id').loc[ss.id].reset_index()
fn = f'subs/{tag}.csv'; sub.to_csv(fn, index=False)
print(fn, hashlib.md5(open(fn, 'rb').read()).hexdigest()[:10], sub.label.value_counts().to_dict())
