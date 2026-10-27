# U6 build: fetch + merge cnn_v7 (all folds), fill missing held-out predictions locally, then stack variants vs smx5.
# usage: python src/run_u6.py [extra cnn already merged, e.g. s_v6w64]
import subprocess, sys, numpy as np
extra = sys.argv[1] if len(sys.argv) > 1 else ''
def run(cmd):
    print('>>', cmd, flush=True); r = subprocess.run(cmd, shell=True)
    if r.returncode: sys.exit(f'FAILED: {cmd}')
run('python src/fetch_modal.py cnn_v7 0,1,2,3,4 modal_out')
run('python src/merge_modal.py cnn_v7 0,1,2,3,4')
folds = np.load('oof/folds.npy'); h = np.load('oof/cnn_v7__oofsimt.npy')
missing = [k for k in range(5) if h[folds == k].sum(1).min() < 0.99]
if missing:
    run(f'python src/cnn.py --name cnn_v7 --width 48 --lvch 1 --folds {",".join(map(str, missing))} --simonly 1 --sim_s 0.75 --sim_seed 7 --sim_tag t')
base = 'cnn_v1,cnn_v4p,cnn_v4a,cnn_v4ap,cnn_v6'
variants = [('smx6a', base + ',cnn_v7')]
if extra: variants += [('smx6b', base + f',{extra}'), ('smx6c', base + f',cnn_v7,{extra}')]
for name, cnns in variants:
    run(f'python src/stack_mix.py {name} mix {cnns} lv')
print('U6 BUILD DONE', flush=True)
