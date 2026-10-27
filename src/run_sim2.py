# Held-out simulator pipeline (s 0.75, seed 7): features on cache/train_w2t + CNN predictions on the same simulation.
import subprocess, sys, pandas as pd
def run(cmd):
    print('>>', cmd, flush=True); r = subprocess.run(cmd, shell=True)
    if r.returncode: sys.exit(f'FAILED: {cmd}')
run('python src/phys.py 2t e')
run('python src/tex.py 2t')
run('python src/phys_wide.py 2t')
P = pd.read_parquet('cache/train_phys2te.parquet'); T = pd.read_parquet('cache/train_tex2t.parquet')
rob = [c for c in T.columns if c.startswith(('tlr', 'tv_d_', 'tm_d_', 'tn3_d_'))]
pd.concat([P, T[rob]], axis=1).to_parquet('cache/train_phys2tf.parquet'); print('built phys2tf', flush=True)
for name, width in [('cnn_v1', 32), ('cnn_v4p', 48), ('cnn_v4a', 48)]:
    run(f'python src/cnn.py --name {name} --width {width} --folds 0,1,2,3,4 --simonly 1 --sim_s 0.75 --sim_seed 7 --sim_tag t')
print('SIM2 PIPELINE DONE', flush=True)
