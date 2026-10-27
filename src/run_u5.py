# U5 build: held-out-sim predictions for cnn_v6 folds 1-4, then stacker variants with v6.
import subprocess, sys
def run(cmd):
    print('>>', cmd, flush=True); r = subprocess.run(cmd, shell=True)
    if r.returncode: sys.exit(f'FAILED: {cmd}')
run('python src/cnn.py --name cnn_v6 --width 48 --lvch 1 --folds 1,2,3,4 --simonly 1 --sim_s 0.75 --sim_seed 7 --sim_tag t')
for name, cnns in [('smx5', 'cnn_v1,cnn_v4p,cnn_v4a,cnn_v4ap,cnn_v6'), ('smx5b', 'cnn_v1,cnn_v4ap,cnn_v6'), ('smx5c', 'cnn_v1,cnn_v4a,cnn_v4ap,cnn_v6')]:
    run(f'python src/stack_mix.py {name} mix {cnns} lv')
print('U5 BUILD DONE', flush=True)
