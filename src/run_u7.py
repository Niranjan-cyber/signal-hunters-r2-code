# U7 build: stack variants adding the pretrained backbone and/or DANN to the U6 stack (smx6a).
import subprocess, sys
base = 'cnn_v1,cnn_v4p,cnn_v4a,cnn_v4ap,cnn_v6,cnn_v7'
for name, cnns in [('smx7a', base + ',n_pretrained'), ('smx7b', base + ',n_dann'), ('smx7c', base + ',n_pretrained,n_dann')]:
    print('>>', name, flush=True)
    r = subprocess.run(f'python src/stack_mix.py {name} mix {cnns} lv', shell=True)
    if r.returncode: sys.exit(f'FAILED: {name}')
print('U7 BUILD DONE', flush=True)
