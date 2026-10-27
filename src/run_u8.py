# Tonight's final stack test: U6 (smx7a) + ConvNeXt / ResNet34.
import subprocess, sys
base = 'cnn_v1,cnn_v4p,cnn_v4a,cnn_v4ap,cnn_v6,cnn_v7,n_pretrained'
for name, cnns in [('smx8a', base + ',n_cnxt'), ('smx8b', base + ',n_cnxt,n_res34'), ('smx8c', base + ',n_res34')]:
    print('>>', name, flush=True)
    r = subprocess.run(f'python src/stack_mix.py {name} mix {cnns} lv', shell=True)
    if r.returncode: sys.exit(f'FAILED: {name}')
print('U8 BUILD DONE', flush=True)
