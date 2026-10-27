# U7 build: stack Swarali's pretrained families (s_regy, s_mbnv3) on top of U6 (smx8a). Merges results/ first.
import os, subprocess, sys
def run(cmd):
    print('>>', cmd, flush=True); r = subprocess.run(cmd, shell=True)
    if r.returncode: sys.exit(f'FAILED: {cmd}')
fams = [f for f in ['s_regy', 's_mbnv3'] if os.path.isdir(f'results/{f}')]
for f in fams: run(f'python src/merge_modal.py {f} 0,1,2,3,4 results')
base = 'cnn_v1,cnn_v4p,cnn_v4a,cnn_v4ap,cnn_v6,cnn_v7,n_pretrained,n_cnxt'
variants = [(f'smx10_{f}', base + f',{f}') for f in fams]
if len(fams) == 2: variants.append(('smx10_both', base + ',s_regy,s_mbnv3'))
for name, cnns in variants: run(f'python src/stack_mix.py {name} mix {cnns} lv')
print('U7 BUILD DONE', flush=True)
