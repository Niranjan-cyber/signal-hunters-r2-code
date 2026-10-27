# Third-round build: fetch + merge n_cnxtt / n_effb2, then stacks on top of U6 (smx8a) incl. all pretrained families.
import subprocess, sys
def run(cmd):
    print('>>', cmd, flush=True); r = subprocess.run(cmd, shell=True)
    if r.returncode: sys.exit(f'FAILED: {cmd}')
for n in ['n_cnxtt', 'n_effb2']:
    run(f'python src/fetch_modal.py {n} 0,1,2,3,4 results'); run(f'python src/merge_modal.py {n} 0,1,2,3,4 results')
base = 'cnn_v1,cnn_v4p,cnn_v4a,cnn_v4ap,cnn_v6,cnn_v7,n_pretrained,n_cnxt'
for name, cnns in [('smx11_cnxtt', base + ',n_cnxtt'), ('smx11_effb2', base + ',n_effb2'),
                   ('smx11_all', base + ',s_regy,s_mbnv3,n_cnxtt,n_effb2')]:
    run(f'python src/stack_mix.py {name} mix {cnns} lv')
print('U8 BUILD DONE', flush=True)
