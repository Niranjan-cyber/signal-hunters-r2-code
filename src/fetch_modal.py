# Download one experiment's per-fold outputs from the Modal volume, file by file (folder downloads overwrite each other).
# usage: python src/fetch_modal.py <name> <folds e.g. 0,1,2,3,4> [dest: modal_out (default, includes ckpt) | results (small files only, for git)]
import os, subprocess, sys
name, folds = sys.argv[1], [int(f) for f in sys.argv[2].split(',')]
dest = sys.argv[3] if len(sys.argv) > 3 else 'modal_out'
env = dict(os.environ, MSYS_NO_PATHCONV='1', PYTHONIOENCODING='utf-8')
for k in folds:
    files = [f'{name}__oof.npy', f'{name}__oofsim.npy', f'{name}__oofsimt.npy', f'{name}__test_f{k}.npy', 'log.txt']
    if dest == 'modal_out': files.append(f'{name}_f{k}.pt')
    for f in files:
        sub = 'ckpt' if f.endswith('.pt') else 'oof'
        os.makedirs(f'{dest}/{name}/f{k}/{sub}', exist_ok=True)
        r = subprocess.run(['modal', 'volume', 'get', 'sh-r2', f'/out/{name}/f{k}/{f}', f'{dest}/{name}/f{k}/{sub}/{f}', '--force'],
                           capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
        print(f'f{k} {f}: {"ok" if r.returncode == 0 else "MISSING"}', flush=True)
