# Restart-proof Modal launcher: deploy the app, spawn one call per fold (runs in Modal's cloud, independent of this
# process), save call ids. Results land in the volume at /out/<name>/f<k>/ as before.
#   python src/modal_submit.py run    <name> "<cnn.py args without --name>" 0,1,2,3,4 [gpu: L4|A10G|A100|H100]
#   python src/modal_submit.py status <name>
#   python src/modal_submit.py cancel <name>      (stops that experiment's running calls; other experiments keep running)
import json, os, subprocess, sys, modal
os.makedirs('logs', exist_ok=True)
cmd, name = sys.argv[1], sys.argv[2]
ids_path = f'logs/spawn_{name}.json'
if cmd == 'run':
    args, folds = f'--name {name} ' + sys.argv[3], [int(f) for f in sys.argv[4].split(',')]
    gpu = sys.argv[5] if len(sys.argv) > 5 else 'L4'
    subprocess.run('modal deploy modal_run.py', shell=True, check=True)          # image embeds the current src/cnn.py
    fn = modal.Function.from_name('sh-r2', 'run_fold')
    if gpu != 'L4': fn = fn.with_options(gpu=gpu)
    calls = {str(k): fn.spawn(args, k).object_id for k in folds}
    json.dump({'args': args, 'gpu': gpu, 'calls': calls}, open(ids_path, 'w'), indent=1)
    print(f'spawned {name} folds {folds} on {gpu}: {calls}')
elif cmd == 'cancel':
    for k, cid in json.load(open(ids_path))['calls'].items():
        try: modal.FunctionCall.from_id(cid).cancel(); print(f'fold {k}: cancel requested')
        except Exception as e: print(f'fold {k}: {type(e).__name__}: {str(e)[:120]}')
else:
    info = json.load(open(ids_path))
    for k, cid in info['calls'].items():
        try:
            print(f'fold {k}: DONE  {modal.FunctionCall.from_id(cid).get(timeout=0).strip()}')
        except TimeoutError:
            print(f'fold {k}: running')
        except Exception as e:
            print(f'fold {k}: FAILED {type(e).__name__}: {str(e)[:200]}')
