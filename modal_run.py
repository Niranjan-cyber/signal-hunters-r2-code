# Modal lane: one CNN fold per GPU container. usage:
#   modal run modal_run.py --args "--name cnn_v4a --epochs 30 --width 48 --snr_p 0.7 --snr_max 1.0" --folds 0,1,2,3,4
import os, modal

TORCH = os.environ.get("SH_TORCH", "2.5.1")                      # SH_TORCH=2.8.0 (CUDA 12.8 wheels) for B200; 2.5.1 has no Blackwell kernels
app = modal.App("sh-r2" if TORCH == "2.5.1" else "sh-r2-b200")
vol = modal.Volume.from_name("sh-r2")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install(f"torch=={TORCH}", "numpy==2.1.3", "pandas==2.2.3", "scipy==1.14.1", "scikit-learn==1.5.2", "timm==1.0.11", "huggingface_hub==0.26.2", "safetensors==0.4.5")
         .run_commands("python -c \"import timm; [timm.create_model(m, pretrained=True) for m in ['efficientnet_b0', 'efficientnet_b2', 'convnext_nano', 'convnext_tiny', 'tf_efficientnetv2_s', 'tf_efficientnetv2_m', 'tf_efficientnetv2_l', 'eca_nfnet_l0']]\"")  # weights baked in: 10 parallel HF downloads hit 429
         .add_local_file("src/cnn.py", "/code/cnn.py").add_local_file("src/prep.py", "/code/prep.py"))

@app.function(image=image, gpu="L4", volumes={"/vol": vol}, timeout=3 * 3600, memory=16384)
def run_fold(args: str, fold: int):
    import os, shutil, subprocess, glob
    import tempfile; w = tempfile.mkdtemp(prefix=f"w{fold}_"); os.makedirs(f"{w}/oof", exist_ok=True); os.makedirs(f"{w}/cache", exist_ok=True); os.chdir(w)
    for f in ["train_cadences.npy", "test_cadences.npy", "train_meta.csv", "test_meta.csv"]: os.symlink(f"/vol/data/{f}", f)
    shutil.copy("/vol/data/folds.npy", "oof/folds.npy"); [shutil.copy(p, "oof/") for p in glob.glob("/vol/data/pseudo_*.npy")]; [shutil.copy(p, "cache/") for p in glob.glob("/vol/data/*_lvz.npy")]
    shutil.copy("/code/cnn.py", "cnn.py"); shutil.copy("/code/prep.py", "prep.py")
    wv = "2" if "--wv 2" in args else ""
    subprocess.run(f"python prep.py {wv}", shell=True, check=True)
    r = subprocess.run(f"python cnn.py {args} --folds {fold}", shell=True, capture_output=True, text=True)
    log = r.stdout + "\n" + r.stderr[-3000:]
    if r.returncode == 0:                                          # held-out simulator predictions (s 0.75, seed 7): no ckpt download needed
        h = subprocess.run(f"python cnn.py {args} --folds {fold} --simonly 1 --sim_s 0.75 --sim_seed 7 --sim_tag t", shell=True, capture_output=True, text=True)
        log += "\n[held-out] " + "\n".join(l for l in h.stdout.splitlines() if "SIM" in l) + h.stderr[-1500:]
    name = args.split("--name")[1].split()[0]; out = f"/vol/out/{name}/f{fold}"; os.makedirs(out, exist_ok=True)
    for p in glob.glob("oof/*") + glob.glob("ckpt/*"):
        if "folds" not in p and "pseudo" not in p: shutil.copy(p, out)
    open(f"{out}/log.txt", "w").write(log); vol.commit()
    keep = [l for l in r.stdout.splitlines() if "CLEAN" in l or l.startswith(f"fold {fold} ep 1 ")]
    keep += [l for l in log.splitlines() if "[held-out]" in l or ("SIM F1" in l and "CLEAN" not in l)]
    return f"fold {fold} rc={r.returncode} | " + " | ".join(keep)

@app.local_entrypoint()
def main(args: str, folds: str = "0"):
    for res in run_fold.starmap([(args, int(f)) for f in folds.split(",")]):
        print(res, flush=True)
