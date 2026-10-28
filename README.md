# Signal Hunters: Hack4Hack ML Challenge 2026, Round 2 (team "idk bro")

Code for our Round 2 solution: nine-class classification of ON/OFF radio cadences (6 panels × 16 × 128), macro-F1.
Train comes from GBT, test from PKS. The central finding: after whitening, PKS signals are ~0.74× as strong relative to
noise, so the pipeline trains and validates against a simulated PKS signal-to-noise ratio.

This repository contains code and documentation only. No competition data, model weights or prediction files are
included; everything is regenerated from the official data by the scripts below.

Final upload: **U12 = stack `smx22_bag`** (21 CNN members + physics and panel-level features, LightGBM stacker
averaged over 5 seeds; held-out simulator macro-F1 0.9512). Best public score before it: U10 `smx19_bag`, 0.9392. History and every decision with its evidence:
`SUBMISSIONS.md`, `LOG.md`, `LEARNINGS_AND_DECISIONS.md`.

## Pipeline

1. **Whitening** (`src/prep.py`): per-panel level, cadence bandpass by a two-pass running median across channels
   (narrow lines masked so they survive), robust noise scale, GBT centre spike (channel 64) interpolated in both sets,
   flat broadband glitch rows removed in the `2` variant. Zero values are treated as missing.
2. **Physics features**: per-panel drift search and global-time ON-track search (`src/phys.py`), connection test
   (global slope vs local ON drift), broadband/dispersion, ON/OFF texture ratios (`src/tex.py`), pre-whitening panel
   levels for scintillation (`src/levels.py`).
3. **CNNs** (`src/cnn.py`): global-time image (panel p, row r at `p*(16+gap)+r`) with ON flag, valid mask and panel-level
   channels. Own ResNet variants (v1, v4p, v4a, v4ap, v6, v7) and ImageNet-pretrained timm backbones
   (EfficientNet-B0/B2, EfficientNetV2-S/M/L, ConvNeXt-Nano/Tiny, RegNetY-016, MobileNetV3-L via `--arch timm:<name>`).
   Training options: SNR-matching augmentation (`--snr_p`), pseudo-labels from confident test predictions (`--pseudo`),
   synthetic ETI tracks and hard negatives (`--syn`), MixUp (`--mixup 0.4`).
4. **PKS simulation**: training copies with extra noise (s = 0.92 → 0.74× SNR). A held-out simulator (s = 0.75, seed 7)
   is only ever scored, never trained on; it predicted the public score within 0.0003 of the observed gain.
5. **Mixed stacker** (`src/stack_mix.py`): LightGBM on physics features + log CNN probabilities, trained on clean rows
   plus simulated rows, 5 fixed stratified folds; the final stack averages 5 LightGBM seeds (`src/run_bag.ps1`,
   `src/bag_stack.py`). Candidates are compared with a paired bootstrap on the held-out simulator (`src/paired.py`).
6. **Submission** (`src/make_sub.py`): validated CSV (10,000 unique ids, valid labels, sample order).

## Reproduce the final upload (U12, `smx22_bag`)

Place `train_cadences.npy`, `test_cadences.npy`, `train_meta.csv`, `test_meta.csv`, `sample_submission.csv` in the repo root.

```bash
pip install -r requirements.txt
mkdir -p oof cache ckpt
python src/s0_baseline.py                         # oof/folds.npy: StratifiedKFold(5, shuffle, seed 42)
python src/prep.py && python src/prep.py 2        # whitened caches
python src/levels.py && python src/levels.py z    # panel-level features and channel
python src/phys.py 2 e && python src/tex.py 2     # physics + texture features (merge into cache/*_phys2f.parquet, see LOG.md E9)
# simulated copies: cache/train_w2s.npy (s 0.92, seed 123) and cache/train_w2t.npy (s 0.75, seed 7), then
python src/phys.py 2s e && python src/tex.py 2s   # features on the simulated copy (phys2sf), and run_sim2.py for the held-out copy
```

CNN members (each 5 folds; `src/modal_submit.py run <name> "<args>" 0,1,2,3,4 A100|H100` runs them on Modal,
or call `python src/cnn.py --name <name> --folds 0,1,2,3,4 <args>` locally):

| Member | Args |
|---|---|
| cnn_v1 | `--epochs 20` |
| cnn_v4p | `--epochs 30 --width 48 --pseudo oof/pseudo_stk_v3.npy` |
| cnn_v4a | `--epochs 30 --width 48 --snr_p 0.7 --snr_max 1.0` |
| cnn_v4ap | `--epochs 30 --width 48 --snr_p 0.7 --snr_max 1.0 --pseudo oof/pseudo_smx3.npy` |
| cnn_v6 | `--epochs 30 --width 48 --snr_p 0.7 --snr_max 1.0 --lvch 1 --pseudo oof/pseudo_smx4.npy` |
| cnn_v7 | `--epochs 30 --width 48 --snr_p 0.7 --snr_max 1.0 --lvch 1 --syn 0.1 --pseudo oof/pseudo_smx4.npy` |
| n_pretrained | `--arch timm:efficientnet_b0 --epochs 20 --lr 1e-3 --bs 64 --snr_p 0.7 --snr_max 1.0 --lvch 1 --pseudo oof/pseudo_smx4.npy` |
| n_cnxt | `--arch timm:convnext_nano --epochs 20 --lr 1e-3 --bs 64 --snr_p 0.7 --snr_max 1.0 --lvch 1 --pseudo oof/pseudo_smx7a.npy` |
| s_regy | `--arch timm:regnety_016 --epochs 20 --lr 1e-3 --bs 64 --snr_p 0.7 --snr_max 1.0 --lvch 1 --pseudo oof/pseudo_smx7a.npy` |
| s_mbnv3 | `--arch timm:mobilenetv3_large_100 --epochs 20 --lr 1e-3 --bs 64 --snr_p 0.7 --snr_max 1.0 --lvch 1 --pseudo oof/pseudo_smx7a.npy` |
| n_cnxtt | `--arch timm:convnext_tiny --epochs 20 --lr 1e-3 --bs 64 --snr_p 0.7 --snr_max 1.0 --lvch 1 --pseudo oof/pseudo_smx8a.npy` |
| n_effb2 | `--arch timm:efficientnet_b2 --epochs 20 --lr 1e-3 --bs 64 --snr_p 0.7 --snr_max 1.0 --lvch 1 --pseudo oof/pseudo_smx8a.npy` |
| s_effv2s | `--arch timm:tf_efficientnetv2_s --epochs 20 --lr 1e-3 --bs 64 --snr_p 0.7 --snr_max 1.0 --lvch 1 --pseudo oof/pseudo_smx11_all.npy` |
| n_cnxtt_mix | `--arch timm:convnext_tiny --epochs 25 --lr 1e-3 --bs 64 --snr_p 0.7 --snr_max 1.0 --lvch 1 --mixup 0.4 --pseudo oof/pseudo_smx11_all.npy` |
| n_cnxtn_mix | `--arch timm:convnext_nano --lr 1e-3 --bs 64 --snr_p 0.7 --snr_max 1.0 --lvch 1 --mixup 0.4 --pseudo oof/pseudo_smx14_a.npy --epochs 25` |
| n_effv2s_mix | `--arch timm:tf_efficientnetv2_s --epochs 25 --lr 1e-3 --bs 64 --snr_p 0.7 --snr_max 1.0 --lvch 1 --mixup 0.4 --pseudo oof/pseudo_smx14_a.npy` |
| n_effb2_mix | `--arch timm:efficientnet_b2 --epochs 25 --lr 1e-3 --bs 64 --snr_p 0.7 --snr_max 1.0 --lvch 1 --mixup 0.4 --pseudo oof/pseudo_smx14_a.npy` |
| s_effv2m | `--arch timm:tf_efficientnetv2_m --epochs 25 --lr 1e-3 --bs 64 --snr_p 0.7 --snr_max 1.0 --lvch 1 --mixup 0.4 --pseudo oof/pseudo_smx11_all.npy` |
| n_effv2m_mix | `--arch timm:tf_efficientnetv2_m --epochs 25 --lr 1e-3 --bs 64 --snr_p 0.7 --snr_max 1.0 --lvch 1 --mixup 0.4 --pseudo oof/pseudo_smx14_a.npy` |
| s_effv2l | `--arch timm:tf_efficientnetv2_l --epochs 25 --lr 1e-3 --bs 64 --snr_p 0.7 --snr_max 1.0 --lvch 1 --mixup 0.4 --pseudo oof/pseudo_smx17_bag.npy` |
| s_effv2l_u10 | `--arch timm:tf_efficientnetv2_l --epochs 25 --lr 1e-3 --bs 64 --snr_p 0.7 --snr_max 1.0 --lvch 1 --mixup 0.4 --pseudo oof/pseudo_smx19_bag.npy` |

Pseudo-label files are the test probabilities of the previous best stack (`oof/<stack>__test.npy`), so members are
trained in teacher order: stk_v3 → smx3 → smx4 → smx7a → smx8a → smx11_all → smx14_a → smx17_bag → smx19_bag, each
teacher built with `src/stack_mix.py` on the members that existed at that point (see `LOG.md`).

```bash
M=cnn_v1,cnn_v4p,cnn_v4a,cnn_v4ap,cnn_v6,cnn_v7,n_pretrained,n_cnxt,s_regy,s_mbnv3,n_cnxtt,n_effb2,s_effv2s,n_cnxtt_mix,n_cnxtn_mix,n_effv2s_mix,n_effb2_mix,s_effv2m,n_effv2m_mix,s_effv2l,s_effv2l_u10
python src/stack_mix.py smx22 mix $M lv                                     # LightGBM default seed
for s in 1 2 3 4; do python src/stack_mix.py smx22_s$s mix $M lv,seed$s; done
python src/bag_stack.py smx22_bag smx22,smx22_s1,smx22_s2,smx22_s3,smx22_s4
python src/make_sub.py smx22_bag U12_smx22_bag                              # subs/U12_smx22_bag.csv
```

Hardware used: laptop RTX 4060 (bf16), Kaggle T4 (`--amp fp16`), Modal A100 / H100 (`modal_run.py`, one fold per
container). `SH_TORCH=2.8.0` deploys a torch 2.8 (CUDA 12.8) image for Blackwell GPUs.
