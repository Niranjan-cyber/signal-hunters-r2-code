# Signal Hunters R2 — learning log

Skill: ml-hackathon (Skill-maker HEAD, commit 27 Oct 04:34 UTC). Window 27 Oct 10:00 → 23:59 IST.
Gates (offsets from 23:59): feature freeze 22:45, code freeze 23:15, final file 23:30. Code link due before close.

## Rules digest
- Metric macro F1 over 9 labels; public 30% / private 70%; final = **best public upload** → every upload is a final
  candidate, never upload a worse-CV probe. **5 uploads in the whole round.** LB refresh 15 min.
- Train data only; pretrained public weights OK; no outside labelled data, no hand-labelling test.

## Recon: fact → consequence
| Fact | Consequence |
|---|---|
| tsamp 18.25 vs 16.0, chan width 2.79 vs 3.81 Hz, constant per telescope | Drop as features (telescope ID) |
| center_freq train 1100–1900, test 1200–1500 | Drop |
| 1 Hz/s = 6.54 ch/step GBT, 4.20 ch/step PKS | Check whether test drifts are in pixel or Hz/s units before trusting drift features |
| slew_gap 3–6 same dist in train/test, label-independent | Use only for the ON-track geometry |
| PKS vs GBT: drop rows 0.38% vs 0.15%, glitch rows 1.7% vs 0.9%, panel-level sd 2.6 vs 1.7, noise sd 16.7 vs 12.3, ripple sd 10.0 vs 7.0 | Per-panel bandpass flatten + per-panel noise standardise (robust MAD); mask dropped and glitch rows; augment train with PKS-like noise/ripple/glitches/gain |
| S0 baseline (starter feats, HGB): OOF F1 0.523, LL 1.217, but test preds 73% radar_chirp | Raw panel stats do not transfer; NOT an upload candidate |
| eti_candidate 2.7% of train; test label mix differs | Macro-F1 class multipliers / prior shift; don't learn train priors |

## Hypothesis
Each cadence = instrument (bandpass × panel gain + noise + glitch/dropped rows) + injected sources, label = top source by
priority. Model that matches: robust per-panel normalisation (whitening) → physics features (drift search per panel, ON
track consistency, dedispersed broadband, ON/OFF texture) + a small 2D CNN on 6-panel whitened input with
instrument augmentation; check adversarial AUC train-vs-test on whitened inputs.

## Validation of the research plan (10:35)
- Agree: physics+GBDT and 6-panel CNN as the two families, OOF blend, split-half class multipliers, 5-upload discipline.
- Changed: per-channel time-median flattening per panel ERASES non-drifting lines (rfi_persistent, zero-drift traps)
  → bandpass = cadence time-median spectrum, two-pass running median across channels (lines masked). Line SNR raw 12.3 → 13.1.
- Changed: GBT centre spike ch64 (+35) absent in PKS → interpolate ch64 in both sets.
- Changed: upload 1 ≠ starter (test preds 73% radar_chirp; best-public-counts → every upload is a final candidate).
- Targets ~unique (max 4 cadences/star) → plain stratified 5-fold OK (folds in oof/folds.npy).
- CNN input = global-time image (panels at p*(16+g)+r, gaps empty) + ON indicator + valid mask, so an ETI track is a straight line.

## Noise floor (OOF, 18k rows, bootstrap 200)
LL sd 0.008, paired LL diff sd 0.003, F1 sd 0.004, paired F1 sd 0.002. **Keep bar: paired ΔLL < −0.007 (full OOF), ~−0.015 fold 0.**

## Experiments
| ID | Time | Change | Fold0 F1 / LL | OOF F1 / LL | Decision |
|---|---|---|---|---|---|
| S0 | 10:15 | starter feats (no tel. meta) HGB | – | 0.523 / 1.217 | reject: test 73% radar |
| E1 | 10:50 | physics feats v1 + LGB | 0.774 / 0.625 | 0.7804 / 0.6041 | baseline family A; adv AUC 0.90 (broadband feats) |
| E2 | 11:00 | + deglitch (flat bright rows minus row median) | 0.791 / 0.591 | 0.7924 / 0.5731 | KEEP ΔLL −0.031 (9σ); adv AUC 0.88 |
| E3 | 10:45 | CNN v1 global-time, w32, 20 ep, aug, w (no deglitch) | ep15 0.847 / 0.294 | running | |
| E3 | 10:57 | (result) CNN v1 fold0 | 0.870 / 0.254 | – | KEEP as family B. ETI F1 0.37 (69/96 ETI → rfi_intermittent): can't link ON segments |
| E4 | 11:00 | CNN v2 = v1 on deglitched input | 0.857 / 0.251 | – | inconclusive (ΔLL −0.003, bar 0.015); keep v1 |
| E5 | 11:00 | stack proto fold0 inner-CV: phys v2 + log CNN probs | 0.902 / 0.222 | – | stacker route; avg 0.890 / geo LL 0.275 |
| E6 | 11:03 | phys v3: ON-mean / ON-median track scores (faint ON panel) | 0.794 / 0.567 | 0.7967 / 0.5544 | KEEP ΔLL −0.019 (9σ) |
| E7 | 11:12 | + robust tex (ON/OFF log-var ratios, ON−OFF) = phys2c | 0.823 / 0.509 | 0.8220 / 0.4942 | KEEP ΔLL −0.060 (18σ); test dist stable |
| E8 | 11:12 | + absolute ON tex var = phys2d | 0.835 / 0.474 | 0.8342 / 0.4595 | HOLD: test radar 362→678, tv_on_2x8 in adv top; test-noise p10 0.91 vs train-noise 0.75 |

Facts: test ETI-candidate tracks drift less (median 0.075 vs train ETI 0.195 ch/row; PKS/GBT pixel scale 0.64).
Lanes: laptop CNN v1 f1,2; Kaggle lane1 CNN v1 f3,4; lane2 CNN v3 (deep, RF spans cadence) f0.
| E9 | 11:22 | + connection test (global slope vs local ON drift) = phys2f | 0.828 / 0.500 | 0.8294 / 0.4896 | KEEP: ETI F1 0.762→0.830, ΔF1 +0.0074 (3.5σ), ΔLL −0.0046 (2.7σ paired) |
Fact: drift distribution of strong tracks matches train/test in PIXEL units (median 0.225 both), not Hz/s → keep pixel features.
| E10 | 12:01 | CNN v4 (w48, 30 ep) fold0 | 0.889 / 0.236 | – | KEEP vs v1 ΔLL −0.018 (> fold0 bar 0.015); test ETI 147 vs 64 |
| E11 | 12:35 | CNN v1 all 5 folds | – | 0.8748 / 0.2438 | family B reference |
| E12 | 12:36 | STACK stk_v1: LGB on phys2f + log CNN v1 probs | – | **0.9338 / 0.1721** | KEEP → U1. ETI 0.849, scint 0.752 weakest |
| E13 | 12:36 | split-half class multipliers on stk_v1 | – | honest 0.9338 → 0.9333 | REJECT (no gain) |
Kaggle lanes stopped 12:15: bf16 emulated on T4 → 280 s/epoch vs 20 s laptop. Fix: --amp fp16 (verify locally first).
| E14 | 13:03 | CNN v5 = v4 + deep stage (Kaggle fp16, 45 s/ep) fold0 | 0.884 / 0.240 | – | REJECT vs v4 ΔLL +0.011; ETI 0.48 vs 0.63 → ETI logic stays in physics feats |
Kaggle fp16 works: 45 s/epoch (w48) on T4 vs 280 s with bf16.
| U1 | 13:15 | **public 0.8823** (3rd; top 0.8858, 2nd 0.8839) | – | CV 0.9338 | CV→LB gap −0.051 = telescope shift is the main lever |
| E15 | 13:20 | CNN v4 all folds (3,4 from Kaggle fp16) | – | 0.9038 / 0.2264 | KEEP |
| E16 | 13:22 | stk_v3 = phys2f + CNN v1 + CNN v4 | – | **0.9368 / 0.1634** | KEEP ΔLL −0.0087 (6σ) vs stk_v1; not uploaded alone (expected +0.003 < LB noise) |
| E17 | 13:24 | shift diagnosis: test low-conf 3.7% vs OOF 1.9%; scint low-conf 17% vs 8.7% (scint|noise, scint|rfi_pers); test rfi_int share 19.5% vs 15% | | | → pseudo-labels |
| E18 | 13:26 | CNN v4p = v4 + 8,251 test pseudo-labels (stk_v3 conf ≥ 0.9; OOF acc at that conf 0.978) | running | | distribution-changing: needs paired LB probe vs stk_v3; keep a non-PL final candidate |
| A2 | 14:55 | physics LGB trained on clean+degraded mix | – | clean 0.8307/0.481; SIM 0.7691/0.668 (vs clean-trained SIM 0.7287/0.822) | KEEP: SIM ΔLL −0.155, clean no loss |
| E19 | 15:00 | CNN v4p (pseudo) all folds: SIM LL f0–4 0.377/0.399/0.375/0.355/0.379 vs v1 0.40–0.43 | | | KEEP (PL helps PKS-sim robustness) |
| E20 | 15:05 | mix stacker on v1: SIM 0.867→0.888, clean 0.1725→0.1693 | | | KEEP mix-stacking |
| E21 | 15:10 | **smx_v1p_mix** = mix stacker, CNN v1 + v4p | – | clean 0.9430/0.1492; **SIM 0.8999/0.2802** | → U2 candidate |
| E22 | 15:30 | CNN v4a (SNR-matched aug p0.7, sd U(0.4,1.0), renorm) fold0 | clean 0.890 / 0.223; SIM 0.840 / 0.338 | – | KEEP: SIM ΔLL −0.077 vs v4 (bar −0.03), clean LL −0.006 |
| E23 | 15:40 | CNN v4a all folds | – | clean 0.9011/0.2164; SIM 0.8516/0.3365 (best CNN on SIM) | KEEP |
| E24 | 15:42 | mix stacker v1+v4a (no PL) | – | clean 0.9389/0.1521; SIM 0.8982/0.2684 | non-PL fallback → subs/C_smx_v1v4a_noPL.csv |
| E25 | 15:42 | mix stacker v1+v4p+v4a | – | clean 0.9461/0.1449; SIM 0.9010/0.2633 | best so far → subs/C_smx_v1v4pv4a.csv (ΔSIM LL −0.017 vs U2) |
| E26 | 15:45 | CNN v4ap = SNR aug + PL from smx_v1v4pv4a (8,388 rows ≥0.9), Modal 5 folds | | | running |
| U2 | 15:19 | **public 0.9017** (+0.0194 vs U1; 3rd; top 0.9120, 2nd 0.9056) | – | clean 0.9430, SIM 0.8999 | SIM OOF predicts public within 0.002 → SIM is the main selection metric |
RULE CHANGE (platform UI, 15:59): 10 uploads/day, reset at midnight IST; "Round 2 ends 28 Oct 10:00 IST". Not yet confirmed in writing → keep 23:59 tonight as safety gate.
| E27 | 16:05 | SIM error analysis: scint SIM F1 0.66 (352/1431 → noise), headroom +0.032 macro; ETI 0.81 (+0.016) | | | target scint |
| E28 | 16:10 | whitening removed panel level; scint lifts ON panels (+2/−2 raw alternating). lv_alt AUC 0.903 scint/noise, 0.905 scint/rfi_pers; train/test noise p50/p90 0/0.42 vs 0/0.48 | | | keep 5 robust lv feats, drop lv_sd/lv_nonalt (gain-change shift) |
| E29 | 16:12 | smx3_lv = mix stacker v1+v4p+v4a + lv | – | clean 0.9529/0.1262; **SIM 0.9164/0.2324**; scint SIM 0.66→0.77 | KEEP SIM ΔLL −0.031 (15σ), clean −0.019 (12σ) → U3 |
| U3 | 16:08 | **public 0.9087** (2nd; top 0.9120) | – | SIM 0.9164 | public Δ vs U2 +0.0070 (SIM Δ +0.0165); SIM−public gaps: U2 −0.002, U3 +0.008 → SIM still the selection metric, expect ~0.5× of SIM gains on public |
| E30 | 16:50 | held-out simulator (s 0.75 = milder, seed 7, never trained on) | U3 held-out 0.9279/0.1947 | | U3 gains hold on independent sim |
| E31 | 16:50 | wide ETI filter: SIM ΔLL −0.0016 (2.3σ), held-out −0.0011 (1.8σ), clean +0.0003; ETI SIM 0.806→0.82 | | | marginal, same direction on both sims |
| E32 | 17:00 | gain stress: lv features recomputed with +0.34 z panel offsets (measured PKS extra spread) on held-out rows | U3 held-out 0.9279→0.9267, scint 0.762→0.756 | | lv features robust to PKS gain changes → keep |
| E33 | 17:20 | CNN v4ap (SNR aug + PL from smx_v1v4pv4a) all folds | – | SIM LL f0–4 0.299/0.308/0.288/0.281/0.298; held-out 0.26 | best CNN |
| E34 | 17:25 | **smx4** = mix stacker v1+v4p+v4a+v4ap + lv | – | clean 0.9530/0.1230; SIM 0.9226/0.2132; HELD-OUT 0.9324/0.1834 | KEEP vs U3: held-out ΔLL −0.0113 (7.5σ) ΔF1 +0.0043; SIM ΔLL −0.019; clean −0.003 → U4 |
| E35 | 17:25 | wide ETI filter on smx4: held-out ΔLL −0.0010, ΔF1 −0.0005; clean +0.0005 | | | DROP (noise-level, 3rd run below bar → family closed) |
| E36 | 17:45 | EM label-shift correction on smx4 | – | held-out paired ΔF1 −0.0009 ± 0.0013; 36 test rows change | REJECT (D42) |
| E37 | 18:15 | CNN v6 = v4ap + level channel + PL from U4 (8,902 rows) | – | per-fold SIM LL 0.249–0.267; held-out 0.218–0.236 (v4ap ~0.26) | KEEP |
| E38 | 18:50 | smx5 = stacker v1+v4p+v4a+v4ap+v6 + lv | – | clean 0.9538/0.1224; SIM 0.9249/0.2086; HELD-OUT 0.9347/0.1794 | KEEP vs U4: held-out ΔLL −0.0040 (3.3σ), ΔF1 +0.0023; not uploaded today (< +0.005 bar) |
| E39 | 19:05 | synthetic ETI injection check: syn ETI track score 5.96 / ON-med 6.29 / slope mismatch 0.05 vs real ETI 6.75 / 6.81 / 0.04; drift median 0.18 (log-uniform) | | | injection realistic |
| E40 | 19:10 | CNN v7 = v6 + synthetic ETI positives & hard negatives (syn 0.1); v6 seeds 1, 2 | | | running on Modal |
| U5 | 18:55 | **public 0.9160** (2nd; top 0.9213) | – | SIM 0.9249, held-out 0.9347 | U3→U5 public +0.0073 vs held-out +0.0068, SIM +0.0085: public now tracks validation gains ~1:1 |
| E41 | 20:00 | LR stacker on CNN log-probs, geometric blend 0.3 with smx5 | – | held-out 0.9331/0.1773 vs smx5 0.9347/0.1794 | REJECT (F1 down; LL gain only) |
| E42 | 20:20 | AdaBN on v6 (BN stats from held-out-sim rows; test from PKS) | – | held-out F1 worse 3/4 folds | REJECT |
| E43 | 20:45 | smx6a = smx5 + cnn_v7 | – | clean 0.9559/0.1209; SIM 0.9267/0.2053; HELD-OUT 0.9357/0.1783 | held-out ΔF1 +0.0010 (0.9σ), ΔLL −0.0011; SIM ΔLL −0.0033 (3.7σ); below +0.003 gate but ≥ U5 everywhere → tonight's expiring slot if nothing better by 22:15 |
| E44 | 20:55 | n_dann (DANN λ 0.3, A100) all folds | – | held-out f0–4 0.923/0.914/0.921/0.925/0.924 | CNN ≈ v6 |
| E45 | 21:05 | n_pretrained (timm EfficientNet-B0 ImageNet, 20 ep, A100) all folds | – | held-out f0–4 0.931/0.927/0.929/0.932/0.929 | best single CNN on held-out |
| E46 | 21:20 | smx7a = smx6a + n_pretrained | – | clean 0.9580/0.1141; SIM 0.9299/0.1924; **HELD-OUT 0.9390/0.1680** | KEEP: vs U5 held-out ΔF1 +0.0043 (3.3σ), ΔLL −0.0114 (7σ); scint 0.768→0.781 → tonight's upload |
| E47 | 21:20 | smx7b (+DANN) held-out 0.9355/0.1781; smx7c (+both) vs smx7a ΔF1 −0.0004 | | | DANN REJECT (redundant) |
| E48 | 21:30 | n_res34 (timm ResNet34, PL smx7a) | – | held-out 0.909–0.923 | weaker single; stack +res34 (smx8b vs smx8a) no gain → REJECT |
| E49 | 21:40 | n_cnxt (timm ConvNeXt-Nano, PL smx7a) | – | held-out 0.932–0.942, LL 0.189–0.207 | best single model |
| E50 | 21:50 | **smx8a** = smx7a + n_cnxt | – | clean 0.9612/0.1080; SIM 0.9345/0.1840; **HELD-OUT 0.9419/0.1583** | KEEP: vs smx7a held-out ΔF1 +0.0032, ΔLL −0.0096 (6σ); vs U5 ΔF1 +0.0074; ETI 0.867→0.881 → tonight's upload |
| E51 | 22:00 | smx9a = smx8a + s_v8 (PL round 2) | – | held-out 0.9425/0.1579 vs 0.9419/0.1583 | REJECT (noise); PL round 2 closed |
| E52 | 22:35 | smx10_both = smx8a + s_regy + s_mbnv3 (single adds: +0.0007 each) | – | held-out 0.9432/0.1576; vs smx8a ΔF1 +0.0011 (1.4σ), ΔLL −0.0006 (<1σ); SIM ΔLL −0.0029 (3.6σ) | HOLD: below gate, not uploaded; stack saturating (~+0.001 per family) |
| U6 | 21:47 | **public 0.9232** (2nd; PatrickXDEXTER 0.9254, Loss Minimizers 0.9219, terminal-R 0.9213) | – | held-out 0.9419 | U5→U6 public +0.0072 = held-out +0.0072 exactly; public = held-out − 0.0187 for both U5 and U6 |
