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
| E53 | 22:50 | n_cnxtt (ConvNeXt-Tiny) held-out 0.932–0.945; n_effb2 (EfficientNet-B2) 0.930–0.933 | | | strongest singles |
| E54 | 23:00 | smx11_effb2 = U6 + EffNet-B2: held-out ΔF1 +0.0025 (3.1σ), ΔLL −0.0034 (3.8σ) | – | 0.9444/0.1548 | passes gate |
| E55 | 23:00 | **smx11_all** = U6 + RegNetY + MobileNetV3 + ConvNeXt-Tiny + EffNet-B2 | – | clean 0.9636/0.1040; SIM 0.9398/0.1746; **HELD-OUT 0.9443/0.1543** | KEEP → U7: held-out ΔF1 +0.0022 (2.2σ), ΔLL −0.0039 (3.5σ); SIM ΔF1 +0.0055; tied with effb2 on held-out, better on both guards |
| U7 | 23:05 | **public 0.9307** (2nd; Epoch 0.9397 jumped to 1st; Ctrl+C 0.9292; PatrickXDEXTER 0.9254) | – | held-out 0.9443 | public gain +0.0075 = 3.4x held-out gain (+0.0022): part public noise (±0.006); direction agrees again |
| E56 | 23:40 | class log-biases fitted on held-out sim (split halves) on smx11_all | – | honest held-out 0.9443→0.9437 | REJECT (3rd time; family closed) |
| E57 | 00:10 | n_onoff (ON/OFF comparison net) in stack (smx12_onoff) | – | held-out 0.9451/0.1536; vs U7 ΔF1 +0.0008 (1.1σ), ΔLL −0.0007 | REJECT (noise) |
| E58 | 00:15 | per-member held-out scint F1: all 0.75–0.78 (v6 0.760 … effb2 0.783); ETI 0.675 (onoff) … 0.863 (cnxtt) | | | scint = shared ceiling; ETI scales with backbone |
| E59 | 00:30 | s_cnxts (ConvNeXt-Small) 0.9372 ≈ Tiny 0.9367; s_effv2s (EffNetV2-S) **0.9395** | | | ConvNeXt size not paying; EffNetV2 scales |
| E60 | 00:45 | n_cnxtt_mix = ConvNeXt-Tiny + MixUp 0.4, 25 ep | – | held-out **0.9411** / 0.2070 (vs 0.9367 no MixUp; ETI 0.875, scint 0.781) | KEEP: MixUp +0.004 single |
| E61 | 00:55 | **smx14_a** = U7 + effv2s + cnxtt_mix | – | clean 0.9652/0.1013; SIM 0.9418/0.1712; **HELD-OUT 0.9469/0.1499** | KEEP → U8: vs U7 held-out ΔF1 +0.0026 (2.6σ), ΔLL −0.0046 (4.6σ); ETI 0.885→0.896 |
| U8 | 00:49 | **public 0.9337** (2nd; Epoch 0.9397, Ctrl+C 0.9292, PatrickXDEXTER 0.9272) | – | held-out 0.9469 | U7→U8 public +0.0030 vs held-out +0.0026 (~1:1); gap to 1st 0.009 → 0.006 |
| E62 | 02:00 | smx14_bag = U8 stack, LightGBM seeds default+1..4 averaged | – | clean 0.9648/0.1002; SIM 0.9411/0.1707; HELD-OUT 0.9469/0.1489 | vs smx14_a: held-out ΔLL −0.0009 (3.2σ), ΔF1 0.0000; seeds alone held-out F1 0.9464–0.9471, SIM 0.9401–0.9418 (U8's default seed was the best on SIM). 10 test rows change → not an upload by itself; use bagging on the final stack |
| E63 | 02:40 | n_effb0_mix (B0 + MixUp, PL smx14_a) held-out 0.9311 / 0.2962 (no-MixUp 0.9297); n_cnxtn_mix (ConvNeXt-Nano + MixUp) **0.9391** / 0.2060 (no-MixUp 0.9363) | – | | MixUp helps singles again (+0.0014, +0.003) |
| E64 | 02:55 | smx16_ab = U8 + effb0_mix + cnxtn_mix; smx16_c = U8 + cnxtn_mix | – | ab held-out 0.9464/0.1493 (ΔF1 −0.0006, ΔLL −0.0005, 0.8σ; scint −0.004); c 0.9476/0.1491 (ΔF1 +0.0007, 1.1σ; ΔLL −0.0007, 1.2σ; ETI +0.005) | both below gate: small-backbone MixUp members are redundant in the 14-member stack (clean LL −0.0015 only) |
| E65 | 03:00 | n_effv2s_mix held-out 0.9396 / 0.2411 (no-MixUp 0.9395: no gain); n_effb2_mix 0.9345 / 0.2686 (no-MixUp 0.9317) | – | | MixUp gain is family-dependent |
| E66 | 03:10 | smx17_all = U8 + cnxtn_mix + effv2s_mix + effb2_mix; smx17_vb = U8 + effv2s_mix + effb2_mix | – | all: held-out 0.9475/0.1480 (ΔLL −0.0019, 2.4σ; ΔF1 +0.0005, 0.6σ; ETI +0.005); vb: 0.9468/0.1476 (ΔLL −0.0023, 3.0σ; ΔF1 −0.0002) | LL gate passes, F1 gate (+0.002) fails → no upload; MixUp-of-existing-members family closed (4 runs). Seed-bagging smx17_all to check if F1 follows LL |
| E67 | 03:20 | smx17_bag = smx17_all, 5 LightGBM seeds averaged | – | clean 0.9665/0.0986; SIM 0.9413/0.1694; **HELD-OUT 0.9477/0.1472** | vs U8 smx14_a: held-out ΔLL −0.0027 (3.6σ), ΔF1 +0.0008 (0.9σ), no class worse than −0.0006; vs smx14_bag: ΔLL −0.0017 (2.5σ), ΔF1 +0.0008, clean F1 +0.0017 (2.2σ). Best candidate so far; below the +0.002 F1 bar → hold, rebuild with V2-M / NFNet when Swarali is back |
| U9 | 05:58 | **public 0.9364** (4th; Ctrl+C 0.9573, Packet Pushers 0.9525, Epoch 0.9397) | – | held-out 0.9477 | public +0.0027 vs held-out +0.0008 (3.4x, like U7) |
| E68 | 06:10 | Gap check after leaders jumped +0.028: metadata (target repeats mixed-label, freq uniform per class, no train/test target leak), pixel drift already matched (D10), test confidence ≈ held-out (1.2% vs 1.4% below 0.6; uncertain pairs = scint vs noise/satellite/rfi_pers) | – | | no quick lever found; leaders' errors-vs-ours not diagnosable without labels. Target: 3rd (Epoch 0.9397, +0.0033) |
| E69 | 06:15 | Held-out error analysis U8 → U9: errors 780 → 768 (fixed 60, new 48); half of all U9 errors are missed scintillation (scint→noise 226, →rfi_pers 78, →satellite 57, →radar 38); next eti→rfi_int 39 | – | | scintillation recall is the main headroom; research-4 Exp 4 (scint view) not run: 3 scint-feature/bias families already closed (D58, D60), no time for a 5-fold pilot |
| E70 | 06:35 | s_effv2m (Swarali, PL smx11_all) held-out **0.9429** / 0.2271 (best single; V2-S 0.9395); s_nfl0 0.9362 / 0.2105 (clean LL 0.1677 best) | – | | V2 family scales S → M |
| E71 | 06:52 | smx18_m = smx17_all + s_effv2m: held-out ΔLL −0.0023 (3.1σ), ΔF1 +0.0005, SIM ΔF1 +0.0020 (2.4σ), ΔLL −0.0032 (4.1σ); scint +0.003, no class worse. smx19_mn (+ s_nfl0 too): ΔLL −0.0020, ΔF1 +0.0002, ETI −0.003 | – | | V2-M passes D75; NFNet-L0 redundant → dropped. n_effv2m_mix (ours, PL smx14_a) folds 0.940–0.946, n_nfl0 0.933–0.939. Building smx19_mm (both V2-M) × 5 seeds |
| E72 | 07:03 | **smx19_bag** = U9 members + s_effv2m + n_effv2m_mix, 5 seeds | – | clean 0.9681/0.0962; SIM 0.9428/0.1660; **HELD-OUT 0.9491/0.1442** | vs U9 (smx17_bag): held-out ΔLL −0.0030 (3.4σ), ΔF1 +0.0014; SIM ΔLL −0.0033 (3.8σ), ΔF1 +0.0015; clean ΔF1 +0.0017; ETI +0.0126, scint −0.0014 (noise) → U10 |
| U10 | 07:08 | **public 0.9392** (4th; Epoch 0.9397 by 0.0005; Ctrl+C 0.9573, Packet Pushers 0.9525) | – | held-out 0.9491 | public +0.0028 vs held-out +0.0014 (2x) |
| E73 | 07:30 | s_effv2l (Swarali, H100, PL smx17_bag) held-out **0.9452** (best single; V2-M 0.9429), 29 min, $8.29 | – | | V2 family keeps scaling: S 0.9395 → M 0.9429 → L 0.9452 |
| E74 | 07:49 | **smx20_bag** = U10 members + s_effv2l, 5 seeds | – | clean 0.9686/0.0951; SIM 0.9445/0.1644; **HELD-OUT 0.9498/0.1419** | vs U10: held-out ΔLL −0.0022 (4.6σ), ΔF1 +0.0006; SIM ΔF1 +0.0018 (2.4σ); scint +0.0034; no class worse → U11 |
| U11 | 07:49 | **public 0.9365** (U10 0.9392 stays best; 6 uploads left) | – | held-out 0.9498 | public −0.0027 vs held-out +0.0007: direction disagrees for the first time; public noise ±0.006 → only upload stacks clearly above U10 on held-out |
| E75 | 08:24 | smx21_bag = U11 members + n_effv2m_u9 (3rd V2-M, PL smx17_bag), 5 seeds | – | clean 0.9685/0.0947; SIM 0.9441/0.1641; HELD-OUT 0.9501/0.1422 | vs U11: held-out ΔLL +0.0003, ΔF1 +0.0003 (0.5sd); SIM F1 −0.0004 → REJECT (3rd V2-M redundant). Note: first bag attempt ran all seeds at default (run_bag.ps1 passed 'seedN' as a separate arg; stack_mix reads only argv[4] → fixed to 'lv,seedN') |
| E76 | 08:53 | s_effv2l_u10 (V2-L, PL smx19_bag) held-out **0.9467** (best single). **smx22_bag** = U11 members + s_effv2l_u10, 5 seeds | – | clean 0.9694/0.0940; SIM 0.9447/0.1637; **HELD-OUT 0.9512/0.1411** | vs U10: held-out ΔLL −0.0031 (4.4sd), ΔF1 +0.0021 (2.3sd); SIM ΔF1 +0.0020; scint +0.0034, ETI +0.011, no class worse → passes full gate → U12 candidate. vs U11: ΔF1 +0.0015 (1.9sd), ΔLL −0.0009 |
