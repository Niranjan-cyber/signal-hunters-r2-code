# Signal Hunters R2: learnings and decisions

Running record of what we decided, why, and what we learned. The per-run numbers live in `LOG.md`, the uploads in
`SUBMISSIONS.md`. Times are IST, 27 Oct 2026. Updated whenever a decision is taken or a lesson shows up.

## Current state (19:35)
- Best public: **U5** `subs/U5_smx5.csv` = **0.9160**, 2nd (leader terminal-R 0.9213). Uploads today: 4 used, 1 left (U4 built but never uploaded).
- U5 = mixed stacker on CNN v1 + v4p + v4a + v4ap + v6 + physics + panel-level features. Clean OOF 0.9538 / 0.1224, PKS-sim 0.9249 / 0.2086, held-out sim 0.9347 / 0.1794.
- Public history: U1 0.8823 → U2 0.9017 → U3 0.9087 → U5 0.9160. Public now tracks held-out-sim gains ~1:1 (U3 → U5: +0.0073 vs +0.0068).
- Remaining held-out headroom: scintillation 0.768, ETI 0.861; every other class ≥ 0.95.
- In progress: v7 (synthetic ETI) folds 2–4 + L4 vs A100 timing on Modal (spawned calls); teammate lanes per `coord/BOARD.md`.
- Deadline 28 Oct 10:00 IST, 5 uploads that day; repo goes public just before (checklist below).

## Decisions

| # | Time | Decision | Why (evidence) | Alternatives rejected |
|---|---|---|---|---|
| D1 | 10:20 | Drop `tsamp_s`, `channel_width_hz`, `center_freq_mhz` as features | Constant per telescope or different range in test, so they only identify the telescope | Keep them (starter does) |
| D2 | 10:20 | Starter baseline is not an upload candidate | Test predictions 73% `radar_chirp`: raw panel stats do not transfer | Upload as "pipeline check" (research plan's U1) |
| D3 | 10:40 | Bandpass = cadence time-median spectrum, two-pass running median across channels (narrow lines masked) | Per-channel time-median flattening (research plan) erases non-drifting lines; two-pass keeps line SNR 12.3 → 13.1 | Per-panel per-channel median division |
| D4 | 10:40 | Interpolate channel 64 in both sets | GBT has a +35 centre spike that PKS lacks | Leave it (model would learn an always-on line) |
| D5 | 10:45 | Plain stratified 5-fold, one `folds.npy` for every model | Targets almost unique (max 4 cadences per star), so grouping is unnecessary | Group by `target` |
| D6 | 10:45 | CNN input = global-time image (panels at `p*(16+gap)+r`) + ON flag + valid mask | An ETI track becomes a literal straight line | Six panels as channels |
| D7 | 11:00 | Remove flat bright rows (glitches) in physics features | Glitch rate 0.63% in every train class, 1.24% in test; LGB ΔLL −0.031 (9σ) | Mask all bright rows (would hit pulsars/FRBs) |
| D8 | 11:00 | Keep CNN v1 on the plain input, not the deglitched one | v2 ΔLL −0.003, inside the 0.015 fold-0 bar; CNN already sees glitch augmentation | — |
| D9 | 11:05 | Stack (LGB on physics + log CNN probs) instead of averaging | CNN can't link ON segments (ETI F1 0.37); physics features can; fold-0 stack 0.902 vs average 0.890 | Equal-weight average |
| D10 | 11:15 | Keep drift features in pixel units | Strong-track drift percentiles match train/test in channels per row (median 0.225 both), not in Hz/s | Convert to Hz/s |
| D11 | 11:15 | Main line uses only robust texture features (ON/OFF ratios); absolute ON variance held back | Absolute variance AUC 0.93 but shifted (test-noise p10 0.91 vs train-noise 0.75) and doubles test `radar_chirp` | Use all texture features |
| D12 | 11:22 | Add the connection test (global slope vs local ON drift) | ETI F1 0.762 → 0.830; targets the "same drift, does not connect" trap | — |
| D13 | 12:01 | Laptop runs the folds that matter; Kaggle only after fp16 | Kaggle bf16 was emulated: 280 s/epoch vs 20 s locally; fp16 gives 45 s | Wait for Kaggle |
| D14 | 12:01 | CNN v4 (width 48, 30 epochs) replaces v1 as main CNN | Fold-0 ΔLL −0.018 (> 0.015 bar), more ETI found; full OOF 0.904 vs 0.875 | Seeds of v1 |
| D15 | 12:36 | No class multipliers | Split-half honest F1 did not improve (0.9338 → 0.9333; again on stk_v2/v3) | Use multipliers |
| D16 | 13:00 | Upload U1 = stk_v1 | Real final candidate, and the first CV → public reading | Wait for v4 |
| D17 | 13:03 | Reject the deep CNN (v5) | Fold-0 ΔLL +0.011, ETI F1 0.48 vs 0.63; ETI logic stays in physics features | — |
| D18 | 13:22 | Do not upload stk_v3 on its own | Expected public gain +0.003 < leaderboard noise ±0.006; slots are scarce | Upload now |
| D19 | 13:26 | Pseudo-label round (test conf ≥ 0.9, 8,251 rows) for CNN and LGB | Shift sits in scint (low-conf 17% vs 8.7%) and persistent/intermittent RFI; test is 55% of train size | Augmentation-only, domain-adversarial training |
| D20 | 13:45 | Judge pseudo-labels on the leaderboard, not OOF; keep stk_v3 as the non-PL final candidate | Teacher saw all train rows, so student OOF is optimistic (LGB PL OOF ΔLL −0.0097 is partly leakage) | Trust OOF gain |
| D21 | 14:00 | After validating the second research doc, run SNR matching first | Measured: confident test signals are 0.76–0.89x train strength (persistent/intermittent RFI, pulsar, FRB 0.76; satellite 0.85; ETI 0.89), close to the noise ratio 12.3/16.7 = 0.74. Current CNN aug adds at most ~6% noise and never renormalises | Ripple aug (residual column std equal 0.274 vs 0.274: not needed); shift TTA (flip TTA already in) |
| D22 | 14:00 | New validation proxy: PKS-simulated fold (noise s = 0.92, renormalised, giving 0.74x SNR) | v4 fold 0 clean 0.900/0.229 → sim 0.825/0.415 (−0.075 F1), the size of the observed CV → public gap; v1 sim 0.800/0.427 | Research doc's rule "fold-0 −0.015 on clean val": wrong test for a shift lever |
| D23 | 14:05 | CNN v4a = v4 + SNR-matching aug (p 0.7, extra sd U(0.4, 1.0), renormalised); 5 folds in parallel on Modal L4 | Pre-set keep rule: fold-0 SIM ΔLL < −0.03 and clean LL no worse than +0.015 | — |
| D24 | 14:10 | A2: physics features recomputed on an SNR-degraded train copy (`cache/train_w2s`, s 0.92, seed 123) | Physics thresholds were learned at GBT SNR; test features shrink to 0.76x | — |
| D25 | 14:15 | Keep the stacker-confidence pseudo-label filter (no all-family agreement filter) | All-agree keeps 6,999 / 8,251, but ETI agreement is 28% (LGB 97%, CNN v4 60%) and radar 43% (LGB 42%, CNN v4 100%): disagreement reflects known family blind spots, not label noise | Agreement filter (would drop ETI/radar pseudo-labels exactly where the weak family needs them) |
| D26 | 14:55 | Physics LGB trained on clean + SNR-degraded rows | SIM LL 0.822 → 0.668 (−0.155), clean LL 0.490 → 0.481; degraded-only loses clean (0.552) | Degraded-only |
| D27 | 15:05 | Stacker trained on clean + PKS-sim rows (degraded physics feats + CNN preds on sim validation) | On CNN v1: SIM F1 0.867 → 0.888, LL 0.369 → 0.307; clean LL 0.1725 → 0.1693 | Clean-only stacker |
| D28 | 15:10 | U2 = smx_v1p_mix (mix stacker on CNN v1 + pseudo-label CNN v4p) | Clean 0.9430 / 0.1492, SIM 0.8999 / 0.2802 vs U1 recipe SIM 0.867; every class up on SIM (ETI 0.75 → 0.81, scint 0.60 → 0.66). 344 test rows differ from U1 | Wait for v4a (goes into U3) |
| D29 | 15:10 | Modal launches only as detached Windows processes; fetch outputs file by file | Memory reap of the local launcher killed all 5 v4a containers (lost ~$2); `modal volume get` of a folder overwrote files into one path | Background shell launches |
| D30 | 15:59 | PKS-sim OOF becomes the main selection metric; public only as a calibration check | U2 SIM OOF 0.8999 vs public 0.9017 (within 0.002); clean OOF overstates public by 0.04; SIM uses all 18k rows vs ~3k public rows (noise ±0.006) | Select by public score |
| D31 | 15:59 | Treat the platform's "Round 2 ends 28 Oct 10:00, 10 uploads/day" as likely but keep 23:59 tonight as the safety gate until confirmed in writing | Platform UI changed mid-event; playbook: earliest stated deadline is real until confirmed | — |
| D32 | 16:12 | Add 5 panel-level features (pre-whitening ON/OFF level pattern); drop lv_sd, lv_nonalt | Whitening erased the scint cue (ON panels +2, OFF −2 raw). lv_alt AUC 0.90; train/test noise distributions match; lv_sd/lv_nonalt follow PKS gain changes | Absolute ON texture variance (still held back) |
| D33 | 16:12 | U3 = smx3_lv; C_smx_v1v4pv4a superseded, not uploaded | SIM 0.9164 / 0.2324 vs 0.9010 / 0.2633 (ΔLL −0.031, 15σ); clean ΔLL −0.019 (12σ); scint SIM 0.66 → 0.77 | Upload the older candidate too |
| D34 | 16:15 | Private-leaderboard policy: upload only files we would accept as the final pick (best SIM, clean as guard), never public-tuned probes | Final ranking takes the best-public upload; a lucky weaker probe that wins public becomes the private entry | Probe many variants on public |
| D35 | 16:17 | U3 public 0.9087 (2nd). Expect ~0.5x of SIM gains on public; add a held-out simulator (s 0.75, new seed) that no model trains on | U2→U3: public +0.0070 vs SIM +0.0165; the lv features are scaled by exactly 0.74 in SIM rows, which flatters SIM | Trust SIM deltas 1:1 |
| D36 | 16:20 | Wide matched filter (3-channel boxcar) for ETI track scores | ETI SIM recall 0.739 vs clean 0.879; 50 of 71 SIM misses are hits on clean; missed tracks have SIM onmed 2.85 vs 5.81 for hits | Re-tune connection thresholds |
| D37 | 16:50 | Held-out simulator (s 0.75, seed 7) scored but never trained on; all future candidates must hold on it | Third research doc asked for independent validation. Note: s 0.75 is MILDER (0.80x SNR) than the training sim (0.92 → 0.74x), not harder as the doc states | Treat s 0.75 as "harder" |
| D38 | 17:00 | Keep the panel-level scint features | Gain stress test (+0.34 z random panel offsets = measured PKS extra spread for noise class) costs only −0.0012 held-out F1, scint 0.762 → 0.756 | Redesign normalisation |
| D39 | 17:00 | Skip the segmentation branch and another pseudo-label round for now; next = v4ap into the stack, wide width 3 vs 5, code repo | Expected gains of morphology branch small vs time; PL confidence audit done (D25) | — |
| D40 | 16:55 | Code in private repo github.com/Niranjan-cyber/signal-hunters-r2; flip to public just before the deadline, then submit the link from the Submission tab | Rules require a code link before close and reproducibility for prize winners; private until then so other teams can't see it | Public now; notebook link |
| D41 | 17:25 | U4 = smx4 (stacker on CNN v1 + v4p + v4a + v4ap, level features, no wide filter) | vs U3: held-out ΔLL −0.0113 (7.5σ), ΔF1 +0.0043; SIM ΔLL −0.019 (11σ), ΔF1 +0.0063; clean ΔLL −0.003 | smx4_w (wide adds noise-level change; family closed after 3 sub-bar runs) |
| D42 | 17:45 | Reject EM label-shift correction (Saerens) | Estimated test priors close to current predictions (rfi_int 0.197, rfi_pers 0.132, scint 0.073); on held-out sim resampled to that mix: paired ΔF1 −0.0009 ± 0.0013; only 36 test rows change. Class-mix shift is not the source of the public gap | Damped EM (0.5): also no gain |
| D43 | 17:55 | CNN v6 = v4ap recipe + 4th input channel with the pre-whitening panel level (+ same SNR scaling and 0.34 z gain jitter in training) + pseudo-labels from U4 (8,902 rows) | Scint is the largest residual error (held-out 0.76) and its strongest cue is erased from CNN input by whitening; train/test level spread equal (p1/p99 −2.33/2.54 vs −2.38/2.44). Smoke: no NaN, 43 s/epoch; plain path reproduces epoch-0 numbers exactly | ON-only CNN, gating meta-model, VAE (lower expected gain) |
| D44 | 18:55 | Upload U5 (smx5) instead of U4 (U4 never uploaded) | U5 contains U4 plus v6 and beats it on every validation (held-out ΔLL −0.0040, 3.3σ); public result 0.9160 (+0.0073 vs U3) | Upload U4 then U5 (wastes a slot) |
| D45 | 19:05 | Synthetic ETI positives + hard negatives (disconnected / zero-drift / also-in-OFF) in CNN training (v7, syn 0.1) | ETI is the only class with recoverable headroom; injected tracks score like real ETI on the physics detector (track 5.96 vs 6.75, slope mismatch 0.05 vs 0.04, drift median 0.18 vs 0.18 after switching to log-uniform drift) | More pseudo-label rounds first |
| D46 | 19:15 | Add teammate (Swarali) as compute lanes, not upload lanes; private repo collaborator `swaralipatil11` | Upload quota is per team; our Modal runs only ~4 GPUs at once, so a second workspace doubles throughput | — |
| D47 | 19:25 | Every lane tests a different hypothesis (AdaBN, other GBDT families, pretrained backbone, DANN, drift-space CNN, LR stacker); v6 extra seeds dropped | User direction; seeds of v6 matched seed 0 within noise (fold 0 SIM 0.910 / 0.910 / 0.907) | More seeds of v6 / v7 |
| D48 | 19:30 | Two Claude sessions coordinate through the repo: `CLAUDE.md` protocol, `coord/BOARD.md` (claim before run), `coord/MESSAGES.md`, owner prefixes n_ / s_ | Sessions on different machines can only share state through git; claim-then-push makes duplicate work fail at push time | Chat relay through the users |
| D49 | 19:30 | Modal runs only via `src/modal_submit.py` (deployed app + spawned calls) | Session restart killed both the laptop v7 run (epoch 7) and the Modal launchers, and with them the containers (v7 2/5, seeds 1/5 each survived); spawned calls on a deployed app run independent of the client | `modal run --detach` (keeps only the last call alive) |
| D50 | 19:30 | Measure L4 vs A100 seconds/epoch before buying faster GPUs | Model is small; observed bottlenecks are Modal concurrency (~4 GPUs) and launcher deaths, not per-epoch speed | Assume a faster GPU helps |
| D51 | 20:20 | Reject LR stacker blend and AdaBN | LR blend: held-out F1 0.9347 → 0.9331 (LL slightly better); AdaBN on v6: held-out F1 worse on 3 of 4 folds | — |
| D52 | 20:20 | All Modal runs on A100 | A100 12 s/epoch vs L4 41 s/epoch (3.4x), cheaper per fold, identical metrics | L4 default |
| D53 | 22:35 | Hold U7 (smx8a + RegNetY + MobileNetV3) | held-out ΔF1 +0.0011 (1.4σ), ΔLL n.s.; with exact 1:1 held-out→public calibration it would score ~0.9243, below the leader and inside noise | Upload for the expiring slot |
| D54 | 22:35 | Treat held-out sim as an exact public predictor (public = held-out − 0.019) | U5 and U6 both sit 0.0187 below held-out; U5→U6 gain identical (+0.0072) | — |

## End-of-round checklist (deadline 28 Oct 10:00 IST per platform; 5 uploads on 28 Oct)
- [ ] 09:00 final candidate rebuilt from repo code, byte-identical CSV check
- [ ] 09:15 final uploads done (best overall + non-PL hedge)
- [ ] 09:30 push final code, flip repo to public (`gh repo edit Niranjan-cyber/signal-hunters-r2 --visibility public --accept-visibility-change-consequences`)
- [ ] 09:35 submit repo link from the Submission tab; confirm it opens logged-out
- [ ] 09:40 SUBMISSIONS.md final: selected upload, md5, public score

## Keep / kill rules in force
- Keep bar (in-distribution changes): paired ΔLL < −2x paired sd (about −0.007 full OOF, −0.015 fold 0). F1 only as a sanity check.
- Keep bar (shift levers): fold-0 PKS-sim ΔLL < −0.03 with clean LL no worse than +0.015; then confirm on the leaderboard.
- Close a model family after 3 runs below the bar.
- Pseudo-label CNN: stop if fold-0 val log-loss at epoch 15 is > 0.10 above plain v4 at epoch 15 (not triggered).
- Never upload a file whose CV is worse than the best uploaded (best public counts on private).

## Learnings (for the retrospective)
1. **Check test predictions before anything else.** The starter scored 0.52 OOF but put 73% of test into one class. Same lesson as Planet Hunters R1, caught here before an upload.
2. **Signal-preservation checks catch real bugs.** The two-pass bandpass came from looking at before/after plots: the first version turned a bright RFI line dark.
3. **Recon facts become levers.** Glitch rate (D7), pixel-unit drift (D10) and the centre spike (D4) were all measured in the first hour.
4. **The CNN and the physics features fail differently.** True-class probability correlation 0.42; the CNN wins everywhere except ETI, where the explicit track test wins. Stacking turned that into +0.06 F1.
5. **Read the trap descriptions literally.** The "same drift but does not connect" trap needed a feature that checks global slope against local drift; it took ETI F1 from 0.76 to 0.83.
6. **Kaggle T4s have no native bf16.** Training was 14x slower until switched to fp16. Smoke-test speed on the target machine before pushing a long run (the playbook says so; skipped here, cost about 1 h of lanes).
7. **Detached watchers are fragile on Windows.** The stop-after-fold-2 watcher didn't fire and the local job nearly overwrote merged Kaggle folds. Fix next time: give each lane its own output file name and merge explicitly.
8. **RAM stayed at 1–3 GB free all day.** A background task was reaped once. Heavy jobs were detached from the start, which is what kept them alive.
9. **CV → public gap of 0.05 on a cross-telescope test.** Budget for shift work early when train and test come from different instruments.
10. **Measure signal strength by class on train vs confident test predictions.** It turned "domain shift" into one number (0.74x SNR) that a simulator can reproduce; a PKS-simulated validation fold then reproduced the leaderboard gap. Do this at recon next time.
11. **Model-agreement filters for pseudo-labels can backfire** when families have complementary blind spots; check agreement per class before filtering.
12. **On Windows, Python's default file encoding is cp1252.** A rewrite with `open(..., 'w')` truncated this file and then failed on a non-ASCII character. Always pass `encoding='utf-8'`, or write to a temp file and rename.
13. **Test cheap corrections before uploading them.** EM label-shift correction looked principled (the PS says the label mix differs) but gave −0.0009 on held-out; it would have wasted a slot.
14. **Whitening can erase a class's best cue.** Scintillation lifts whole ON panels; per-panel median removal hid it from every model until the panel level was fed back as features (lv) and as a CNN channel (v6).
15. **Strong single models saturate inside a stack.** v6 cut CNN held-out log-loss by ~0.04 but the stack gained only −0.004: the stack already had the same information through other members. Judge members by stack gain, not standalone score.
16. **Launchers are part of the experiment.** Four runs were lost today to dead local processes (memory reaps, session restart, detached-mode semantics). Use cloud-side spawning and check the job, not the launcher.
17. **RAM was the binding constraint all day** (0.4–3 GB free on a 16 GB laptop, with chat apps and browsers open). Close them before an event day.
19. **A reaped shell can leave its Python child running.** The killed timm smoke test kept 3.2 GB until found and stopped; after any reap, list python processes.
20. **Measure GPUs instead of guessing.** A 2-epoch L4 vs A100 test (cents) showed the A100 is 3.4x faster and cheaper per fold.
18. **Parallel people need a protocol, not just access.** A shared board with claim-then-push plus a message log lets two agent sessions split hypotheses without duplicating work.
