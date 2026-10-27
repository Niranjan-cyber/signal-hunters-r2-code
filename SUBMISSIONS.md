# Submissions ledger (final = best public upload)

Upload quota (confirmed by Niranjan 22:10): **10 per day, reset at midnight IST**. 27 Oct used: U1, U2, U3, U5, U6 → 5 left.

| # | File | md5[:10] | Model | OOF F1 / LL | Uploaded (IST) | Public F1 | CV→LB gap |
|---|---|---|---|---|---|---|---|
| U1 | subs/U1_stk_v1.csv | 5f3d1d3f46 | LGB stacker: phys2f + log CNN v1 (5-fold), no multipliers | 0.9338 / 0.1721 | ~13:00 | 0.8823 (3rd; top 0.8858) | −0.0515 |
| U2 | subs/U2_smx_v1p_mix.csv | 7b499c5dd2 | mix stacker (clean+PKS-sim rows): phys2f/phys2sf + CNN v1 + CNN v4p (pseudo-labels) | clean 0.9430 / 0.1492; SIM 0.8999 / 0.2802 | 15:19 | **0.9017** (3rd; top 0.9120) | +0.0018 vs SIM; −0.041 vs clean |
| (C) | subs/C_smx_v1v4pv4a.csv | 39fbc86c70 | mix stacker v1+v4p+v4a (superseded by U3, not uploaded) | clean 0.9461 / 0.1449; SIM 0.9010 / 0.2633 | | | |
| U3 | subs/U3_smx3_lv.csv | d6cb9279ff | mix stacker v1+v4p+v4a + panel-level scint feats (lv) | clean 0.9529 / 0.1262; SIM 0.9164 / 0.2324 | 16:08 | **0.9087** (2nd; top 0.9120) | −0.0077 vs SIM |
| (U4) | subs/U4_smx4.csv (superseded by U5, not uploaded) | a7a5bdd607 | mix stacker v1+v4p+v4a+v4ap + lv | clean 0.9530 / 0.1230; SIM 0.9226 / 0.2132; HELD-OUT 0.9324 / 0.1834 | | | |
| U5 | subs/U5_smx5.csv | 4d5f4bc6a4 | mix stacker v1+v4p+v4a+v4ap+v6 + lv | clean 0.9538 / 0.1224; SIM 0.9249 / 0.2086; HELD-OUT 0.9347 / 0.1794 | ~18:55 | **0.9160** (2nd; top 0.9213) | −0.0089 vs SIM; −0.0187 vs held-out |
| (U6a) | subs/U6_smx6a.csv (superseded by U6_smx7a, not uploaded) | 536cdd5fb4 | U5 stack + cnn_v7 (synthetic ETI) | clean 0.9559 / 0.1209; SIM 0.9267 / 0.2053; HELD-OUT 0.9357 / 0.1783 | | | |
| (U6b) | subs/U6_smx7a.csv (superseded by U6_smx8a, not uploaded) | c73e154091 | U5 stack + cnn_v7 + n_pretrained (timm EfficientNet-B0, ImageNet) | clean 0.9580 / 0.1141; SIM 0.9299 / 0.1924; HELD-OUT 0.9390 / 0.1680 | | | |
| U6 | subs/U6_smx8a.csv | b66fb815b8 | U5 stack + v7 + EfficientNet-B0 + ConvNeXt-Nano (ImageNet-pretrained) | clean 0.9612 / 0.1080; SIM 0.9345 / 0.1840; HELD-OUT 0.9419 / 0.1583 | 21:47 (by Swarali) | **0.9232** (2nd; top PatrickXDEXTER 0.9254) | −0.0187 vs held-out (same as U5) |
