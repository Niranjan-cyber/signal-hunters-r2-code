# U8 final candidates: U7 + EffNetV2-S + ConvNeXt-Tiny+MixUp (+ ConvNeXt-Small).
import subprocess, sys
base = 'cnn_v1,cnn_v4p,cnn_v4a,cnn_v4ap,cnn_v6,cnn_v7,n_pretrained,n_cnxt,s_regy,s_mbnv3,n_cnxtt,n_effb2'
for name, cnns in [('smx14_a', base + ',s_effv2s,n_cnxtt_mix'), ('smx14_b', base + ',s_effv2s,n_cnxtt_mix,s_cnxts')]:
    print('>>', name, flush=True)
    if subprocess.run(f'python src/stack_mix.py {name} mix {cnns} lv', shell=True).returncode: sys.exit(f'FAILED: {name}')
print('U8D BUILD DONE', flush=True)
