# U8 test: U7 (smx11_all) + Swarali's ConvNeXt-Small / EfficientNetV2-S.
import subprocess, sys
base = 'cnn_v1,cnn_v4p,cnn_v4a,cnn_v4ap,cnn_v6,cnn_v7,n_pretrained,n_cnxt,s_regy,s_mbnv3,n_cnxtt,n_effb2'
for name, cnns in [('smx13_v2s', base + ',s_effv2s'), ('smx13_both', base + ',s_effv2s,s_cnxts')]:
    print('>>', name, flush=True)
    if subprocess.run(f'python src/stack_mix.py {name} mix {cnns} lv', shell=True).returncode: sys.exit(f'FAILED: {name}')
print('U8 BUILD DONE', flush=True)
