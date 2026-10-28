# Seed-bagged stack runner (detached-safe). usage: powershell -File src/run_bag.ps1 <name> <members> <seeds e.g. "0,1,2">
# seed 0 = LightGBM default (no 'seed' arg); outputs oof/<name>_s<N>; average later with src/bag_stack.py
param([string]$name, [string]$members, [string]$seeds)
Set-Location 'C:\Career\Hackathons\Hack4Hack R2\signal-hunters-round2'
foreach ($s in $seeds.Split(',')) {
    if ($s -eq '0') { python src/stack_mix.py $name mix $members lv 2>&1 | Select-String -NotMatch 'PerformanceWarning|frame.insert|^\s*$|1e-6' }
    else { python src/stack_mix.py "${name}_s$s" mix $members "lv,seed$s" 2>&1 | Select-String -NotMatch 'PerformanceWarning|frame.insert|^\s*$|1e-6' }
}
"BAGDONE $name $seeds"
