#!/usr/bin/env bash

for sym in 0 1 2 3 4
do
    python model_run.py --setting 'multi' --base 'TAOD-Net' --mode 0 --symptom $sym 
done




