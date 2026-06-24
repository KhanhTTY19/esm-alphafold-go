#!/bin/bash
set -e  # dừng ngay nếu có lệnh nào lỗi

echo "========== START TRAINING =========="

python train_Struct2GO2.py -branch bp -labels_num 610 -learningrate 0.0001 -dropout 0.1 -batch_size 128 -gpu 1 -exp_name v1_lr1e4_do01 -patience 3 -es_monitor fscore

python train_Struct2GO2.py -branch cc -labels_num 310 -learningrate 0.0001 -dropout 0.1 -batch_size 128 -gpu 1 -exp_name v1_lr1e4_do01 -patience 3 -es_monitor fscore

echo "========== START TESTING =========="

python eval_Struct2GO2.py -branch mf -exp_name v4_lr1e4_do01 -gpu 1 -batch 1 -thresh 0.01

python eval_Struct2GO2.py -branch bp -exp_name v1_lr1e4_do01 -gpu 1 -batch 1 -thresh 0.01

python eval_Struct2GO2.py -branch cc -exp_name v1_lr1e4_do01 -gpu 1 -batch 1 -thresh 0.01

echo "========== GIT PUSH =========="

git add .
git commit -m "add result testing"
git push

echo "========== DONE =========="