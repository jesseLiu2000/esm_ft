#!/bin/bash

eval_file=("new_cut" "price_cut" "multi_cut" "halogenase_cut")
model_path="/scratch0/zx22/zijie/esm_ft/esm_full/results_70/esm2/650mlora_2024-05-13_01-32-14"
extract_ids=('12804')

for i in "${extract_ids[@]}"
do
  for j in "${eval_file[@]}"
  do
     nohup python eval.py $i $j $model_path > eval_results_70/esm2/650m_lora/result_${i}_${j}.txt 2>&1
  done
done
