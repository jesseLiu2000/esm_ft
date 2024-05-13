#!/bin/bash

eval_file=("new_cut" "price_cut" "multi_cut" "halogenase_cut")
model_path="/scratch0/zx22/zijie/esm_ft/results_50/esm1b/650mlora_2024-04-16_13-43-00"
extract_ids=('24525' '49050' '73575' '98100')

for i in "${extract_ids[@]}"
do
  for j in "${eval_file[@]}"
  do
     nohup python eval.py $i $j $model_path > eval_results_50/esm1/650m_lora/result_${i}_${j}.txt 2>&1
  done
done
