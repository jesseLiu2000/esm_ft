#!/bin/bash

eval_file=("new_cut" "price_cut" "multi_cut" "halogenase_cut")
model_path="/scratch0/zx22/zijie/esm/results_70/esm1b/650mlora_2024-02-27_23-12-28"
extract_ids=('15520' '31040' '46560' '62080' '77600' '93120' '108640' '124160' '139680' '155200')

for i in "${extract_ids[@]}"
do
  for j in "${eval_file[@]}"
  do
     nohup python eval.py $i $j $model_path > eval_results_70/esm1/650mlora/result_${i}_${j}.txt 2>&1
  done
done


