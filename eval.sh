#!/bin/bash

eval_file=("new_cut" "price_cut" "multi_cut" "halogenase_cut")
model_path="/scratch0/zx22/zijie/esm/results_70/esm1b/650m_2024-02-27_23-12-28"
extract_ids=('11640' '23280' '34920' '46560' '58200' '69840' '81480' '93120' '104760' '116400')

for i in "${extract_ids[@]}"
do
  for j in "${eval_file[@]}"
  do
     nohup python eval.py $i $j $model_path > eval_results_70/esm1/650m_pb/result_${i}_${j}.txt 2>&1
  done
done
