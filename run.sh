export HF_HOME=/scratch0/zx22/zijie/cache
# nohup python esm_pb.py > log/esm_25_13.pb 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u esm_lora.py > log/esm_lora_13.pb 2>&1 &
# nohup python esm_lora.py > esm_10.lora 2>&1 &
