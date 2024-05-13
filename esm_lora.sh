export HF_HOME=/scratch0/zx22/zijie/cache
CUDA_VISIBLE_DEVICES=0,1 nohup python -u esm_lora.py > log/esm2_650m.lora 2>&1 &

