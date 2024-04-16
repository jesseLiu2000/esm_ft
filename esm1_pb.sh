export HF_HOME=/scratch0/zx22/zijie/cache
CUDA_VISIBLE_DEVICES=6,7 nohup python -u esm1_pb.py > log/esm1_80m.pb 2>&1 &

