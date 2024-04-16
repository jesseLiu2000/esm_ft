export HF_HOME=/scratch0/zx22/zijie/cache
CUDA_VISIBLE_DEVICES=0,1,3 nohup python -u esm_650m_pb.py > log/esm_650m.pb 2>&1 &
