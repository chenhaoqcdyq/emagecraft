#!/bin/bash
#SBATCH -J test              # 作业名为test
#SBATCH -p zjhu             # 提交到gpu-1分区
#SBATCH -N 1                 # 使用1个节点
#SBATCH --nodelist=node[38]
#SBATCH --cpus-per-task=4    # 每个进程占用8个cpu核心，申请的内存与cpu核心数成正比
#SBATCH --gres=gpu:2  # 申请1块GPU卡
#SBATCH -t 7-00:00:00        # 最长运行时间为1天
#SBATCH -o run_rvqvae_upper_mix_code512_ddp.out          

path="/public/home/qianyijie2023/workspace/emagecraft"
# 查看显卡信息
srun nvidia-smi
# 切换工作目录，并执行显卡测试程序
cd ${path}
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
#conda install pytorch3d -c pytorch3d
# conda init && source ~/.bashrc && conda activate motionv2
# bash tools/dist_train.sh ./configs/t2m_gpt/T2M_GPT_body_vqvae_512.py  \
#     ./work_dirs/T2M_GPT_body_vqvae_512 2
# python ./scripts/EMAGE_2024/train.py --config=./scripts/EMAGE_2024/configs/transformer_vqvae_upper_30.yaml
export PYTHONPATH=${path}:$PYTHONPATH && \
torchrun  --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=29500  mogen/datasets/EMAGE_2024/train_mix.py --config=mogen/datasets/EMAGE_2024/configs_mix/cnn_rvqvae_upper_30_mix_code512_ddp.yaml
