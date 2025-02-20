export PYTHONPATH=/workspace/motion_diffusion/MotionCraft:$PYTHONPATH && \
python mogen/datasets/EMAGE_2024/train_mix.py --config=./mogen/datasets/EMAGE_2024/configs_mix/cnn_vqvae_upper_30_mix.yaml

# export PYTHONPATH=/workspace/motion_diffusion/MotionCraft:$PYTHONPATH && \
# export CUDA_VISIBLE_DEVICES=0 && \
# python mogen/datasets/EMAGE_2024/train_mix.py --config=./mogen/datasets/EMAGE_2024/configs_mix/cnn_vqvae_lower_foot_30_mix_ema_code512.yaml