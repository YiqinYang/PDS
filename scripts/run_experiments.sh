#!/bin/bash

# Script to reproduce results
cd ..
envs=(
  "halfcheetah-medium-v2"
  "hopper-medium-v2"
  "walker2d-medium-v2"
  "halfcheetah-expert-v2"
  "hopper-expert-v2"
  "walker2d-expert-v2"
)


gpus=(0 2 4 5 6 7)
reward_type="max"
for ((i = 0; i < 6; i += 1)); do
  env=${envs[i]}
  CUDA_VISIBLE_DEVICES=${gpus[i]} nohup python train_with_random_reward.py \
    --env_name $env \
    --config=configs/mujoco_config.py \
    --comment random_reward_${reward_type}_scale=1 \
    --reward_type ${reward_type} \
    --seed 0 >nohup_logs/random_reward_${reward_type}_scale=1_"${env}".out &
  sleep 1
done

#for ((i = 0; i < 6; i += 1)); do
#  env=${envs[i]}
#  CUDA_VISIBLE_DEVICES=${gpus[i]} nohup python train_with_random_reward.py \
#    --env_name $env \
#    --config=configs/mujoco_config.py \
#    --comment "random_reward_max" \
#    --max_coe True \
#    --reward_type "random" \
#    --seed 0 >nohup_logs/random_reward_max_"${env}".out &
#  sleep 1
#done
