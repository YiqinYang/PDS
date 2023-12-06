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

#source_envs=(
#  "halfcheetah-random-v2"
#  "hopper-random-v2"
#  "walker2d-random-v2"
#  "halfcheetah-random-v2"
#  "hopper-random-v2"
#  "walker2d-random-v2"
#)

gpus=(0 1 2 3 4 5 6 7)
strategies=("none" "learn" "pess" "all" "zero")
for ((i = 0; i < 6; i += 1)); do
  for ((j = 0; j < 5; j += 1)); do
    env=${envs[i]}
    source_env=${envs[i]}
    strategy=${strategies[j]}
    CUDA_VISIBLE_DEVICES=${gpus[i]} nohup python train_data_sharing.py \
      --env_name "$env" \
      --source_name "$source_env" \
      --config=configs/mujoco_config.py \
      --comment=split_data \
      --source_split=0.9 --target_split=0.1 \
      --data_share "${strategy}" >nohup_logs/data_sharing_"${strategy}"_"${env}"_"${source_env}".out &
  done

  sleep 1
done
