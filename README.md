# Provable Benefits of Unsupervised Data Sharing in Offline RL

This is a jax implementation of PDS on [Datasets for Deep Data-Driven Reinforcement Learning (D4RL)](https://github.com/rail-berkeley/d4rl), the corresponding paper is [The provable benefits of unsupervised data sharing for offline reinforcement learning](https://proceedings.mlr.press/v162/hu22d.html).

## Quick Start
For experiments on D4RL, our code is implemented based on IQL:

```shell
$ python3 train_data_sharing.py --env_name=walker2d-expert-v2 --source_name=walker2d-random-v2 --config=configs/mujoco_config.py --data_share=learn  --target_split=0.05  --source_split=0.1
```


## Citing
If you find this open source release useful, please reference in your paper (it is our honor):
```
@article{hu2023provable,
  title={The provable benefits of unsupervised data sharing for offline reinforcement learning},
  author={Hu, Hao and Yang, Yiqin and Zhao, Qianchuan and Zhang, Chongjie},
  journal={arXiv preprint arXiv:2302.13493},
  year={2023}
}
```

## Note
+ If you have any questions, please contact me: yangyiqi19@mails.tsinghua.edu.cn. 
