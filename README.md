# Safe Model-free Reinforcement Learning with Stability Guarantee
Safe Model-free Reinforcement Learning with Stability Guarantee



## Conda environment
From the general python package sanity perspective, it is a good idea to use conda environments to make sure packages from different projects do not interfere with each other.


To create a conda env with python3, one runs 
```bash
conda create -n test python=3.6
```
To activate the env: 
```
conda activate test
```

### MuJoCo
Some of the baselines examples use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py)

# Installation Environment

```bash
git clone https://github.com/RLControlTheoreticGuarantee/Guarantee_Learning_Control
pip install numpy==1.16.3
pip install tensorflow==1.13.1
pip install tensorflow-probability==0.6.0
pip install opencv-python
pip install cloudpickle
pip install gym
pip install gym[atari]
pip3 install -U 'mujoco-py==1.50.1.68'
pip install matplotlib

```

### Example 1. LSAC with CartPole
For instance, to train a fully-connected network controlling MuJoCo Point-Circle using LPPO for 2M timesteps
```bash
python main_for_sac.py
```


The hyperparameters, the tasks and the learning algorithm can be change via change the variant.py, for example
```bash
VARIANT = {
    'env_name': 'Pointcircle-v0',
    # 'env_name': 'CartPolecons-v0',
    # 'env_name': 'HalfCheetahcons-v0',
    'algorithm_name': 'LSAC',
    'additional_description': '-new-init-clip-80',
    'disturber': 'SAC',
    'evaluate': False,
    'train':True,
    'evaluation_frequency': 2048,
    'num_of_paths': 1,
    'num_of_trials': 5,
    'store_last_n_paths': 10,
    'start_of_trial': 2,

    # Evaluation Settings

    # 'evaluation_form': 'impulse',
    'evaluation_form': 'safety_eval',
    # 'evaluation_form': 'param_variation',
    # 'evaluation_form': 'trained_disturber',
    'eval_list': [

        # 'MPC',
        # 'LQR',
        # 'LSAC-clip-20',
        'LSAC-new-init-clip-80',
        'LSAC-new-init-clip-50',
        'SSAC-new-init',
        'SAC-new-init',
        'LSAC-new-init',
        'SSAC',
        # 'SSAC',
        # 'SAC',
        # 'LAC',
        # 'RLAC',
    ],
    'trials_for_eval': [str(i) for i in range(0, 10)],
}
```

