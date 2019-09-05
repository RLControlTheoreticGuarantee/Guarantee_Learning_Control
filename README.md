# Guarantee_Learning_Control
Model Free Reinforcement Learning with Control Theoretic Guarantee



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

### Example 1. LPPO with MuJoCo Point-Circle
For instance, to train a fully-connected network controlling MuJoCo Point-Circle using LPPO for 2M timesteps
```bash
python run.py
```

The hyperparameters, the tasks and the learning algorithm can be change via change the run.py, for example
```bash
 alg = 'ppo2_lyapunov'
 # alg='ppo2'
 additional_description ='-clip-0.8'
 env = 'Pointcircle-v0'
# env = 'Antcons-v0'
# env = 'HalfCheetahcons-v0'
# env = 'Quadrotorcons-v0'
# env = 'PongNoFrameskip-v5'
# env = 'Point-v1'
info = ['--num_timesteps=5e6', '--save_path=./Model/'+env]
```
### Example 2. LAC with continous cartpole
```
python main_for_sac.py
```
The hyperparameters, the tasks and the learning algorithm can be change via change the variant.py, for example
```bash
VARIANT = {
    'env_name': 'Carcost-v0',
    'algorithm_name': 'LAC',
    'additional_description': '-continuous-25',
    'evaluate': False,
    'train':True,
    'evaluation_frequency': 2048,
    'num_of_paths': 1,
    'num_of_trials': 10,
    'store_last_n_paths': 5,
    'start_of_trial': 0,
}
```

