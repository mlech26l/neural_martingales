# Supplementary code for the paper Learning Control Policies for Stochastic Systems with Reach-avoid Guarantees

## Requirements

Python 3.8 or newer. 
For the installation of JAX with GPU support see [here](https://github.com/google/jax)

```bash
pip3 install flax optax gym numpy tqdm tensorflow 
```

Note that Tensorflow is only used for the ```tf.data``` API.

## Pre-training policies with PPO

To train a policy network on the 2D system task with a Lipschitz threshold fo 4.0 for 100 PPO iterations run:

```bash
python3 rsm_loop.py --env lds_100 --p_lip 4.0 --only_ppo --ppo_iters 100
```

The policy is then saved in ```checkpoints/lds_100_ppo.jax```

## Available pre-trained policies

The ```checkpoints``` directory contains the pre-trained policies used in the experiments

- ```lds_100_ppo.jax```
- ```pend_100_ppo.jax```
- ```cavoid_100_ppo.jax```
- ```lds_20_ppo.jax```
- ```pend_20_ppo.jax```
- ```cavoid_20_ppo.jax```


## Learning a RASM

```bash
python3 rsm_loop.py --env lds_100 --skip_ppo --p_lip 4.0 --grid_factor 8 --batch_size 2048 --reach_prob 0.8
python3 rsm_loop.py --env cavoid_20_ppo --skip_ppo --p_lip 4.0 --grid_factor 8 --batch_size 2048 --reach_prob 0.9
python3 rsm_loop.py --env pend_100 --skip_ppo --grid_factor 16 --batch_size 2048  --fail_check_fast 1 --jitter_grid 1 --reach_prob 0.8
```

To learn a RASM with only the decrease condition (=RSM), run 

```bash
python3 rsm_loop.py --env lds_100 --skip_ppo --p_lip 4.0 --grid_factor 8 --batch_size 2048 --reach_prob 1.0
```

To learn only the RASM network and fix the policy, run 

```bash
python3 rsm_loop.py --env lds_100 --skip_ppo --p_lip 4.0 --grid_factor 8 --batch_size 2048 --reach_prob 0.8 --train_p 0
```


## Note on the obtained bounds

After the conditions are checked by the verifier module, the RASM network is normalized such that the sup. of V at the initial set is 1 and the inf. of V on the entire domain is 0. 
This normalization allows us to obtain even slightly better bounds than the verifier concluded (see Appendix F in the paper).
