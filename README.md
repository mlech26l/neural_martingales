# Learning Control Policies for Region Stabilization in Stochastic Systems

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


## Learning a sRASM

```bash
python3 rsm_loop.py --skip_ppo --env lds_100 --p_lip 4.0 --grid_factor 4 --batch_size 4*2048 --stability_check
python3 rsm_loop.py --skip_ppo --env pend_100 --p_lip 4.0 --grid_factor 4 --batch_size 4*2048 --stability_check --epsilon_as_tau --eps 0.003
```

To learn a sRASM with fixing epsilon for the loss function, run 

```bash
python3 rsm_loop.py --skip_ppo --env lds_100 --p_lip 4.0 --grid_factor 4 --batch_size 4*2048 --stability_check
```

To learn a sRASM and use (K * tau) instead of epsilon in the loss function (L'_{cond 2} in the Supplementary Material), run 

```bash
python3 rsm_loop.py --skip_ppo --env lds_100 --p_lip 4.0 --grid_factor 4 --batch_size 4*2048 --stability_check --epsilon_as_tau --eps 0.0007
```

In the above for the parameter --eps, you should pass the mesh value that you use in the discretization.

## Note on the obtained bounds

After the conditions are checked by the verifier module, the sRASM network is normalized such that the inf. of V on the entire domain is 0. 
This normalization allows us to obtain even slightly better bounds than the verifier concluded.
