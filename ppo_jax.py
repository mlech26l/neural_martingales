import os
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import jax
import numpy as np
import tensorflow as tf

import jax.numpy as jnp
from tqdm.auto import tqdm

from klax import lipschitz_l1_jax, pretty_time


class ExperienceBuffer:
    def __init__(self):
        self.action_sampled = []
        self.states = []
        self.returns = []
        self.action_log_prob = []
        self.advantage = None

    def close_episode(self, returns):
        self.returns.extend(returns)

    def append(self, state, sampled_action, log_prob):
        self.states.append(state)
        self.action_sampled.append(sampled_action)
        self.action_log_prob.append(log_prob)

    def clear(self):
        self.action_sampled = []
        self.states = []
        self.returns = []
        self.action_log_prob = []

    def get_states(self):
        return np.stack(self.states)

    def set_advantage(self, values):
        returns = np.stack(self.returns)
        advantage = returns - np.array(values)
        mean, std = np.mean(advantage), np.std(advantage)
        advantage = (advantage - mean) / (std + 1e-6)
        self.advantage = advantage

    def get_value_train_iter(self, batch_size=128):
        states = np.stack(self.states, axis=0)
        returns = np.stack(self.returns).reshape((-1, 1))
        train_ds = tf.data.Dataset.from_tensor_slices((states, returns))
        train_ds = train_ds.shuffle(4096).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return train_ds

    def get_policy_train_iter(self, batch_size=128):
        states = np.stack(self.states, axis=0)
        action_sampled = np.stack(self.action_sampled)
        action_log_prob = np.stack(self.action_log_prob).flatten()
        train_ds = tf.data.Dataset.from_tensor_slices(
            (states, action_sampled, action_log_prob, self.advantage)
        )
        train_ds = train_ds.shuffle(4096).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return train_ds


@jax.jit
def train_step_value(value_s, states, returns):
    def value_loss_fn(v_params):
        value = value_s.apply_fn(v_params, states)
        loss = jnp.mean(jnp.square(value - returns))
        return loss, value

    v_grad_fn = jax.value_and_grad(value_loss_fn, has_aux=True)
    (v_loss, values), grads = v_grad_fn(value_s.params)
    value_s = value_s.apply_gradients(grads=grads)
    metrics = {"v_loss": v_loss}
    return value_s, metrics


def gauss_log_prob(mean, std, x):
    return -1 / (2 * jnp.square(std)) * jnp.sum(jnp.square(mean - x), axis=-1)


def np_gauss_log_prob(mean, std, x):
    return -1 / (2 * np.square(std)) * np.sum(np.square(mean - x), axis=-1)


@jax.jit
def train_step_policy(
    policy_s,
    states,
    sampled_actions,
    action_logprobs,
    advantage,
    action_std,
    lip,
    max_lip,
):
    clip_ratio = 0.2

    def policy_loss_fn(p_params):
        mean = policy_s.apply_fn(p_params, states)
        log_prob = gauss_log_prob(mean, action_std, sampled_actions)
        ratio = jnp.exp(log_prob - action_logprobs)
        surr1 = ratio * advantage
        surr2 = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
        loss = -jnp.mean(jnp.minimum(surr1, surr2))
        lip_loss = jnp.maximum(lipschitz_l1_jax(p_params), max_lip)
        loss += lip * lip_loss

        return loss

    p_grad_fn = jax.value_and_grad(policy_loss_fn)
    p_loss, grads = p_grad_fn(policy_s.params)
    policy_s = policy_s.apply_gradients(grads=grads)
    metrics = {"p_loss": p_loss}

    return policy_s, metrics


class PPO:
    def __init__(self, p_state, v_state, env, max_lip):
        self.env = env
        self.action_dim = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]
        self.p_state = p_state
        self.v_state = v_state
        self.max_lip = jnp.float32(max_lip)

        self.gamma = 0.99
        self.lip_factor = 0
        self.action_std = 0.2
        self.i = 0
        self.buffer = ExperienceBuffer()

    def clear_history(self):
        self.buffer.clear()

    def sample_rollout(self):
        # state = self.env.observation_space.sample()
        # self.env.reset(state)
        state = self.env.reset()
        rewards_history = []
        done = False
        while not done:
            action_mean = np.array(self.p_state.apply_fn(self.p_state.params, state))
            action = (
                np.random.default_rng().normal(size=self.action_dim) * self.action_std
                + action_mean
            )
            log_prob = np_gauss_log_prob(action_mean, self.action_std, action)
            self.buffer.append(
                state,
                action,
                log_prob,
            )
            next_state, reward, done, _ = self.env.step(action)
            rewards_history.append(reward)
            state = next_state
        episode_return = np.sum(rewards_history)
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + self.gamma * discounted_sum
            returns.append(discounted_sum)
        returns.reverse()
        self.buffer.close_episode(returns)

        return episode_return

    def run(self, num_iters, std_start, std_end, lip_start, lip_end):
        stds = jnp.linspace(std_start, std_end, num_iters)
        lip = jnp.linspace(lip_start, lip_end, num_iters)
        for i in range(num_iters):
            self.action_std = stds[i]
            self.lip_factor = lip[i]
            r = self.run_iter()

    def run_iter(self):
        start_time = time.time()
        rs = [self.sample_rollout() for i in range(30)]
        p_lip = lipschitz_l1_jax(self.p_state.params)
        print(
            f"Iter {self.i} R={np.mean(rs):0.2f} [{np.min(rs):0.2f}, {np.max(rs):0.2f}] with lip = {p_lip:0.3f} (rollouts took {pretty_time(time.time()-start_time)})",
            flush=True,
        )
        self.train_epoch()
        self.clear_history()
        self.i += 1
        return np.mean(rs)

    def get_mean_return(self):
        rs = [self.sample_rollout() for i in range(10)]
        return np.mean(rs)

    def train_epoch(self):
        states = self.buffer.get_states()
        values = self.v_state.apply_fn(self.v_state.params, states).flatten()

        self.buffer.set_advantage(values)

        policy_ds = self.buffer.get_policy_train_iter()
        policy_epochs = 10
        pbar = tqdm(total=policy_epochs * len(policy_ds))
        for p_epoch in range(policy_epochs):
            for batch in policy_ds.as_numpy_iterator():
                action_std = jnp.float32(self.action_std)
                lip = jnp.float32(self.lip_factor)
                self.p_state, metrics = train_step_policy(
                    self.p_state,
                    batch[0],
                    batch[1],
                    batch[2],
                    batch[3],
                    action_std,
                    lip,
                    self.max_lip,
                )
                pbar.set_description_str(f"policy_loss={metrics['p_loss']:0.3f}")
                pbar.update(1)
        pbar.close()

        value_ds = self.buffer.get_value_train_iter()
        value_epochs = 5
        pbar = tqdm(total=value_epochs * len(value_ds))
        for v_epoch in range(value_epochs):
            for batch in value_ds.as_numpy_iterator():
                self.v_state, metrics = train_step_value(
                    self.v_state, batch[0], batch[1]
                )
                pbar.set_description_str(f"value_loss={metrics['v_loss']:0.3f}")
                pbar.update(1)
        pbar.close()
