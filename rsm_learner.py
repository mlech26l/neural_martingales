import jax
import jax.numpy as jnp
from functools import partial

import tensorflow as tf

from rsm_utils import (
    jax_save,
    jax_load,
    lipschitz_l1_jax,
    martingale_loss,
    triangular,
    IBPMLP,
    MLP,
    create_train_state,
    clip_grad_norm,
)
import numpy as np

from ppo_jax import PPO
from vppo_jax import vPPO


class RSMLearner:
    def __init__(
        self,
        l_hidden,
        p_hidden,
        env,
        lip_lambda,
        p_lip,
        v_lip,
        eps,
        reach_prob,
        stability_check,
        epsilon_as_tau,
        alpha_max=1,
        alpha_min=1,
        small_delta=0.1
    ) -> None:
        self.env = env
        self.eps = jnp.float32(eps)
        self.reach_prob = jnp.float32(reach_prob)
        self.stability_check = stability_check
        self.epsilon_as_tau = epsilon_as_tau
        if alpha_max < alpha_min:
            alpha_max = alpha_min
        self.alpha_max = jnp.float32(alpha_max)
        self.alpha_min = jnp.float32(alpha_min)
        self.small_delta = jnp.float32(small_delta)
        action_dim = self.env.action_space.shape[0]
        obs_dim = self.env.observation_dim
        v_net = MLP(l_hidden + [1], activation="relu", softplus_output=True)
        c_net = MLP(l_hidden + [1], activation="relu", softplus_output=False)
        p_net = MLP(p_hidden + [action_dim], activation="relu")

        self.v_ibp = IBPMLP(l_hidden + [1], activation="relu", softplus_output=True)
        self.p_ibp = IBPMLP(
            p_hidden + [action_dim], activation="relu", softplus_output=False
        )
        self.v_state = create_train_state(
            v_net,
            jax.random.PRNGKey(1),
            obs_dim,
            0.0005,
        )
        self.c_state = create_train_state(
            c_net,
            jax.random.PRNGKey(3),
            obs_dim,
            0.0005,
        )
        self.p_state = create_train_state(
            p_net,
            jax.random.PRNGKey(2),
            obs_dim,
            0.00005,
        )
        self.p_lip = jnp.float32(p_lip)
        self.v_lip = jnp.float32(v_lip)
        self.lip_lambda = jnp.float32(lip_lambda)

        self.rng = jax.random.PRNGKey(777)
        self._debug_init = []
        self._debug_unsafe = []

    def pretrain_policy(
        self,
        num_iters=10,
        std_start=0.3,
        std_end=0.03,
        lip_start=0.0,
        lip_end=0.1,
        save_every=None,
    ):
        if self.env.is_paralyzed:
            ppo = vPPO(
                self.p_state,
                self.c_state,
                self.env,
                self.p_lip,
            )
        else:
            ppo = PPO(
                self.p_state,
                self.c_state,
                self.env,
                self.p_lip,
            )
        ppo.run(num_iters, std_start, std_end, lip_start, lip_end, save_every)

        # Copy from PPO
        self.p_state = ppo.p_state
        self.c_state = ppo.c_state

    def evaluate_rl_single(self):
        state = self.env.reset()
        total_reward = 0
        done = False
        while not done:
            action_mean = np.array(self.p_state.apply_fn(self.p_state.params, state))
            next_state, reward, done, _ = self.env.step(action_mean)
            total_reward += reward
            state = next_state
        return total_reward

    def evaluate_rl(self):
        if not self.env.is_paralyzed:
            n = 10
            rs = [self.evaluate_rl_single() for i in range(n)]
            text = f"Rollouts (n={n}): {np.mean(rs):0.1f} +- {np.std(rs):0.1f} [{np.min(rs):0.1f}, {np.max(rs):0.1f}]"
            print(text)
            return text
        else:
            n = 50
            rng = jax.random.PRNGKey(2)
            rng, r = jax.random.split(rng)
            r = jax.random.split(r, n)
            state, obs = self.env.v_reset(r)
            total_reward = jnp.zeros(n)
            done = jnp.zeros(n, dtype=jnp.bool_)
            while not np.any(done):
                action_mean = self.p_state.apply_fn(self.p_state.params, obs)
                rng, r = jax.random.split(rng)
                r = jax.random.split(r, n)
                state, obs, reward, next_done = self.env.v_step(state, action_mean, r)
                total_reward += reward * (1.0 - done)
                done = next_done
            text = f"Rollouts (n={n}): {np.mean(total_reward):0.1f} +- {np.std(total_reward):0.1f} [{np.min(total_reward):0.1f}, {np.max(total_reward):0.1f}]"
            print(text)
            return text

    @partial(jax.jit, static_argnums=(0, 2))
    def sample_init(self, rng, n):
        rngs = jax.random.split(rng, len(self.env.init_spaces))
        per_space_n = n // len(self.env.init_spaces)

        batch = []
        for i in range(len(self.env.init_spaces)):
            x = jax.random.uniform(
                rngs[i],
                (per_space_n, self.env.observation_dim),
                minval=self.env.init_spaces[i].low,
                maxval=self.env.init_spaces[i].high,
            )
            batch.append(x)
        return jnp.concatenate(batch, axis=0)

    @partial(jax.jit, static_argnums=(0, 2))
    def sample_unsafe(self, rng, n):
        rngs = jax.random.split(rng, len(self.env.unsafe_spaces))
        per_space_n = n // len(self.env.unsafe_spaces)

        batch = []
        for i in range(len(self.env.unsafe_spaces)):
            x = jax.random.uniform(
                rngs[i],
                (per_space_n, self.env.observation_dim),
                minval=self.env.unsafe_spaces[i].low,
                maxval=self.env.unsafe_spaces[i].high,
            )
            batch.append(x)
        return jnp.concatenate(batch, axis=0)

    @partial(jax.jit, static_argnums=(0, 2))
    def sample_unsafe_complement(self, rng, n):
        x = jax.random.uniform(
            rng,
            (2 * n, self.env.observation_dim),
            minval=self.env.observation_space.low,
            maxval=self.env.observation_space.high,
        )

        mask = jnp.zeros(2 * n, dtype=jnp.bool_)
        for unsafe_space in self.env.unsafe_spaces:
            b_low = jnp.expand_dims(unsafe_space.low, axis=0)
            b_high = jnp.expand_dims(unsafe_space.high, axis=0)
            contains = jnp.logical_and(
                jnp.all(x >= b_low, axis=1), jnp.all(x <= b_high, axis=1)
            )
            mask = jnp.logical_or(
                mask,
                contains,
            )

        mask = jnp.logical_not(mask)

        return x, mask


    @partial(jax.jit, static_argnums=(0, 2))
    def sample_target(self, rng, n):
        rngs = jax.random.split(rng, len(self.env.target_spaces))
        per_space_n = n // len(self.env.target_spaces)

        batch = []
        for i in range(len(self.env.target_spaces)):
            x = jax.random.uniform(
                rngs[i],
                (per_space_n, self.env.observation_dim),
                minval=self.env.target_spaces[i].low,
                maxval=self.env.target_spaces[i].high,
            )
            batch.append(x)
        return jnp.concatenate(batch, axis=0)

    @partial(jax.jit, static_argnums=(0, 2))
    def get_center_grid(self, n=50):
        dims = self.env.observation_dim
        grid, steps = [], []
        for i in range(dims):
            samples, step = jnp.linspace(
                -0.02,
                0.02,
                n,
                endpoint=False,
                retstep=True,
            )
            grid.append(samples)
            steps.append(step)
        grid = jnp.meshgrid(*grid)
        grid_lb = [x.flatten() for x in grid]
        grid_ub = [grid_lb[i] + steps[i] for i in range(dims)]
        grid_centers = [grid_lb[i] + steps[i] / 2 for i in range(dims)]

        grid_lb = jnp.stack(grid_lb, axis=1)
        grid_ub = jnp.stack(grid_ub, axis=1)
        grid_centers = jnp.stack(grid_centers, axis=1)
        return grid_centers, grid_lb, grid_ub

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, v_state, p_state, state, rng, current_delta, current_alpha, unsafe_lb_flag):
        """Train for a single step."""
        rngs = jax.random.split(rng, 6)
        init_samples = self.sample_init(rngs[1], 256)
        unsafe_samples = self.sample_unsafe(rngs[2], 256)
        target_samples = self.sample_target(rngs[3], 256)
        unsafe_complement_samples, unsafe_complement_mask = self.sample_unsafe_complement(rngs[5], 256)

        # center_grid, _, _ = self.get_center_grid()
        # Adds a bit of randomization to the grid
        s_random = jax.random.uniform(rngs[4], state.shape, minval=-0.5, maxval=0.5)
        state = state + current_delta * s_random

        def loss_fn(l_params, p_params):
            l = v_state.apply_fn(l_params, state)
            a = p_state.apply_fn(p_params, state)

            s_next = self.env.v_next(state, a)
            s_next = jnp.expand_dims(
                s_next, axis=1
            )  # broadcast dim 1 with random noise
            noise = triangular(rngs[0], (s_next.shape[0], 16, self.env.observation_dim))
            noise = noise * self.env.noise
            s_next_random = s_next + noise
            l_next_fn = jax.vmap(v_state.apply_fn, in_axes=(None, 0))
            l_next = l_next_fn(l_params, s_next_random)
            exp_l_next = jnp.mean(l_next, axis=1)

            violations = (exp_l_next >= l).astype(jnp.float32)
            violations = jnp.mean(violations)

            if self.epsilon_as_tau:
                K_f = self.env.lipschitz_constant
                K_l = lipschitz_l1_jax(l_params)
                K_p = lipschitz_l1_jax(p_params)

                lipschitz_k = K_l * K_f * (1 + K_p) + K_l

                dec_loss = martingale_loss(l, exp_l_next, jnp.float32(lipschitz_k * self.eps))
            else:
                dec_loss = martingale_loss(l, exp_l_next, self.eps)
            loss = dec_loss
            K_l = lipschitz_l1_jax(l_params)
            K_p = lipschitz_l1_jax(p_params)
            lip_loss_l = jnp.maximum(K_l - self.v_lip, 0)
            lip_loss_p = jnp.maximum(K_p - self.p_lip, 0)
            loss += self.lip_lambda * (lip_loss_l + lip_loss_p)

            if not self.stability_check:
                if float(self.reach_prob) < 1.0:
                    l_at_init = v_state.apply_fn(l_params, init_samples)
                    l_at_unsafe = v_state.apply_fn(l_params, unsafe_samples)
                    l_at_target = v_state.apply_fn(l_params, target_samples)
                    # l_at_center = v_state.apply_fn(l_params, center_grid)
                    # Train RA objectives

                    # Global minimum should be inside target
                    min_at_target = jnp.min(l_at_target)
                    min_at_init = jnp.min(l_at_init)
                    min_at_unsafe = jnp.min(l_at_unsafe)
                    loss += jnp.maximum(min_at_target - min_at_init, 0)
                    loss += jnp.maximum(min_at_target - min_at_unsafe, 0)

                    # Zero at zero
                    s_zero = jnp.zeros(self.env.observation_dim)
                    l_at_zero = v_state.apply_fn(l_params, s_zero)
                    loss += jnp.sum(
                        jnp.maximum(jnp.abs(l_at_zero), 0.3)
                    )  # min to an eps of 0.3

                    max_at_init = jnp.max(l_at_init)
                    min_at_unsafe = jnp.min(l_at_unsafe)
                    # Maximize this term to at least 1/(1-reach prob)
                    loss += -jnp.minimum(min_at_unsafe, 1 / (1 - self.reach_prob))

                    # Minimize the max at init to below 1
                    loss += jnp.maximum(max_at_init, 1)

            else:
                l_at_unsafe = v_state.apply_fn(l_params, unsafe_samples)
                l_at_target = v_state.apply_fn(l_params, target_samples)
                # l_at_center = v_state.apply_fn(l_params, center_grid)
                loss += -jnp.minimum(K_l, 0.01) * 10

                lip_f = self.env.lipschitz_constant
                lip_p = lipschitz_l1_jax(p_params)
                lip_norm = lip_f * (lip_p + 1) * current_delta
                t = p_state.apply_fn(p_params, unsafe_complement_samples)
                next_det_grids = self.env.v_next(unsafe_complement_samples, t)
                next_dis = jnp.sum(jnp.abs(next_det_grids - unsafe_complement_samples), axis=1)
                next_max_norm = jnp.max(jnp.where(unsafe_complement_mask, next_dis, 0))
                noise_norm = jnp.sum(jnp.abs(self.env.noise))
                current_big_delta = next_max_norm + noise_norm + lip_norm

                min_at_unsafe = jnp.min(l_at_unsafe)
                unsafe_lb = 1 + current_alpha * K_l * current_big_delta + self.small_delta
                # unsafe_lb = 1 * unsafe_lb_flag + current_alpha * K_l * current_big_delta + self.small_delta
                # unsafe_lb = 1 + current_alpha * K_l * current_big_delta + self.small_delta
                # unsafe_lb = 1 + current_alpha * prev_lip_l * current_big_delta + self.small_delta
                # loss += -jnp.minimum(min_at_unsafe - unsafe_lb, 0) / 2 * unsafe_lb_flag
                loss += -jnp.minimum(min_at_unsafe - unsafe_lb, 0)

                l_at_unsafe_comp = v_state.apply_fn(l_params, unsafe_complement_samples)
                l_at_unsafe_comp = jnp.reshape(l_at_unsafe_comp, -1)
                # max_at_unsafe_comp = jnp.max(jnp.where(unsafe_complement_mask, l_at_unsafe_comp, 0))
                #
                # loss += jnp.maximum(max_at_unsafe_comp, unsafe_lb)
                l_at_unsafe_comp = jnp.where(unsafe_complement_mask, l_at_unsafe_comp, jnp.float64(1e18))
                min_at_unsafe_comp = jnp.max(l_at_unsafe_comp[jnp.argsort(l_at_unsafe_comp)[:40]])
                # min_at_unsafe_comp = jnp.min(jnp.where(unsafe_complement_mask, l_at_unsafe_comp, jnp.float64(1e18)))
                # loss += jnp.maximum(min_at_unsafe_comp - 1, 0)
                # loss += jnp.maximum(min_at_unsafe_comp - 1 * unsafe_lb_flag, 0)
                # loss += -jnp.minimum(min_at_unsafe - min_at_unsafe_comp - unsafe_lb + 0.99, 0)

                # In the target, minimum should be less than 1
                max_at_target = jnp.max(l_at_target)
                loss += jnp.maximum(max_at_target - 1, 0)
                # min_at_target = jnp.min(l_at_target)
                # loss += jnp.maximum(min_at_target - 1, 0)
                # max_at_center = jnp.max(l_at_center)
                # loss += jnp.maximum(max_at_center - 0.99, 0)
                # Global minimum should be inside target
                min_at_target = jnp.min(l_at_target)
                # min_at_init = jnp.min(l_at_init)
                min_at_unsafe = jnp.min(l_at_unsafe)
                loss += jnp.maximum(min_at_target - min_at_unsafe_comp, 0)
                # loss += jnp.maximum(min_at_target - min_at_init, 0)
                loss += jnp.maximum(min_at_target - min_at_unsafe, 0)

            return loss, (dec_loss, violations)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0, 1))
        (loss, (dec_loss, violations)), (l_grad, p_grad) = grad_fn(
            v_state.params, p_state.params
        )
        # Apply gradient clipping to stabilize training
        p_grad = clip_grad_norm(p_grad, 1)
        l_grad = clip_grad_norm(l_grad, 1)
        v_state = v_state.apply_gradients(grads=l_grad)
        p_state = p_state.apply_gradients(grads=p_grad)

        metrics = {"loss": loss, "dec_loss": dec_loss, "train_violations": violations}
        return v_state, p_state, metrics

    def train_epoch(self, train_ds, current_delta=0, current_alpha=1.01, train_v=True, train_p=True):
        """Train for a single epoch."""
        # current_big_delta = jnp.float32(current_big_delta)
        current_delta = jnp.float32(current_delta)
        current_alpha = jnp.float32(current_alpha)
        batch_metrics = []
        unsafe_lb_flag = jnp.float32(1 if train_p else 0)

        for state in train_ds.as_numpy_iterator():
            state = jnp.array(state)
            self.rng, rng = jax.random.split(self.rng, 2)

            new_v_state, new_p_state, metrics = self.train_step(
                self.v_state,
                self.p_state,
                state,
                rng,
                current_delta,
                current_alpha,
                unsafe_lb_flag
            )
            if train_p:
                self.p_state = new_p_state
            if train_v:
                self.v_state = new_v_state
            batch_metrics.append(metrics)

        # compute mean of metrics across each batch in epoch.
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {
            k: np.mean([metrics[k] for metrics in batch_metrics_np])
            for k in batch_metrics_np[0]
        }

        return epoch_metrics_np

    def save(self, filename):
        jax_save(
            {"policy": self.p_state, "value": self.c_state, "martingale": self.v_state},
            filename,
        )

    def load(self, filename, force_load_all=False):
        try:
            params = jax_load(
                {
                    "policy": self.p_state,
                    "value": self.c_state,
                    "martingale": self.v_state,
                },
                filename,
            )
            self.p_state = params["policy"]
            self.v_state = params["martingale"]
            self.c_state = params["value"]
        except KeyError as e:
            if force_load_all:
                raise e
            # Legacy load
            try:
                params = {"policy": self.p_state, "value": self.c_state}
                params = jax_load(params, filename)
                self.p_state = params["policy"]
                self.c_state = params["value"]
            except KeyError:
                params = {"policy": self.p_state}
                params = jax_load(params, filename)
                self.p_state = params["policy"]
