import gym
from gym import spaces
import numpy as np
from os import path
from scipy.stats import triang
import jax.numpy as jnp
from functools import partial
import jax
import matplotlib.pyplot as plt
import os
from klax import triangular, make_unsafe_spaces, contained_in_any


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


# (1 - |x|)/(1


class LDSEnv(gym.Env):
    def __init__(self, difficulty=1):
        self.steps = None
        self.state = None
        self.has_render = False
        self._difficulty = difficulty
        self.name = f"lds"

        safe = np.array([0.2, 0.2], np.float32)
        self.safe_space = spaces.Box(low=-safe, high=safe, dtype=np.float32)

        # init and  safe should be non-overlapping
        # init = np.array([0.4, 0.4], np.float32)
        # self.init_spaces = make_unsafe_spaces(
        #     spaces.Box(low=-init, high=init, dtype=np.float32), safe
        # )
        self.init_spaces = [
            spaces.Box(
                low=np.array([-0.25, -0.1]),
                high=np.array([-0.2, 0.1]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([0.25, -0.1]),
                high=np.array([0.2, 0.1]),
                dtype=np.float32,
            ),
        ]

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            # low=-0.8 * np.ones(2, dtype=np.float32),
            # high=0.8 * np.ones(2, dtype=np.float32),
            low=-1.5 * np.ones(2, dtype=np.float32),
            high=1.5 * np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )
        self.noise = np.array([0.01, 0.005])
        # self.noise = np.array([0.0005, 0.0002])

        # self.unsafe_spaces = make_unsafe_spaces(
        #     self.observation_space, np.array([0.9, 0.9], np.float32)
        # )[0:2]
        self.unsafe_spaces = [
            spaces.Box(
                low=self.observation_space.low,
                high=np.array([self.observation_space.low[0] + 0.1, 0.0]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([self.observation_space.high[0] - 0.1, 0.0]),
                high=self.observation_space.high,
                dtype=np.float32,
            ),
        ]

        self.reach_space = self.observation_space
        # self.noise = np.array([0.001, 0.001])
        self._jax_rng = jax.random.PRNGKey(777)
        self.v_next = jax.vmap(self.next, in_axes=(0, 0), out_axes=0)
        self.reset()

    @property
    def noise_bounds(self):
        return -self.noise, self.noise

    @partial(jax.jit, static_argnums=(0,))
    def next(self, state, action):
        action = jnp.clip(action, -1, 1)

        if self._difficulty == 0:
            # Easiest env
            new_y = 0.9 * state[1] + action[0] * 0.3
            new_x = 0.9 * state[0] + new_y * 0.1
        elif self._difficulty == 1:
            # mid env
            new_y = state[1] * 0.9 + action[0] * 0.3
            new_x = state[0] * 1.0 + new_y * 0.1

            # new_x = oldx + 0.045 y + 0.45u
            # new y = 0.9*oldy + 0 + 0.5u
            # new_x = state[0] * 1.0 + new_y * 0.2 + action[0] * 0.1
        else:
            # hard harder
            new_y = state[1] + action[0] * 0.2
            new_x = state[0] + new_y * 0.3 + action[0] * 0.05
        new_y = np.clip(
            new_y, self.observation_space.low[1], self.observation_space.high[1]
        )
        new_x = np.clip(
            new_x, self.observation_space.low[0], self.observation_space.high[0]
        )
        return jnp.array([new_x, new_y])

    def add_noise(self, state):
        self._jax_rng, rng = jax.random.split(self._jax_rng, 2)
        noise = triangular(rng, (self.observation_space.shape[0],))
        noise = noise * self.noise
        return state + noise

    def step(self, action, deterministic=False):
        self.steps += 1

        next_state = self.next(self.state, action)

        if not deterministic:
            next_state = self.add_noise(next_state)
        next_state = np.array(next_state)

        reward = 0
        # unsafe_box = spaces.Box(
        #     low=-self.unsafe_bounds, high=self.unsafe_bounds, dtype=np.float32
        # )
        # if not unsafe_box.contains(next_state):
        #     reward = -1
        if contained_in_any(self.unsafe_spaces, next_state):
            reward = -1

        if self.safe_space.contains(next_state):
            reward = 1

        reward -= np.mean(np.abs(next_state / self.observation_space.high))
        self.state = next_state
        done = self.steps >= 200
        return self.state, reward, done, {}

    @property
    def lipschitz_constant(self):
        if self._difficulty == 0:
            A = np.max(np.sum(np.array([[1, 0.2, 0.0], [0, 1, 0.3]]), axis=0))
        elif self._difficulty == 1:
            A = np.max(np.sum(np.array([[1, 0.045, 0.45], [0, 0.9, 0.5]]), axis=0))
        else:
            A = np.max(np.sum(np.array([[0, 0.9, 0.5], [0, 1, 0.2]]), axis=0))
        return A

    def integrate_noise(self, a: list, b: list):
        dims = 2
        pmass = np.ones(a[0].shape[0])
        for i in range(dims):
            loc = self.noise_bounds[0][i]
            scale = self.noise_bounds[1][i] - self.noise_bounds[0][i]
            marginal_pmass = triang.cdf(b[i], c=0.5, loc=loc, scale=scale) - triang.cdf(
                a[i], c=0.5, loc=loc, scale=scale
            )
            pmass *= marginal_pmass
        return pmass

    def reset(self, state=None):
        if state is None:
            state = self.init_spaces[0].sample()
        self.state = state
        self.steps = 0
        return self.state


class InvertedPendulum(gym.Env):
    def __init__(self):
        self.name = "pend"
        self.has_render = True
        self.steps = 0
        self.viewer = None

        init = np.array([0.3, 0.3], np.float32)
        self.init_spaces = [spaces.Box(low=-init, high=init, dtype=np.float32)]
        init = np.array([-1, 1], np.float32)
        self.init_spaces_train = [spaces.Box(low=-init, high=init, dtype=np.float32)]

        high = np.array([3, 3], dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.noise = np.array([0.02, 0.01])

        safe = np.array([0.2, 0.2], np.float32)
        self.safe_space = spaces.Box(low=-safe, high=safe, dtype=np.float32)
        safe = np.array([0.1, 0.1], np.float32)
        self.safe_space_train = spaces.Box(low=-safe, high=safe, dtype=np.float32)

        # reach_space = np.array([1.5, 1.5], np.float32)  # make it fail
        reach_space = np.array([0.7, 0.7], np.float32)
        # reach_space = np.array([0.5, 0.5], np.float32)  # same as in AAAI
        self.reach_space = spaces.Box(
            low=-reach_space, high=reach_space, dtype=np.float32
        )

        self.unsafe_spaces = [
            spaces.Box(
                low=self.reach_space.low,
                high=np.array([self.reach_space.low[0] + 0.1, 0.0]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([self.reach_space.high[0] - 0.1, 0.0]),
                high=self.reach_space.high,
                dtype=np.float32,
            ),
        ]

        self._jax_rng = jax.random.PRNGKey(777)
        self.v_next = jax.vmap(self.next, in_axes=(0, 0), out_axes=0)
        self._fig_id = 0
        self.reset()

    @partial(jax.jit, static_argnums=(0,))
    def next(self, state, action):
        th, thdot = state  # th := theta
        max_speed = 5
        dt = 0.05
        g = 10
        m = 0.15
        l = 0.5
        b = 0.1

        u = 2 * jnp.clip(action, -1, 1)[0]
        newthdot = (1 - b) * thdot + (
            -3 * g * 0.5 / (2 * l) * jnp.sin(th + jnp.pi) + 3.0 / (m * l ** 2) * u
        ) * dt
        newthdot = jnp.clip(newthdot, -max_speed, max_speed)
        newth = th + newthdot * dt

        newth = jnp.clip(newth, self.reach_space.low[0], self.reach_space.high[0])
        newthdot = jnp.clip(newthdot, self.reach_space.low[1], self.reach_space.high[1])
        return jnp.array([newth, newthdot])

    def add_noise(self, state):
        self._jax_rng, rng = jax.random.split(self._jax_rng, 2)
        noise = triangular(rng, (self.observation_space.shape[0],))
        noise = noise * self.noise
        return state + noise

    def step(self, action, deterministic=False):
        self.steps += 1

        next_state = self.next(self.state, action)
        next_state = np.array(next_state)
        th, thdot = next_state
        u = np.clip(action[0], -1, 1)
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2
        done = self.steps >= 200 or np.abs(th) > 2

        if not deterministic:
            next_state = self.add_noise(next_state)

        reward = -costs + 1
        if self.safe_space_train.contains(next_state):
            reward += 1
        self.state = next_state
        return next_state, reward, done, {}

    @property
    def noise_bounds(self):
        return -self.noise, self.noise

    @property
    def lipschitz_constant(self):
        return 1.78

    def integrate_noise(self, a: list, b: list):
        dims = 2
        pmass = np.ones(a[0].shape[0])
        for i in range(dims):
            loc = self.noise_bounds[0][i]
            scale = self.noise_bounds[1][i] - self.noise_bounds[0][i]
            marginal_pmass = triang.cdf(b[i], c=0.5, loc=loc, scale=scale) - triang.cdf(
                a[i], c=0.5, loc=loc, scale=scale
            )
            pmass *= marginal_pmass
        return pmass

    def reset(self, state=None):
        if state is None:
            i = np.random.default_rng().integers(0, len(self.init_spaces_train))
            state = self.init_spaces_train[i].sample()
        self.state = state
        self.steps = 0
        return self.state


class CollisionAvoidanceEnv(gym.Env):
    name = "cavoid"

    def __init__(self):
        self.steps = None
        self.state = None
        self.has_render = False

        # init = np.array([1.0, 1.0], np.float32)
        # self.init_space = spaces.Box(low=-init, high=init, dtype=np.float32)

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )
        # was 0.05 before
        self.noise = np.array([0.05, 0.05])  # was 0.02 before
        safe = np.array([0.2, 0.2], np.float32)  # was 0.1 before
        self.safe_space = spaces.Box(low=-safe, high=safe, dtype=np.float32)

        self.init_spaces_train = make_unsafe_spaces(
            self.observation_space, np.array([0.9, 0.9], np.float32)
        )
        self.init_spaces = [
            spaces.Box(
                low=np.array([-1, -0.6]),
                high=np.array([-0.9, 0.6]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([0.9, -0.6]),
                high=np.array([1.0, 0.6]),
                dtype=np.float32,
            ),
        ]

        self.unsafe_spaces = []
        self.unsafe_spaces.append(
            spaces.Box(
                low=np.array([-0.3, 0.7]), high=np.array([0.3, 1.0]), dtype=np.float32
            )
        )
        self.unsafe_spaces.append(
            spaces.Box(
                low=np.array([-0.3, -1.0]), high=np.array([0.3, -0.7]), dtype=np.float32
            )
        )
        self.reach_space = self.observation_space
        # self.noise = np.array([0.001, 0.001])
        self._jax_rng = jax.random.PRNGKey(777)
        self.v_next = jax.vmap(self.next, in_axes=(0, 0), out_axes=0)
        self.reset()

    @property
    def noise_bounds(self):
        return -self.noise, self.noise

    @partial(jax.jit, static_argnums=(0,))
    def next(self, state, action):
        action = jnp.clip(action, -1, 1)

        obstacle1 = jnp.array((0, 1))
        force1 = jnp.array((0, 1))
        dist1 = jnp.linalg.norm(obstacle1 - state)
        dist1 = jnp.clip(dist1 / 0.3, 0, 1)
        action = action * dist1 + (1 - dist1) * force1

        obstacle2 = jnp.array((0, -1))
        force2 = jnp.array((0, -1))
        dist2 = jnp.linalg.norm(obstacle2 - state)
        dist2 = jnp.clip(dist2 / 0.3, 0, 1)
        action = action * dist2 + (1 - dist2) * force2

        state = state + action * 0.2
        state = jnp.clip(state, self.observation_space.low, self.observation_space.high)

        return state

    def add_noise(self, state):
        self._jax_rng, rng = jax.random.split(self._jax_rng, 2)
        noise = triangular(rng, (self.observation_space.shape[0],))
        noise = noise * self.noise
        return state + noise

    def step(self, action, deterministic=False):
        self.steps += 1

        next_state = self.next(self.state, action)

        if not deterministic:
            next_state = self.add_noise(next_state)
        next_state = np.array(next_state)

        reward = 0
        # unsafe_box = spaces.Box(
        #     low=-self.unsafe_bounds, high=self.unsafe_bounds, dtype=np.float32
        # )
        # if not unsafe_box.contains(next_state):
        #     reward = -1
        # if contained_in_any(self.unsafe_spaces, next_state):
        #     reward = -2
        obstacle1 = jnp.array((0, 1))
        dist1 = jnp.linalg.norm(obstacle1 - next_state)
        obstacle2 = jnp.array((0, -1))
        dist2 = jnp.linalg.norm(obstacle2 - next_state)

        if dist1 < 0.4 or dist2 < 0.4:
            reward -= 1

        if self.safe_space.contains(next_state):
            reward += 1

        reward -= np.mean(np.abs(next_state / self.observation_space.high))
        self.state = next_state
        done = self.steps >= 200
        return self.state, reward, done, {}

    @property
    def lipschitz_constant(self):
        return 1.2

    def integrate_noise(self, a: list, b: list):
        dims = 2
        pmass = np.ones(a[0].shape[0])
        for i in range(dims):
            loc = self.noise_bounds[0][i]
            scale = self.noise_bounds[1][i] - self.noise_bounds[0][i]
            marginal_pmass = triang.cdf(b[i], c=0.5, loc=loc, scale=scale) - triang.cdf(
                a[i], c=0.5, loc=loc, scale=scale
            )
            pmass *= marginal_pmass
        return pmass

    def reset(self, state=None):
        if state is None:
            i = np.random.default_rng().integers(0, len(self.init_spaces_train))
            state = self.init_spaces_train[i].sample()

        self.state = state
        self.steps = 0
        return self.state
