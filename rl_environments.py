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
from rsm_utils import triangular, make_unsafe_spaces, contained_in_any


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class LDSEnv(gym.Env):
    is_paralyzed = False

    def __init__(self, difficulty=1):
        self.steps = None
        self.state = None
        self.has_render = False
        self._difficulty = difficulty
        self.name = f"lds"

        safe = np.array([0.2, 0.2], np.float32)
        self.target_spaces = [spaces.Box(low=-safe, high=safe, dtype=np.float32)]

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
            # low=-0.6 * np.ones(2, dtype=np.float32),
            # high=0.6 * np.ones(2, dtype=np.float32),
            low=-0.7 * np.ones(2, dtype=np.float32),
            high=0.7 * np.ones(2, dtype=np.float32),
            # low=-1.5 * np.ones(2, dtype=np.float32),
            # high=1.5 * np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )
        # self.noise = np.array([0.0033, 0.0016])
        self.noise = np.array([0.002, 0.001])
        # self.noise = np.array([0.01, 0.005])
        # self.noise = np.array([0.0005, 0.0002])

        # self.unsafe_spaces = make_unsafe_spaces(
        #     self.observation_space, np.array([0.9, 0.9], np.float32)
        # )[0:2]
        self.unsafe_spaces = [
            spaces.Box(
                low=self.observation_space.low,
                high=np.array([self.observation_space.low[0] + 0.1, -0.4]),
                # high=np.array([self.observation_space.low[0] + 0.1, 0.0]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([self.observation_space.high[0] - 0.1, 0.4]),
                # low=np.array([self.observation_space.high[0] - 0.1, 0.0]),
                high=self.observation_space.high,
                dtype=np.float32,
            ),
        ]

        # self.noise = np.array([0.001, 0.001])
        self._jax_rng = jax.random.PRNGKey(777)
        self.v_next = jax.vmap(self.next, in_axes=(0, 0), out_axes=0)
        self.reset()

    @property
    def noise_bounds(self):
        return -self.noise, self.noise

    @property
    def observation_dim(self):
        return self.observation_space.shape[0]

    @property
    def action_dim(self):
        return self.action_space.shape[0]

    @partial(jax.jit, static_argnums=(0,))
    def next(self, state, action):
        action = jnp.clip(action, -1, 1)

        if self._difficulty == 0:
            # Easiest env
            new_y = 0.9 * state[1] + action[0] * 0.3
            new_x = 0.9 * state[0] + new_y * 0.1
        elif self._difficulty == 1:
            # mid env
            # new_y = state[1] * 0.966 + action[0] * 0.1
            # new_x = state[0] * 1.0 + new_y * 0.033

            new_y = state[1] * 0.98 + action[0] * 0.1
            new_x = state[0] * 1.0 + new_y * 0.02

            # new_y = state[1] * 0.9 + action[0] * 0.3
            # new_x = state[0] * 1.0 + new_y * 0.1

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

        if self.target_spaces[0].contains(next_state):
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
            # A = np.max(np.sum(np.array([[1, 0.2 * 0.98, 0.2 * 0.1], [0, 0.98, 0.1]]), axis=0))
            A = np.max(np.sum(np.array([[1, 0.1 * 0.9, 0.1 * 0.3], [0, 0.9, 0.3]]), axis=0))
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
    is_paralyzed = False

    def __init__(self):
        self.name = "pend"
        self.has_render = True
        self.steps = 0
        self.viewer = None

        # init = np.array([0.3, 0.3], np.float32)
        init = np.array([0.7, 0.7], np.float32)
        self.init_spaces = [spaces.Box(low=-init, high=init, dtype=np.float32)]
        # init = np.array([1, 1], np.float32)
        init = np.array([0.75, 0.75], np.float32)
        self.init_spaces_train = [spaces.Box(low=-init, high=init, dtype=np.float32)]

        high = np.array([3, 3], dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # self.noise = np.array([0.02, 0.01])
        self.noise = np.array([0.005, 0.002])

        safe = np.array([0.2, 0.2], np.float32)
        self.target_spaces = [spaces.Box(low=-safe, high=safe, dtype=np.float32)]
        safe = np.array([0.1, 0.1], np.float32)
        self.target_space_train = spaces.Box(low=-safe, high=safe, dtype=np.float32)

        # observation_space = np.array([1.5, 1.5], np.float32)  # make it fail
        # observation_space = np.array([0.7, 0.7], np.float32)
        observation_space = np.array([3, 3], np.float32)
        # observation_space = np.array([0.5, 0.5], np.float32)  # same as in AAAI
        self.observation_space = spaces.Box(
            low=-observation_space, high=observation_space, dtype=np.float32
        )

        self.unsafe_spaces = [
            spaces.Box(
                low=self.observation_space.low,
                high=np.array([self.observation_space.low[0] + 0.1, 0]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([self.observation_space.high[0] - 0.1, 0]),
                high=self.observation_space.high,
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
        # l = 0.3  # was 0.5 before
        b = 0.1

        u = 2 * jnp.clip(action, -1, 1)[0]
        newthdot = (1 - b) * thdot + (
                -3 * g * 0.5 / (2 * l) * jnp.sin(th + jnp.pi) + 3.0 / (m * l ** 2) * u
        ) * dt
        newthdot = jnp.clip(newthdot, -max_speed, max_speed)
        newth = th + newthdot * dt

        newth = jnp.clip(
            newth, self.observation_space.low[0], self.observation_space.high[0]
        )
        newthdot = jnp.clip(
            newthdot, self.observation_space.low[1], self.observation_space.high[1]
        )
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
        # done = self.steps >= 200 or np.abs(th) > 2
        done = self.steps >= 200

        if not deterministic:
            next_state = self.add_noise(next_state)

        if contained_in_any(self.unsafe_spaces, next_state):
            reward = -1

        reward = -costs
        if self.target_space_train.contains(next_state):
            reward += 1
        self.state = next_state
        return next_state, reward, done, {}

    @property
    def observation_dim(self):
        return self.observation_space.shape[0]

    @property
    def action_dim(self):
        return self.action_space.shape[0]

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
    is_paralyzed = False

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
        self.target_spaces = [spaces.Box(low=-safe, high=safe, dtype=np.float32)]

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

        if self.target_spaces[0].contains(next_state):
            reward += 1

        reward -= np.mean(np.abs(next_state / self.observation_space.high))
        self.state = next_state
        done = self.steps >= 200
        return self.state, reward, done, {}

    @property
    def observation_dim(self):
        return self.observation_space.shape[0]

    @property
    def action_dim(self):
        return self.action_space.shape[0]

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


class CollisionAvoidance3DEnv(gym.Env):
    name = "ca3d"
    is_paralyzed = False

    def __init__(self):
        self.steps = None
        self.state = None
        self.has_render = False

        # init = np.array([1.0, 1.0], np.float32)
        # self.init_space = spaces.Box(low=-init, high=init, dtype=np.float32)

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.ones(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            dtype=np.float32,
        )
        # was 0.05 before
        # self.noise = np.array([0.02, 0.02, 0.02])  # was 0.02 before
        self.noise = np.array([0.005, 0.005, 0.005])  # was 0.02 before
        safe = np.array([0.2, 0.2, 0.2], np.float32)  # was 0.1 before
        # safe = np.array([0.1, 0.1, 0.1], np.float32)
        self.safe_space = spaces.Box(low=-safe, high=safe, dtype=np.float32)

        self.target_spaces = [spaces.Box(low=-safe, high=safe, dtype=np.float32)]
        # self.target_spaces = []
        #
        # self.target_spaces.append(
        #     spaces.Box(
        #         low=np.array([0.8, -0.2, -0.2]),
        #         high=np.array([1.0, 0.2, 0.2]),
        #         dtype=np.float32,
        #     )
        # )
        # self.target_spaces.append(
        #     spaces.Box(
        #         low=np.array([-1.0, -0.2, -0.2]),
        #         high=np.array([-0.8, 0.2, 0.2]),
        #         dtype=np.float32,
        #     )
        # )

        self.init_spaces_train = make_unsafe_spaces(
            self.observation_space, np.array([0.9, 0.9, 0.9], np.float32)
        )
        self.init_spaces = [
            spaces.Box(
                low=np.array([-1, -0.6, -1]),
                high=np.array([-0.9, 0.6, 1]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([0.9, -0.6, -1]),
                high=np.array([1.0, 0.6, 1]),
                dtype=np.float32,
            ),
        ]

        self.unsafe_spaces = []
        # self.unsafe_spaces.append(
        #     spaces.Box(
        #         low=np.array([0.5, -0.1]), high=np.array([0.6, 0.1]), dtype=np.float32
        #     )
        # )
        # self.unsafe_spaces.append(
        #     spaces.Box(
        #         low=np.array([-0.6, -0.1]), high=np.array([-0.5, 0.1]), dtype=np.float32
        #     )
        # )
        self.unsafe_spaces.append(
            spaces.Box(
                low=np.array([0.7, 0.7, 0.7]),
                high=np.array([1.0, 1.0, 1.0]),
                dtype=np.float32,
            )
        )
        self.unsafe_spaces.append(
            spaces.Box(
                low=np.array([-1.0, -1.0, -1.0]),
                high=np.array([-0.7, -0.7, -0.7]),
                dtype=np.float32,
            )
        )

        # self.unsafe_spaces.append(
        #     spaces.Box(
        #         low=np.array([-0.3, 0.7, -0.3]),
        #         high=np.array([0.3, 1.0, 0.3]),
        #         dtype=np.float32,
        #     )
        # )
        # self.unsafe_spaces.append(
        #     spaces.Box(
        #         low=np.array([-0.3, -1.0, -0.3]),
        #         high=np.array([0.3, -0.7, 0.3]),
        #         dtype=np.float32,
        #     )
        # )

        # self.observation_space = spaces.Box(
        #     low=np.array([0.0, -1]),
        #     high=np.array([1.0, -0.0]),
        #     dtype=np.float32,
        # )
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

        # obstacle1 = jnp.array((0, 1, 0))
        # force1 = jnp.array((0, 1, 0))
        obstacle1 = jnp.array((1, 1, 1))
        force1 = jnp.array((1, 1, 1))
        dist1 = jnp.linalg.norm(obstacle1 - state)
        # dist1 = jnp.clip(dist1 / 0.3, 0, 1)
        dist1 = jnp.clip(dist1 / 0.3, 0.1, 1)
        action = action * dist1 + (1 - dist1) * force1

        # obstacle2 = jnp.array((0, -1, 0))
        # force2 = jnp.array((0, -1, 0))
        obstacle2 = jnp.array((-1, -1, -1))
        force2 = jnp.array((-1, -1, -1))
        dist2 = jnp.linalg.norm(obstacle2 - state)
        # dist2 = jnp.clip(dist2 / 0.3, 0, 1)
        dist2 = jnp.clip(dist2 / 0.3, 0.1, 1)
        action = action * dist2 + (1 - dist2) * force2

        # not_in_unsafe = 1.0
        # for unsafe in self.unsafe_spaces:
        #     is_inside = jnp.all(state >= unsafe.low).astype(jnp.float32) * jnp.all(
        #         state <= unsafe.high
        #     ).astype(jnp.float32)
        #     not_in_unsafe = not_in_unsafe * (1.0 - is_inside)

        # state = state + action * 0.2
        state = state + action * 0.02
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
        # obstacle1 = jnp.array((0, 1, 0))
        obstacle1 = jnp.array((1, 1, 1))
        dist1 = jnp.linalg.norm(obstacle1 - next_state)
        # obstacle2 = jnp.array((0, -1, 0))
        obstacle2 = jnp.array((-1, -1, -1))
        dist2 = jnp.linalg.norm(obstacle2 - next_state)

        if dist1 < 0.4 or dist2 < 0.4:
            reward -= 1

        # if self.safe_space.contains(next_state):
        #     reward += 1

        for targe in self.target_spaces:
            if targe.contains(next_state):
                reward += 1

        reward -= np.mean(np.abs(next_state / self.observation_space.high))
        self.state = next_state
        done = self.steps >= 200
        return self.state, reward, done, {}

    @property
    def lipschitz_constant(self):
        return 1.2

    def integrate_noise(self, a: list, b: list):
        dims = 3
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

    @property
    def observation_dim(self):
        return self.observation_space.shape[0]

    @property
    def action_dim(self):
        return self.action_space.shape[0]


class vDroneEnv(gym.Env):
    name = "vdrone"
    is_paralyzed = True

    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.ones(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            # low=-0.7 * np.ones(3, dtype=np.float32),
            # high=0.7 * np.ones(3, dtype=np.float32),
            dtype=np.float32,
        )
        # was 0.05 before
        self.noise = np.array([0.02, 0.02, 0.02])  # was 0.02 before
        # self.noise = np.array([0.005, 0.005, 0.005])
        safe = np.array([0.2, 0.2, 0.2], np.float32)  # was 0.1 before
        self.target_spaces = [spaces.Box(low=-safe, high=safe, dtype=np.float32)]

        self.init_spaces = [
            spaces.Box(
                low=np.array([-0.8, -0.5, -0.5]),
                high=np.array([-0.6, 0.5, 0.5]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([0.6, -0.5, -0.5]),
                high=np.array([0.8, 0.5, 0.5]),
                dtype=np.float32,
            ),
        ]
        self.init_spaces_train = make_unsafe_spaces(
            self.observation_space, np.array([0.1, 0.1, 0.1])
        )
        self.unsafe_spaces = []
        self.unsafe_spaces.append(
            spaces.Box(
                low=np.array([-0.3, 0.7, -0.3]),
                high=np.array([0.3, 1.0, 0.3]),
                dtype=np.float32,
            )
        )
        self.unsafe_spaces.append(
            spaces.Box(
                low=np.array([-0.3, -1.0, -0.3]),
                high=np.array([0.3, -0.7, 0.3]),
                dtype=np.float32,
            )
        )

        # self.unsafe_spaces.append(
        #     spaces.Box(
        #         # low=np.array([0.7, 0.7, 0.7]),
        #         # high=np.array([1.0, 1.0, 1.0]),
        #         # low=np.array([0.4, 0.4, 0.4]),
        #         # high=np.array([0.7, 0.7, 0.7]),
        #         # low=np.array([0.6, -0.7, -0.7]),
        #         # high=np.array([0.7, 0.7, 0.7]),
        #         # low=np.array([0.6, 0.4, -0.7]),
        #         # high=np.array([0.7, 0.7, 0.7]),
        #         low=np.array([0.65, -0.7, -0.7]),
        #         high=np.array([0.7, 0.7, 0.7]),
        #         dtype=np.float32,
        #     )
        # )
        # self.unsafe_spaces.append(
        #     spaces.Box(
        #         # low=np.array([-1.0, -1.0, -1.0]),
        #         # high=np.array([-0.7, -0.7, -0.7]),
        #         # low=np.array([-0.7, -0.7, -0.7]),
        #         # high=np.array([-0.4, -0.4, -0.4]),
        #         # low=np.array([-0.7, -0.7, -0.7]),
        #         # high=np.array([-0.6, 0.7, 0.7]),
        #         # low=np.array([-0.7, -0.7, -0.7]),
        #         # high=np.array([-0.6, -0.4, 0.7]),
        #         low=np.array([-0.7, -0.7, -0.7]),
        #         high=np.array([-0.65, 0.7, 0.7]),
        #         dtype=np.float32,
        #     )
        # )
        #
        # self.unsafe_spaces.append(
        #     spaces.Box(
        #         low=np.array([-0.65, 0.65, -0.7]),
        #         high=np.array([0.65, 0.7, 0.7]),
        #         dtype=np.float32,
        #     )
        # )
        #
        # self.unsafe_spaces.append(
        #     spaces.Box(
        #         low=np.array([-0.65, -0.7, -0.7]),
        #         high=np.array([0.65, -0.65, 0.7]),
        #         dtype=np.float32,
        #     )
        # )
        # self.unsafe_spaces.append(
        #     spaces.Box(
        #         low=np.array([-0.65, -0.65, 0.65]),
        #         high=np.array([0.65, 0.65, 0.7]),
        #         dtype=np.float32,
        #     )
        # )
        # self.unsafe_spaces.append(
        #     spaces.Box(
        #         low=np.array([-0.65, -0.65, -0.7]),
        #         high=np.array([0.65, 0.65, -0.65]),
        #         dtype=np.float32,
        #     )
        # )

        self._jax_rng = jax.random.PRNGKey(777)
        self.v_next = jax.vmap(self.next, in_axes=(0, 0), out_axes=0)
        self.v_step = jax.jit(jax.vmap(self.step))
        self.v_reset = jax.jit(jax.vmap(self.reset))

    @property
    def noise_bounds(self):
        return -self.noise, self.noise

    @partial(jax.jit, static_argnums=(0,))
    def next(self, state, action):
        action = jnp.clip(action, -1, 1)

        #         obstacle1 = jnp.array((0, 1, 0))
        #         force1 = jnp.array((0, 1, 0))
        #         dist1 = jnp.linalg.norm(obstacle1 - state)
        #         # dist1 = jnp.clip(dist1 / 0.3, 0, 1)
        #         # dist1 = jnp.clip(dist1 / 0.3, 0.5, 1)
        #         dist1 = jnp.clip(dist1 / 0.5, 0.6, 1)
        #         action = action * dist1 + (1 - dist1) * force1

        #         obstacle2 = jnp.array((0, -1, 0))
        #         force2 = jnp.array((0, -1, 0))
        #         dist2 = jnp.linalg.norm(obstacle2 - state)
        #         # dist2 = jnp.clip(dist2 / 0.3, 0, 1)
        #         # dist2 = jnp.clip(dist2 / 0.3, 0.5, 1)
        #         dist2 = jnp.clip(dist2 / 0.5, 0.6, 1)
        #         action = action * dist2 + (1 - dist2) * force2

        state = state + action * 0.2
        # state = state + action * 0.02
        state = jnp.clip(state, self.observation_space.low, self.observation_space.high)

        return state

    def add_noise(self, state):
        self._jax_rng, rng = jax.random.split(self._jax_rng, 2)
        noise = triangular(rng, (self.observation_space.shape[0],))
        noise = noise * self.noise
        return state + noise

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action, rng):
        step = state[0]
        state = state[1:4]
        next_state = self.next(state, action)

        noise = triangular(rng, (self.observation_space.shape[0],))
        noise = noise * self.noise
        next_state = next_state + noise
        next_state = np.clip(
            next_state, self.observation_space.low, self.observation_space.high
        )

        reward = 0
        for unsafe in self.unsafe_spaces:
            contain = jnp.all(
                jnp.logical_and(state >= unsafe.low, state <= unsafe.high)
            )
            reward += -jnp.float32(contain)
            center = 0.5 * (unsafe.low + unsafe.high)
            dist = jnp.sum(jnp.abs(center - next_state))
            dist = jnp.clip(dist, 0, 0.5)
            reward -= 1 * (0.5 - dist)

        for target in self.target_spaces:
            contain = jnp.all(
                jnp.logical_and(state >= target.low, state <= target.high)
            )
            reward += jnp.float32(contain)

        reward -= 2 * jnp.mean(jnp.abs(next_state / self.observation_space.high))
        done = step >= 200
        next_packed = jnp.array([step + 1, next_state[0], next_state[1], next_state[2]])
        return next_packed, next_state, reward, done

    @property
    def observation_dim(self):
        return self.observation_space.shape[0]

    @property
    def action_dim(self):
        return self.action_space.shape[0]

    def integrate_noise(self, a: list, b: list):
        dims = 3
        pmass = np.ones(a[0].shape[0])
        for i in range(dims):
            loc = self.noise_bounds[0][i]
            scale = self.noise_bounds[1][i] - self.noise_bounds[0][i]
            marginal_pmass = triang.cdf(b[i], c=0.5, loc=loc, scale=scale) - triang.cdf(
                a[i], c=0.5, loc=loc, scale=scale
            )
            pmass *= marginal_pmass
        return pmass

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng):
        # obs = jax.random.uniform(
        #     rng,
        #     shape=(self.observation_space.shape[0],),
        #     minval=self.observation_space.low,
        #     maxval=self.observation_space.high,
        # )
        lowers = jnp.stack([init.low for init in self.init_spaces_train], 0)
        high = jnp.stack([init.high for init in self.init_spaces_train], 0)
        rng1, rng2 = jax.random.split(rng, 2)
        index = jax.random.randint(
            rng1, shape=(), minval=0, maxval=len(self.init_spaces_train)
        )
        obs = jax.random.uniform(
            rng2, shape=(lowers.shape[1],), minval=lowers[index], maxval=high[index]
        )
        state = jnp.array([0, obs[0], obs[1], obs[2]])
        return state, obs

    @property
    def lipschitz_constant(self):
        return 1.2
        # return 1.02