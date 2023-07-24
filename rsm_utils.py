from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state  # Useful dataclass to keep train state
import flax
import numpy as np  # Ordinary NumPy
import optax  # Optimizers
from functools import partial
from gym import spaces
import matplotlib.pyplot as plt
import seaborn as sns


def pretty_time(elapsed):
    if elapsed > 60 * 60:
        h = int(elapsed // (60 * 60))
        mins = int((elapsed // 60) % 60)
        return f"{h}h {mins:02d} min"
    elif elapsed > 60:
        mins = elapsed // 60
        secs = int(elapsed) % 60
        return f"{mins:0.0f}min {secs}s"
    elif elapsed < 1:
        return f"{elapsed*1000:0.1f}ms"
    else:
        return f"{elapsed:0.1f}s"


def pretty_number(number):
    if number >= 1.0e9:
        return f"{number/1e9:0.3g}G"
    elif number >= 1.0e6:
        return f"{number/1e6:0.3g}M"
    elif number >= 1.0e3:
        return f"{number/1e3:0.3g}k"
    else:
        return number


def clip_and_filter_spaces(obs_space, space_list):
    new_space_list = []
    for space in space_list:
        new_space = spaces.Box(
            low=np.clip(space.low, obs_space.low, obs_space.high),
            high=np.clip(space.high, obs_space.low, obs_space.high),
        )
        volume = np.prod(new_space.high - new_space.low)
        if volume > 0:
            new_space_list.append(new_space)
    return new_space_list


def make_unsafe_spaces(obs_space, unsafe_bounds):
    unsafe_spaces = []
    dims = obs_space.shape[0]
    for i in range(dims):
        low = np.array(obs_space.low)
        high = np.array(obs_space.high)
        high[i] = -unsafe_bounds[i]
        if not np.allclose(low, high):
            unsafe_spaces.append(spaces.Box(low=low, high=high, dtype=np.float32))

        high = np.array(obs_space.high)
        low = np.array(obs_space.low)
        low[i] = unsafe_bounds[i]
        if not np.allclose(low, high):
            unsafe_spaces.append(spaces.Box(low=low, high=high, dtype=np.float32))
    return unsafe_spaces


def make_corner_spaces(obs_space, unsafe_bounds):
    unsafe_spaces = []
    dims = obs_space.shape[0]
    for i in range(dims):
        low = np.array(obs_space.low)
        high = np.array(obs_space.high)
        high[i] = low[i] + unsafe_bounds[i]
        if not np.allclose(low, high):
            unsafe_spaces.append(spaces.Box(low=low, high=high, dtype=np.float32))

        high = np.array(obs_space.high)
        low = np.array(obs_space.low)
        low[i] = high[i] - unsafe_bounds[i]
        if not np.allclose(low, high):
            unsafe_spaces.append(spaces.Box(low=low, high=high, dtype=np.float32))
    return unsafe_spaces


def enlarge_space(space, bound, limit_space=None):
    new_space = spaces.Box(low=space.low - bound, high=space.high + bound)
    if limit_space is not None:
        new_space = spaces.Box(
            low=np.clip(new_space.low, limit_space.low, limit_space.high),
            high=np.clip(new_space.high, limit_space.low, limit_space.high),
        )
    return new_space


@jax.jit
def clip_grad_norm(grad, max_norm):
    norm = jnp.linalg.norm(
        jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, grad))
    )
    factor = jnp.minimum(max_norm, max_norm / (norm + 1e-6))
    return jax.tree_map((lambda x: x * factor), grad)


def contained_in_any(spaces, state):
    for space in spaces:
        if space.contains(state):
            return True
    return False


def triangular(rng_key, shape):
    U = jax.random.uniform(rng_key, shape=shape)
    p1 = -1 + jnp.sqrt(2 * U)
    p2 = 1 - jnp.sqrt((1 - U) * 2)
    return jnp.where(U <= 0.5, p1, p2)


def softhuber(x):
    return jnp.sqrt(1 + jnp.square(x)) - 1


class MLP(nn.Module):
    features: Sequence[int]
    activation: str = "relu"
    softplus_output: bool = False

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            if self.activation == "relu":
                x = nn.relu(x)
            else:
                x = nn.tanh(x)
        x = nn.Dense(self.features[-1])(x)
        if self.softplus_output:
            x = jax.nn.softplus(x)
        return x


# Must be called "Dense" because flax uses self.__class__.__name__ to name variables
class Dense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, inputs):
        lower_bound_head, upper_bound_head = inputs
        kernel = self.param(
            "kernel",
            jax.nn.initializers.glorot_uniform(),
            (lower_bound_head.shape[-1], self.features),
        )  # shape info.
        bias = self.param("bias", nn.initializers.zeros, (self.features,))
        # Center and width
        center_prev = 0.5 * (upper_bound_head + lower_bound_head)
        edge_len_prev = 0.5 * jnp.maximum(
            upper_bound_head - lower_bound_head, 0
        )  # avoid numerical issues

        # Two matrix multiplications
        center = jnp.matmul(center_prev, kernel) + bias
        edge_len = jnp.matmul(edge_len_prev, jnp.abs(kernel))  # Edge length has no bias

        # New bounds
        lower_bound_head = center - edge_len
        upper_bound_head = center + edge_len
        # self.sow("intermediates", "edge_len", edge_len)
        return [lower_bound_head, upper_bound_head]


class IBPMLP(nn.Module):
    features: Sequence[int]
    activation: str = "relu"
    softplus_output: bool = False

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = Dense(feat)(x)
            if self.activation == "relu":
                x = [nn.relu(x[0]), nn.relu(x[1])]
            else:
                x = [nn.tanh(x[0]), nn.tanh(x[1])]
        x = Dense(self.features[-1])(x)
        if self.softplus_output:
            x = [jax.nn.softplus(x[0]), jax.nn.softplus(x[1])]
        return x


def martingale_loss(l, l_next, eps):
    diff = l_next - l
    return jnp.mean(jnp.maximum(diff + eps, 0))


def jax_save(params, filename):
    bytes_v = flax.serialization.to_bytes(params)
    with open(filename, "wb") as f:
        f.write(bytes_v)


def jax_load(params, filename):
    with open(filename, "rb") as f:
        bytes_v = f.read()
    params = flax.serialization.from_bytes(params, bytes_v)
    return params


def lipschitz_l1_jax(params):
    lipschitz_l1 = 1
    # Max over input axis
    for i, (k, v) in enumerate(params["params"].items()):
        lipschitz_l1 *= jnp.max(jnp.sum(jnp.abs(v["kernel"]), axis=0))

    return lipschitz_l1


def create_train_state(model, rng, in_dim, learning_rate, ema=0):
    """Creates initial `TrainState`."""
    params = model.init(rng, jnp.ones([1, in_dim]))
    tx = optax.adam(learning_rate)
    if ema > 0:
        tx = optax.chain(tx, optax.ema(ema))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def plot_policy(env, policy, filename, rsm=None, title=None):
    dims = env.observation_dim

    sns.set()
    fig, ax = plt.subplots(figsize=(6, 6))

    if env.observation_dim == 2:
        if rsm is not None:
            grid, new_steps = [], []
            for i in range(dims):
                samples = jnp.linspace(
                    env.observation_dim.low[i],
                    env.observation_dim.high[i],
                    50,
                    endpoint=False,
                    retstep=True,
                )
                grid.append(samples.flatten())
            grid = jnp.meshgrid(*grid)
            grid = jnp.stack(grid, axis=1)
            l = rsm.apply_fn(rsm.params, grid).flatten()
            l = np.array(l)
            sc = ax.scatter(
                grid[:, 0], grid[:, 1], marker="s", c=l, zorder=1, alpha=0.7
            )
            fig.colorbar(sc)

    n = 50
    rng = jax.random.PRNGKey(3)
    rng, r = jax.random.split(rng)
    r = jax.random.split(r, n)
    state, obs = env.v_reset(r)
    done = jnp.zeros(n, dtype=jnp.bool_)
    total_returns = jnp.zeros(n)
    obs_list = []
    done_list = []
    while not jnp.any(done):
        action = policy.apply_fn(policy.params, obs)
        rng, r = jax.random.split(rng)
        r = jax.random.split(r, n)
        state, new_obs, reward, new_done = env.v_step(state, action, r)
        total_returns += reward * (1.0 - done)
        done_list.append(done)
        obs_list.append(obs)
        obs, done = new_obs, new_done
    obs_list = jnp.stack(obs_list, 1)
    done_list = jnp.stack(done_list, 1)
    traces = [obs_list[i, jnp.logical_not(done_list[i])] for i in range(n)]

    if title is None:
        title = env.name

    title = (
        title
        + f" ({jnp.mean(total_returns):0.1f} [{jnp.min(total_returns):0.1f},{jnp.max(total_returns):0.1f}])"
    )
    ax.set_title(title)

    terminals_x, terminals_y = [], []
    for i in range(n):
        ax.plot(
            traces[i][:, 0],
            traces[i][:, 1],
            color=sns.color_palette()[0],
            zorder=2,
            alpha=0.15,
        )
        ax.scatter(
            traces[i][:, 0],
            traces[i][:, 1],
            color=sns.color_palette()[0],
            zorder=2,
            marker=".",
            alpha=0.4,
        )
        terminals_x.append(float(traces[i][-1, 0]))
        terminals_y.append(float(traces[i][-1, 1]))
    ax.scatter(terminals_x, terminals_y, color="white", marker="x", zorder=5)
    for init in env.init_spaces:
        x = [
            init.low[0],
            init.high[0],
            init.high[0],
            init.low[0],
            init.low[0],
        ]
        y = [
            init.low[1],
            init.low[1],
            init.high[1],
            init.high[1],
            init.low[1],
        ]
        ax.plot(x, y, color="cyan", alpha=0.5, zorder=7)
    for unsafe in env.unsafe_spaces:
        x = [
            unsafe.low[0],
            unsafe.high[0],
            unsafe.high[0],
            unsafe.low[0],
            unsafe.low[0],
        ]
        y = [
            unsafe.low[1],
            unsafe.low[1],
            unsafe.high[1],
            unsafe.high[1],
            unsafe.low[1],
        ]
        ax.plot(x, y, color="red", alpha=0.5, zorder=7)
    for target_space in env.target_spaces:
        x = [
            target_space.low[0],
            target_space.high[0],
            target_space.high[0],
            target_space.low[0],
            target_space.low[0],
        ]
        y = [
            target_space.low[1],
            target_space.low[1],
            target_space.high[1],
            target_space.high[1],
            target_space.low[1],
        ]
        ax.plot(x, y, color="green", alpha=0.5, zorder=7)

    ax.set_xlim([env.observation_space.low[0], env.observation_space.high[0]])
    ax.set_ylim([env.observation_space.low[1], env.observation_space.high[1]])
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":

    learning_rate = 0.0005
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    layer_size = [64, 16, 5]
    model = MLP(layer_size)
    ibp_model = IBPMLP(layer_size)
    state = create_train_state(model, init_rng, 8, learning_rate)

    print("Lipschitz: ", compute_lipschitz(state.params))
    del init_rng  # Must not be used anymore.

    fake_x = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(1, 8), minval=-1, maxval=1
    )
    fake_y = model.apply(state.params, fake_x)
    fake_x_lb = fake_x - 0.01
    fake_x_ub = fake_x + 0.01
    print("fake_x\n", fake_x)
    print("fake_x_lb\n", fake_x_lb)
    print("fake_x_ub\n", fake_x_ub)
    print("#### output ####")
    print("Fake y\n", fake_y)
    (fake_y_lb, fake_y_ub), mod_vars = ibp_model.apply(
        state.params, [fake_x_lb, fake_x_ub], mutable="intermediates"
    )
    print("Fake lb\n", fake_y_lb)
    print("Fake ub\n", fake_y_ub)

    print("diff", fake_y_ub - fake_y_lb)
    # print("sowed vars", mod_vars)

    # print("Params: ", state.params)
    # model = MLP([12, 8, 4])
    # batch = jnp.ones((32, 10))
    # variables = model.init(jax.random.PRNGKey(0), batch)
    # output = model.apply(variables, batch)