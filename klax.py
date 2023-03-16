from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.jax_utils import prefetch_to_device
from tensorflow.keras.datasets.mnist import load_data
import tensorflow as tf
from flax.training import train_state  # Useful dataclass to keep train state
import flax
import numpy as onp  # Ordinary NumPy
import optax  # Optimizers
from functools import partial
from gym import spaces

klax_config = {"eps": 1e-3}


def pretty_time(elapsed):
    if elapsed > 60 * 60:
        h = elapsed // (60 * 60)
        mins = (elapsed // 60) % 60
        return f"{h}h {mins:02d} min"
    elif elapsed > 60:
        mins = elapsed // 60
        secs = int(elapsed) % 60
        return f"{mins:0.0f}min {secs}s"
    elif elapsed < 1:
        return f"{elapsed*1000:0.1f}ms"
    else:
        return f"{elapsed:0.1f}s"


def make_unsafe_spaces(obs_space, unsafe_bounds):
    unsafe_spaces = []
    dims = obs_space.shape[0]
    for i in range(dims):
        low = onp.array(obs_space.low)
        high = onp.array(obs_space.high)
        high[i] = -unsafe_bounds[i]
        if not onp.allclose(low, high):
            unsafe_spaces.append(spaces.Box(low=low, high=high, dtype=onp.float32))

        high = onp.array(obs_space.high)
        low = onp.array(obs_space.low)
        low[i] = unsafe_bounds[i]
        if not onp.allclose(low, high):
            unsafe_spaces.append(spaces.Box(low=low, high=high, dtype=onp.float32))
    return unsafe_spaces


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
    square_output: bool = False

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            if self.activation == "relu":
                x = nn.relu(x)
            else:
                x = nn.tanh(x)
        x = nn.Dense(self.features[-1])(x)
        if self.square_output:
            # x = jnp.square(x)
            # x = jnp.abs(x)
            x = jax.nn.softplus(x)
            # x = softhuber(x)
        return x


# Must be called "Dense" because flax uses self.__class__.__name__ to name variables
class Dense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, inputs):
        lower_bound_head, upper_bound_head = inputs
        kernel = self.param(
            "kernel",
            # nn.initializers.zeros,  # RNG passed implicitly.
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
    square_output: bool = False

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = Dense(feat)(x)
            if self.activation == "relu":
                x = [nn.relu(x[0]), nn.relu(x[1])]
            else:
                x = [nn.tanh(x[0]), nn.tanh(x[1])]
        x = Dense(self.features[-1])(x)
        if self.square_output:
            x = [jax.nn.softplus(x[0]), jax.nn.softplus(x[1])]
            # sq_lb = jnp.abs(x[0])
            # sq_ub = jnp.abs(x[1])
            # sq_lb = softhuber(x[0])
            # sq_ub = softhuber(x[1])
            # sq_lb = jnp.square(x[0])
            # sq_ub = jnp.square(x[1])
            # new_lb = jnp.minimum(sq_lb, sq_ub)
            # new_ub = jnp.maximum(sq_lb, sq_ub)
            # intersect_zero = x[0] * x[1]
            # new_lb = jnp.where(intersect_zero >= 0, new_lb, 0.0)
            # x = [new_lb, new_ub]

        return x


def non_neg_loss(l):
    return jnp.mean(jnp.maximum(-l, 0))


def zero_at_zero_loss(l_at_zero):
    return jnp.mean(jnp.square(l_at_zero))


def martingale_loss(l, l_next, eps):
    diff = l_next - l
    return jnp.mean(jnp.maximum(diff + eps, 0))


def np_load(filename):
    arr = onp.load(filename)
    return {k: arr[k] for k in arr.files}


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


def create_train_state(model, rng, in_dim, learning_rate):
    """Creates initial `TrainState`."""
    params = model.init(rng, jnp.ones([1, in_dim]))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


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
