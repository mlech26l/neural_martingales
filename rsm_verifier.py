import os
import time
from functools import partial

import gym.spaces

from rsm_utils import pretty_time, pretty_number, lipschitz_l1_jax

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import tensorflow as tf
import jax.numpy as jnp

from tqdm import tqdm
import numpy as np

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def get_n_for_bound_computation(obs_dim):
    if obs_dim == 2:
        n = 200
    elif obs_dim == 3:
        n = 100
    else:
        n = 50
    return n


def v_contains(box, states):
    b_low = np.expand_dims(box.low, axis=0)
    b_high = np.expand_dims(box.high, axis=0)
    contains = np.logical_and(
        np.all(states >= b_low, axis=1), np.all(states <= b_high, axis=1)
    )
    return contains


def v_intersect(box, lb, ub):
    b_low = np.expand_dims(box.low, axis=0)
    b_high = np.expand_dims(box.high, axis=0)
    contain_lb = np.logical_and(lb >= b_low, lb <= b_high)
    contain_ub = np.logical_and(ub >= b_low, ub <= b_high)
    contains_any = np.all(np.logical_or(contain_lb, contain_ub), axis=1)

    return contains_any


def jv_intersect(box, lb, ub):
    b_low = jnp.expand_dims(box.low, axis=0)
    b_high = jnp.expand_dims(box.high, axis=0)
    contain_lb = jnp.logical_and(lb >= b_low, lb <= b_high)
    contain_ub = jnp.logical_and(ub >= b_low, ub <= b_high)
    # every axis much either lb or ub contain
    contains_any = jnp.all(jnp.logical_or(contain_lb, contain_ub), axis=1)

    return contains_any


def jv_contains(box, states):
    b_low = jnp.expand_dims(box.low, axis=0)
    b_high = jnp.expand_dims(box.high, axis=0)
    contains = np.logical_and(
        jnp.all(states >= b_low, axis=1), jnp.all(states <= b_high, axis=1)
    )
    return contains


class TrainBuffer:
    def __init__(self, max_size=6_000_000):
        self.s = []
        self.max_size = max_size
        self._cached_ds = None

    def append(self, s):
        if self.max_size is not None and len(self) > self.max_size:
            return
        self.s.append(np.array(s))
        self._cached_ds = None

    def extend(self, lst):
        for s in lst:
            self.append(s)

    def __len__(self):
        if len(self.s) == 0:
            return 0
        return sum([s.shape[0] for s in self.s])

    @property
    def in_dim(self):
        return len(self.s[0])

    def as_tfds(self, batch_size=32):
        if self._cached_ds is not None:
            return self._cached_ds
        train_s = np.concatenate(self.s, axis=0)
        train_s = np.random.default_rng().permutation(train_s)
        train_ds = tf.data.Dataset.from_tensor_slices(train_s)
        train_ds = train_ds.shuffle(50000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        self._cached_ds = train_ds
        return train_ds

    def clear(self):
        self.s.clear()

    def is_empty(self):
        if len(self.s) == 0:
            return True
        return False

    def get_list(self):
        return self.s


class RSMVerifier:
    def __init__(
        self,
        rsm_learner,
        env,
        batch_size,
        reach_prob,
        fail_check_fast,
        grid_factor,
        stability_check,
        streaming_mode=False,
    ):
        self.learner = rsm_learner
        self.env = env
        self.reach_prob = jnp.float32(reach_prob)
        self.fail_check_fast = fail_check_fast
        self.stability_check = stability_check

        self.streaming_mode = streaming_mode
        self.batch_size = batch_size
        self.block_size = 8 * batch_size
        self.refinement_enabled = True

        if env.observation_space.shape[0] == 2:
            # in 2D -> default grid size is 500
            self.grid_size = int(grid_factor * 500)
        elif env.observation_space.shape[0] == 3:
            # in 3D -> default grid size is 200
            self.grid_size = int(grid_factor * 100)
        else:
            # in 4D -> default grid size is 100
            self.refinement_enabled = False
            self.grid_size = int(grid_factor * 100)

        # Precompute probability mass grid for the expectation computation
        self.pmass_n = 10  # number of sums for the expectation computation
        self._cached_pmass_grid = self.get_pmass_grid()

        self._cached_filtered_grid = None
        self._debug_violations = None
        self.hard_constraint_violation_buffer = None
        self.train_buffer = TrainBuffer()
        self._perf_stats = {
            "apply": 0.0,
            "loop": 0.0,
        }
        self.v_get_grid_item = jax.vmap(
            self.get_grid_item, in_axes=(0, None), out_axes=0
        )

    def prefill_train_buffer(self):
        """
        Fills the train buffer with a coarse grid
        """
        if self.env.observation_dim == 2:
            n = 400
        elif self.env.observation_dim == 3:
            n = 100
        else:
            n = 20
        state_grid, _, _ = self.get_unfiltered_grid(n=n)
        self.train_buffer.append(np.array(state_grid))
        return (
            self.env.observation_space.high[0] - self.env.observation_space.low[0]
        ) / n

    @partial(jax.jit, static_argnums=(0, 2))
    def get_grid_item(self, idx, n):
        """
        Maps an integer cell index and grid size to the bounds of the grid cell
        :param idx: Integer between 0 and n**obs_dim
        :param n: Grid size
        :return: jnp.ndarray corresponding to the center of the idx cell
        """
        dims = self.env.observation_dim
        target_points = [
            jnp.linspace(
                self.env.observation_space.low[i],
                self.env.observation_space.high[i],
                n,
                retstep=True,
                endpoint=False,
            )
            for i in range(dims)
        ]
        target_points, retsteps = zip(*target_points)
        target_points = list(target_points)
        for i in range(dims):
            target_points[i] = target_points[i] + 0.5 * retsteps[i]
        inds = []
        for i in range(dims):
            inds.append(idx % n)
            idx = idx // n
        return jnp.array([target_points[i][inds[i]] for i in range(dims)])

    def get_refined_grid_template(self, steps, n):
        """
        Refines a grid with resolution delta into n smaller grid cells.
        The returned template can be added to cells to create the smaller grid
        """
        dims = self.env.observation_dim
        grid, new_steps = [], []
        for i in range(dims):
            samples, new_step = jnp.linspace(
                -0.5 * steps[i],
                +0.5 * steps[i],
                n,
                endpoint=False,
                retstep=True,
            )
            grid.append(samples.flatten() + new_step * 0.5)
            new_steps.append(new_step)
        grid = jnp.meshgrid(*grid)
        grid = jnp.stack(grid, axis=1)
        return grid, np.array(new_steps)

    def get_pmass_grid(self):
        """
        Compute the bounds of the sum terms and corresponding probability masses
        for the expectation computation
        """
        dims = len(self.env.noise_bounds[0])
        grid, steps = [], []
        for i in range(dims):
            samples, step = jnp.linspace(
                self.env.noise_bounds[0][i],
                self.env.noise_bounds[1][i],
                self.pmass_n,
                endpoint=False,
                retstep=True,
            )
            grid.append(samples)
            steps.append(step)
        grid_lb = jnp.meshgrid(*grid)
        grid_lb = [x.flatten() for x in grid_lb]
        grid_ub = [grid_lb[i] + steps[i] for i in range(dims)]

        if dims < self.env.observation_dim:
            # Fill remaining dimensions with 0
            remaining = self.env.observation_dim - len(self.env.noise_bounds)
            for i in range(remaining):
                grid_lb.append(jnp.zeros_like(grid_lb[0]))
                grid_ub.append(jnp.zeros_like(grid_lb[0]))
        batched_grid_lb = jnp.stack(grid_lb, axis=1)  # stack on input  dim
        batched_grid_ub = jnp.stack(grid_ub, axis=1)  # stack on input dim
        pmass = self.env.integrate_noise(grid_lb, grid_ub)
        return pmass, batched_grid_lb, batched_grid_ub

    def get_unfiltered_grid(self, n=100):
        dims = self.env.observation_dim
        grid, steps = [], []
        for i in range(dims):
            samples, step = np.linspace(
                self.env.observation_space.low[i],
                self.env.observation_space.high[i],
                n,
                endpoint=False,
                retstep=True,
            )
            grid.append(samples)
            steps.append(step)
        grid = np.meshgrid(*grid)
        grid_lb = [x.flatten() for x in grid]
        grid_ub = [grid_lb[i] + steps[i] for i in range(dims)]
        grid_centers = [grid_lb[i] + steps[i] / 2 for i in range(dims)]

        grid_lb = np.stack(grid_lb, axis=1)
        grid_ub = np.stack(grid_ub, axis=1)
        grid_centers = np.stack(grid_centers, axis=1)
        return grid_centers, grid_lb, grid_ub

    def get_filtered_grid(self, l_params, n=100):
        # if self._cached_filtered_grid is not None:
        #     if n == self._cached_filtered_grid:
        #         print(f"Using cached grid of n={n} ", end="", flush=True)
        #         return self._cached_filtered_grid[1], self._cached_filtered_grid[2]
        #     else:
        #         self._cached_filtered_grid = None
        import gc
        gc.collect()

        print(f"Allocating grid of n={n} ", end="", flush=True)
        dims = self.env.observation_space.shape[0]
        grid, steps = [], []
        for i in range(dims):
            samples, step = np.linspace(
                self.env.observation_space.low[i],
                self.env.observation_space.high[i],
                n,
                endpoint=False,
                retstep=True,
            )
            grid.append(samples)
            steps.append(step)
        print(f" meshgrid with steps={steps} ", end="", flush=True)
        grid = np.meshgrid(*grid)
        grid = [grid[i].flatten() + steps[i] / 2 for i in range(dims)]
        grid = np.stack(grid, axis=1)

        mask = np.zeros(grid.shape[0], dtype=np.bool)
        if self.stability_check:
            mask_l_val = np.zeros(grid.shape[0], dtype=np.bool)
            for start in range(0, grid.shape[0], self.batch_size):
                end = min(start + self.batch_size, grid.shape[0])
                v_batch = self.learner.v_state.apply_fn(l_params, grid[start:end]).flatten()
                mask_l_val[start:end] = v_batch < 1
            # mask_l_val = np.reshape(self.learner.v_state.apply_fn(l_params, grid) < 1, -1)
            for unsafe_space in self.env.unsafe_spaces:
                contains = v_contains(unsafe_space, grid)
                mask = np.logical_or(
                    mask,
                    contains,
                )
            mask = np.logical_not(mask)
            mask = np.logical_and(
                mask,
                mask_l_val,
            )
        else:
            for target_space in self.env.target_spaces:
                contains = v_contains(target_space, grid)
                mask = np.logical_or(
                    mask,
                    contains,
                )

        filtered_grid = grid[np.logical_not(mask)]

        steps = np.array(steps)
        # self._cached_filtered_grid = (n, filtered_grid, steps)

        return filtered_grid, steps

    def compute_bound_init(self, n):
        """
        Computes the lower and upper bound of the RSM on the initial state set
        :param n: Discretization (a too high value will cause a long runtime or out-of-memory errors)
        """
        _, grid_lb, grid_ub = self.get_unfiltered_grid(n)

        mask = np.zeros(grid_lb.shape[0], dtype=np.bool)
        # Include if the grid cell intersects with any of the init spaces
        for init_space in self.env.init_spaces:
            intersect = v_intersect(init_space, grid_lb, grid_ub)
            mask = np.logical_or(
                mask,
                intersect,
            )
        # Exclude if both lb AND ub are in the target set
        for target_space in self.env.target_spaces:
            contains_lb = v_contains(target_space, grid_lb)
            contains_ub = v_contains(target_space, grid_ub)
            mask = np.logical_and(
                mask, np.logical_not(np.logical_and(contains_lb, contains_ub))
            )

        grid_lb = grid_lb[mask]
        grid_ub = grid_ub[mask]
        assert grid_ub.shape[0] > 0

        return self.compute_bounds_on_set(grid_lb, grid_ub)

    def compute_bound_unsafe(self, n):
        """
        Computes the lower and upper bound of the RSM on the unsafe state set
        :param n: Discretization (a too high value will cause a long runtime or out-of-memory errors)
        """
        _, grid_lb, grid_ub = self.get_unfiltered_grid(n)

        # Include only if either lb OR ub are in one of the unsafe sets
        mask = np.zeros(grid_lb.shape[0], dtype=np.bool)
        for unsafe_spaces in self.env.unsafe_spaces:
            intersect = v_intersect(unsafe_spaces, grid_lb, grid_ub)
            mask = np.logical_or(
                mask,
                intersect,
            )
        grid_lb = grid_lb[mask]
        grid_ub = grid_ub[mask]
        assert grid_ub.shape[0] > 0
        return self.compute_bounds_on_set(grid_lb, grid_ub)

    def compute_bound_domain(self, n):
        """
        Computes the lower and upper bound of the RSM on the entire state space except the target space
        :param n: Discretization (a too high value will cause a long runtime or out-of-memory errors)
        """
        _, grid_lb, grid_ub = self.get_unfiltered_grid(n)

        if not self.stability_check:
            # Exclude if both lb AND ub are in the target set
            mask = np.zeros(grid_lb.shape[0], dtype=np.bool)
            for target_space in self.env.target_spaces:
                contains_lb = v_contains(target_space, grid_lb)
                contains_ub = v_contains(target_space, grid_ub)
                mask = np.logical_or(
                    mask,
                    np.logical_and(contains_lb, contains_ub),
                )
            mask = np.logical_not(
                mask
            )  # now we have all cells that have both lb and both in a target -> invert for filtering
            grid_lb = grid_lb[mask]
            grid_ub = grid_ub[mask]
        assert grid_ub.shape[0] > 0
        return self.compute_bounds_on_set(grid_lb, grid_ub)

    def compute_bounds_on_set(self, grid_lb, grid_ub):
        """
        Computes the lower and upper bound of the RSM with respect to the given discretization
        """
        global_min = jnp.inf
        global_max = jnp.NINF
        for i in range(int(np.ceil(grid_ub.shape[0] / self.batch_size))):
            start = i * self.batch_size
            end = np.minimum((i + 1) * self.batch_size, grid_ub.shape[0])
            batch_lb = jnp.array(grid_lb[start:end])
            batch_ub = jnp.array(grid_ub[start:end])
            lb, ub = self.learner.v_ibp.apply(
                self.learner.v_state.params, [batch_lb, batch_ub]
            )
            global_min = jnp.minimum(global_min, jnp.min(lb))
            global_max = jnp.maximum(global_max, jnp.max(ub))
        return float(global_min), float(global_max)

    @partial(jax.jit, static_argnums=(0,))
    def compute_expected_l(self, params, s, a, pmass, batched_grid_lb, batched_grid_ub):
        """
        Compute kernel (jit compiled) that computes an upper bounds on the expected value of L(s next)
        """
        deterministic_s_next = self.env.v_next(s, a)
        batch_size = s.shape[0]
        ibp_size = batched_grid_lb.shape[0]
        obs_dim = self.env.observation_dim

        # Broadcasting happens here, that's why we don't do directly vmap (although it's probably possible somehow)
        deterministic_s_next = deterministic_s_next.reshape((batch_size, 1, obs_dim))
        batched_grid_lb = batched_grid_lb.reshape((1, ibp_size, obs_dim))
        batched_grid_ub = batched_grid_ub.reshape((1, ibp_size, obs_dim))

        batched_grid_lb = batched_grid_lb + deterministic_s_next
        batched_grid_ub = batched_grid_ub + deterministic_s_next

        batched_grid_lb = batched_grid_lb.reshape((-1, obs_dim))
        batched_grid_ub = batched_grid_ub.reshape((-1, obs_dim))
        lb, ub = self.learner.v_ibp.apply(params, [batched_grid_lb, batched_grid_ub])
        ub = ub.reshape((batch_size, ibp_size))

        pmass = pmass.reshape((1, ibp_size))  # Boradcast to batch size
        exp_terms = pmass * ub
        expected_value = jnp.sum(exp_terms, axis=1)
        return expected_value

    @partial(jax.jit, static_argnums=(0,))
    def _check_dec_batch(self, l_params, p_params, f_batch, l_batch, K):
        """
        Compute kernel (jit compiled) that checks if a batch of grid cells violate the decrease conditions
        """
        a_batch = self.learner.p_state.apply_fn(p_params, f_batch)
        pmass, batched_grid_lb, batched_grid_ub = self._cached_pmass_grid
        e = self.compute_expected_l(
            l_params,
            f_batch,
            a_batch,
            pmass,
            batched_grid_lb,
            batched_grid_ub,
        )
        decrease = e + K - l_batch
        violating_indices = decrease >= 0
        v = violating_indices.astype(jnp.int32).sum()
        hard_violating_indices = e - l_batch >= 0
        hard_v = hard_violating_indices.astype(jnp.int32).sum()
        return v, violating_indices, hard_v, hard_violating_indices, jnp.max(decrease)

    @partial(jax.jit, static_argnums=(0,))
    def normalize_rsm(self, l, ub_init, domain_min):
        """
        By normalizing the RSM using the global infimum of L and the infimum of L within the init set, we
        improve the Reach-avoid bounds of L slightly.
        """
        l = l - domain_min
        ub_init = ub_init - domain_min
        # now min = 0
        l = l / jnp.maximum(ub_init, 1e-6)
        # now init max = 1
        return l

    def check_dec_cond(self, lipschitz_k):
        """
        This method checks if the decrease condition is fulfilled.
        How the grid is processed (block-wise or allocating the entire grid) is decided by the streaming_mode flag
        :param lipschitz_k: Lipschitz constant of the entire system (environment, policy, RSM)
        :return: Number of violating grid cells, and number of hard violating grid cells
        """
        if self.streaming_mode:
            return self.check_dec_cond_with_stream(lipschitz_k)
        else:
            return self.check_dec_cond_full(lipschitz_k)

    def check_dec_cond_full(self, lipschitz_k):
        """
        This method checks if the decrease condition is fulfilled by creating the allocating grid in memory first.
        This is fast but may require a lot of memory caused out-of-memory errors.
        If such error occur, consider streaming mode, which creates sub-blocks of the grid on-deman
        :param lipschitz_k: Lipschitz constant of the entire system (environment, policy, RSM)
        :return: Number of violating grid cells, and number of hard violating grid cells
        """
        dims = self.env.observation_dim
        grid_total_size = self.grid_size ** dims

        verify_start_time = time.time()
        n = get_n_for_bound_computation(self.env.observation_dim)
        _, ub_init = self.compute_bound_init(n)
        domain_min, _ = self.compute_bound_domain(n)

        grid, steps = self.get_filtered_grid(
            self.learner.v_state.params,
            self.grid_size,
        )
        delta = 0.5 * np.sum(
            steps
        )  # l1-norm of the half the grid cell (=l1 distance from center to corner)
        K = lipschitz_k * delta
        number_of_cells = self.grid_size ** self.env.observation_dim
        print(
            f"Checking GRID with {pretty_number(number_of_cells)} cells and K={K:0.3g}"
        )
        K = jnp.float32(K)

        violations = 0
        hard_violations = 0
        max_decrease = jnp.NINF
        violation_buffer = []
        hard_violation_buffer = []
        # self.train_buffer.clear()
        # self.prefill_train_buffer()
        max_hard = -1

        # block_size size should not be too large
        kernel_start_time = time.perf_counter()
        pbar = tqdm(total=grid.shape[0], unit="cells")
        for start in range(0, grid.shape[0], self.batch_size):
            pbar.update(1)

            end = min(start + self.batch_size, grid.shape[0])
            x_batch = jnp.array(grid[start:end])
            v_batch = self.learner.v_state.apply_fn(
                self.learner.v_state.params, x_batch
            ).flatten()

            if self.reach_prob < 1.0 and not self.stability_check:
                # normalize the RSM to obtain slightly better values
                normalized_l_batch = self.normalize_rsm(v_batch, ub_init, domain_min)
                # Next, we filter the grid cells that are > 1/(1-p)
                less_than_p = normalized_l_batch - K < 1 / (1 - self.reach_prob)
                if jnp.sum(less_than_p.astype(np.int32)) == 0:
                    # If all cells are filtered -> can skip the expectation computation
                    continue
                x_batch = x_batch[less_than_p]
                v_batch = v_batch[less_than_p]

            # Finally, we compute the expectation of the grid cell
            (
                v,
                violating_indices,
                hard_v,
                hard_violating_indices,
                decrease,
            ) = self._check_dec_batch(
                self.learner.v_state.params,
                self.learner.p_state.params,
                x_batch,
                v_batch,
                K,
            )
            max_decrease = jnp.maximum(max_decrease, decrease)
            # Count the number of violations and hard violations
            hard_violations += hard_v
            violations += v
            if v > 0:
                violation_buffer.append(x_batch[violating_indices])
            if hard_v > 0:
                max_hard = jnp.maximum(max_hard, jnp.max(v_batch[hard_violating_indices]))
                hard_violation_buffer.append(np.array(x_batch[hard_violating_indices]))
            total_kernel_time = time.perf_counter() - kernel_start_time
            total_cells = start + self.batch_size
            kcells_per_sec = total_cells / total_kernel_time / 1000
            pbar.set_description(
                f"{pretty_number(violations)}/{pretty_number(total_cells)} cell violating @ {kcells_per_sec:0.1f} Kcells/s"
            )
            if self.fail_check_fast and violations > 0:
                break
        pbar.close()

        # self.train_buffer.extend(hard_violation_buffer)
        if self.stability_check:
            buffer = self.train_buffer.get_list().copy()
            self.train_buffer.clear()
            # self.train_buffer.extend(violation_buffer)
            # if self.train_buffer.is_empty():
            #     self.train_buffer.append(grid)
            # buffer = np.reshape(buffer, (-1, dims))
            # tmp_s = []
            for el in buffer:
                val = self.learner.v_state.apply_fn(self.learner.v_state.params, el)
                val = np.reshape(val, -1)
                final_val = el[val >= 1]
                if len(final_val) > 0:
                    if len(final_val.shape) > 2:
                        final_val = final_val[0]
                    self.train_buffer.append(final_val)
                # if val >= 1:
                #     tmp_s.append(el)

            # self.train_buffer.clear()
            # self.train_buffer.extend(tmp_s)
        # else:
        self.train_buffer.extend(violation_buffer)
        if self.train_buffer.is_empty():
            self.train_buffer.append(grid)
        #     self.train_buffer.extend(grid[::10])
        if hard_violations > 0:
            buffer_hard = [np.array(g) for g in hard_violation_buffer]
            buffer_hard = np.concatenate(buffer_hard)
            self.hard_constraint_violation_buffer = buffer_hard
        else:
            self.hard_constraint_violation_buffer = None
        print(
            f"Verified {pretty_number(grid_total_size)} cells ({pretty_number(violations)} violations, {pretty_number(hard_violations)} hard) in {pretty_time(time.time()-verify_start_time)}"
        )

        # v_tr = 500000 if dims == 2 else grid.shape[0]
        if (
            self.refinement_enabled
            and hard_violations == 0
            and violations > 0
            and len(violation_buffer) > 0
            # and violations <= v_tr
        ):
            print(
                f"Zero hard violations -> refinement of {pretty_number(grid_total_size)} soft violations"
            )
            refine_start = time.time()
            refinement_buffer = [np.array(g) for g in violation_buffer]
            refinement_buffer = np.concatenate(refinement_buffer)
            success, max_decrease = self.refine_grid(
                refinement_buffer, lipschitz_k, steps, ub_init, domain_min
            )
            if success:
                print(
                    f"Refinement successful! (took {pretty_time(time.time()-refine_start)})"
                )
                return 0, 0, max_decrease, max_hard
            else:
                print(
                    f"Refinement unsuccessful! (took {pretty_time(time.time()-refine_start)})"
                )

        return violations, hard_violations, max_decrease, max_hard

    def check_dec_cond_with_stream(self, lipschitz_k):
        """
        This method checks if the decrease condition is fulfilled by created sub-grid block-wise and checking them
        This is slower but has a much smaller memory footprint than the full allocation method
        :param lipschitz_k: Lipschitz constant of the entire system (environment, policy, RSM)
        :return: Number of violating grid cells, and number of hard violating grid cells
        """
        dims = self.env.observation_dim
        grid_total_size = self.grid_size ** dims

        verify_start_time = time.time()
        n = get_n_for_bound_computation(self.env.observation_dim)
        _, ub_init = self.compute_bound_init(n)
        domain_min, _ = self.compute_bound_domain(n)

        steps = (self.env.observation_space.high - self.env.observation_space.low) / (
            self.grid_size - 1
        )
        # l1-norm of the half the grid cell (=l1 distance from center to corner)
        delta = 0.5 * np.sum(steps)
        K = lipschitz_k * delta
        max_decrease = jnp.NINF
        number_of_cells = self.grid_size ** self.env.observation_dim
        print(
            f"Checking GRID with {pretty_number(number_of_cells)} cells and K={K:0.3g}"
        )
        K = jnp.float32(K)

        violations = 0
        hard_violations = 0
        violation_buffer = []
        hard_violation_buffer = []
        max_hard = -1

        # block_size size should not be too large
        block_size = min(grid_total_size, self.block_size)
        kernel_start_time = time.perf_counter()
        total_cells = 0
        pbar = tqdm(total=grid_total_size // block_size, unit="blocks")
        for block_id in range(0, grid_total_size, block_size):
            # Create array of indices of the grid cell in the current block
            idx = jnp.arange(block_id, min(grid_total_size, block_id + block_size))
            # Map indices -> centers of the cells
            sub_grid = self.v_get_grid_item(idx, self.grid_size)
            # Filter out grid cells that are inside in at least one target set
            contains = jnp.ones(sub_grid.shape[0], dtype=np.bool)
            for target_space in self.env.target_spaces:
                c = jv_contains(target_space, sub_grid)
                contains = jnp.logical_or(c)
            sub_grid = sub_grid[jnp.logical_not(contains)]

            # We process the block in batches
            for start in range(0, sub_grid.shape[0], self.batch_size):
                end = min(start + self.batch_size, sub_grid.shape[0])
                x_batch = jnp.array(sub_grid[start:end])
                v_batch = self.learner.v_state.apply_fn(
                    self.learner.v_state.params, x_batch
                ).flatten()
                # normalize the RSM to obtain slightly better values
                normalized_l_batch = self.normalize_rsm(v_batch, ub_init, domain_min)

                # Next, we filter the grid cells that are > 1/(1-p)
                if self.reach_prob < 1.0 and not self.stability_check:
                    less_than_p = normalized_l_batch - K < 1 / (1 - self.reach_prob)
                    if jnp.sum(less_than_p.astype(np.int32)) == 0:
                        # If all cells are filtered -> can skip the expectation computation
                        continue
                    x_batch = x_batch[less_than_p]
                    v_batch = v_batch[less_than_p]

                # Finally, we compute the expectation of the grid cell
                (
                    v,
                    violating_indices,
                    hard_v,
                    hard_violating_indices,
                    decrease,
                ) = self._check_dec_batch(
                    self.learner.v_state.params,
                    self.learner.p_state.params,
                    x_batch,
                    v_batch,
                    K,
                )
                max_decrease = jnp.maximum(max_decrease, decrease)
                # Count the number of violations and hard violations
                hard_violations += hard_v
                violations += v
                if v > 0:
                    violation_buffer.append(x_batch[violating_indices])
                if hard_v > 0:
                    max_hard = jnp.maximum(max_hard, jnp.max(v_batch[hard_violating_indices]))
                    hard_violation_buffer.append(x_batch[hard_violating_indices])
            pbar.update(1)
            total_kernel_time = time.perf_counter() - kernel_start_time
            total_cells += sub_grid.shape[0]
            kcells_per_sec = total_cells / total_kernel_time / 1000
            pbar.set_description(
                f"{pretty_number(violations)}/{pretty_number(total_cells)} cell violating @ {kcells_per_sec:0.1f} Kcells/s"
            )
            if self.fail_check_fast and violations > 0:
                break
        pbar.close()

        # self.train_buffer.extend(hard_violation_buffer)
        self.train_buffer.extend(violation_buffer)
        print(
            f"Verified {pretty_number(grid_total_size)} cells ({pretty_number(violations)} violations, {pretty_number(hard_violations)} hard) in {pretty_time(time.time()-verify_start_time)}"
        )

        if (
            self.refinement_enabled
            and hard_violations == 0
            and violations > 0
            and len(violation_buffer) > 0
        ):
            print(
                f"Zero hard violations -> refinement of {pretty_number(grid_total_size)} soft violations"
            )
            refine_start = time.time()
            refinement_buffer = [np.array(g) for g in violation_buffer]
            refinement_buffer = np.concatenate(refinement_buffer)
            success, max_decrease = self.refine_grid(
                refinement_buffer, lipschitz_k, steps, ub_init, domain_min
            )
            if success:
                print(
                    f"Refinement successful! (took {pretty_time(time.time()-refine_start)})"
                )
                return 0, 0, max_decrease, max_hard
            else:
                print(
                    f"Refinement unsuccessful! (took {pretty_time(time.time()-refine_start)})"
                )

        return violations, hard_violations, max_decrease, max_hard

    def refine_grid(self, refinement_buffer, lipschitz_k, steps, ub_init, domain_min):
        n_dims = self.env.observation_dim

        n = 10
        if self.env.observation_dim > 2:
            n = 6
        template_batch, new_steps = self.get_refined_grid_template(steps, n)
        new_delta = 0.5 * np.sum(
            new_steps
        )  # l1-norm of half the cell len (= distance from center to corner)

        # Refinement template has an extra dimension we need to consider
        batch_size = self.batch_size // template_batch.shape[0]

        # print(f"lipschitz_k={lipschitz_k}")
        # print(f"current_delta={current_delta}")
        # print(f"new_delta={new_delta}")
        # new_delta = current_delta / (n - 1)
        K = jnp.float32(lipschitz_k * new_delta)
        template_batch = template_batch.reshape((1, -1, n_dims))
        max_decrease = jnp.NINF
        for i in tqdm(range(int(np.ceil(refinement_buffer.shape[0] / batch_size)))):
            start = i * batch_size
            end = np.minimum((i + 1) * batch_size, refinement_buffer.shape[0])
            s_batch = jnp.array(refinement_buffer[start:end])
            s_batch = s_batch.reshape((-1, 1, n_dims))
            r_batch = s_batch + template_batch
            r_batch = r_batch.reshape((-1, self.env.observation_dim))  # flatten

            l_batch = self.learner.v_state.apply_fn(
                self.learner.v_state.params, r_batch
            ).flatten()
            if not self.stability_check:
                normalized_l_batch = self.normalize_rsm(l_batch, ub_init, domain_min)
                less_than_p = normalized_l_batch - K < 1 / (1 - self.reach_prob)
                if jnp.sum(less_than_p.astype(np.int32)) == 0:
                    continue
                r_batch = r_batch[less_than_p]
                l_batch = l_batch[less_than_p]

            (
                v,
                violating_indices,
                hard_v,
                hard_violating_indices,
                decrease,
            ) = self._check_dec_batch(
                self.learner.v_state.params,
                self.learner.p_state.params,
                r_batch,
                l_batch,
                K,
            )
            max_decrease = jnp.maximum(max_decrease, decrease)
            if v > 0:
                return False, max_decrease
        return True, max_decrease

    def get_unsafe_complement_grid(self, n=100):
        print(f"Allocating grid of n={n} ", end="", flush=True)
        dims = self.env.observation_space.shape[0]
        grid, steps = [], []
        for i in range(dims):
            samples, step = np.linspace(
                self.env.observation_space.low[i],
                self.env.observation_space.high[i],
                n,
                endpoint=False,
                retstep=True,
            )
            grid.append(samples)
            steps.append(step)
        print(f" meshgrid with steps={steps} ", end="", flush=True)
        grid = np.meshgrid(*grid)
        grid = [grid[i].flatten() + steps[i] / 2 for i in range(dims)]
        grid = np.stack(grid, axis=1)

        mask = np.zeros(grid.shape[0], dtype=np.bool)
        for unsafe_space in self.env.unsafe_spaces:
            contains = v_contains(unsafe_space, grid)
            mask = np.logical_or(
                mask,
                contains,
            )

        filtered_grid = grid[np.logical_not(mask)]

        return filtered_grid, np.array(steps)

    def get_mask_in_and_safe(self, grid):
        mask = np.zeros(grid.shape[0], dtype=np.bool)
        for unsafe_space in self.env.unsafe_spaces:
            contains = v_contains(unsafe_space, grid)
            mask = np.logical_or(mask, contains)

        contains = v_contains(self.env.observation_space, grid)
        mask = np.logical_or(mask, np.logical_not(contains))
        return np.logical_not(mask)

    def get_unsafe_d_grid(self, d, n=100):
        print(f"Allocating grid of n={n} ", end="", flush=True)
        dims = self.env.observation_space.shape[0]
        grid, steps = [], []
        for i in range(dims):
            samples, step = np.linspace(
                self.env.observation_space.low[i],
                self.env.observation_space.high[i],
                n,
                endpoint=False,
                retstep=True,
            )
            grid.append(samples)
            steps.append(step)
        print(f" meshgrid with steps={steps} ", end="", flush=True)
        grid = np.meshgrid(*grid)
        grid = [grid[i].flatten() + steps[i] / 2 for i in range(dims)]
        grid = np.stack(grid, axis=1)

        mask = np.zeros(grid.shape[0], dtype=np.bool)
        for unsafe_space in self.env.unsafe_spaces:
            contains = v_contains(unsafe_space, grid)
            mask = np.logical_or(
                mask,
                contains,
            )

        filtered_grid = grid[mask]

        mask = np.zeros(filtered_grid.shape[0], dtype=np.bool)

        for i in range(dims):
            sh = np.zeros(dims, dtype=np.float)
            sh[i] = d

            grid_n = filtered_grid - sh
            grid_p = filtered_grid + sh

            mask = np.logical_or(mask, self.get_mask_in_and_safe(grid_n))
            mask = np.logical_or(mask, self.get_mask_in_and_safe(grid_p))

        filtered_grid = filtered_grid[mask]

        return filtered_grid, np.array(steps)

    def get_big_delta(self):
        grids, steps = self.get_unsafe_complement_grid(self.grid_size)
        p_params = self.learner.p_state.params

        delta = 0.5 * jnp.sum(
            steps
        )
        lip_f = self.env.lipschitz_constant
        lip_p = lipschitz_l1_jax(p_params)
        lip_norm = lip_f * (lip_p + 1) * delta

        next_det_grids = self.env.v_next(grids, self.learner.p_state.apply_fn(p_params, grids))
        next_max_norm = jnp.max(jnp.sum(jnp.abs(next_det_grids - grids), axis=1))

        noise_norm = jnp.sum(jnp.abs(self.env.noise))

        big_delta = next_max_norm + noise_norm + lip_norm

        print(
            "\n {Getting the big delta, next_max_norm: ",
            next_max_norm,
            ", noise norm: ",
            noise_norm,
            ", lip norm: ",
            lip_norm,
            "\n"

        )
        return big_delta

    def get_m_d(self, d):
        grids, steps = self.get_unsafe_d_grid(d, self.grid_size)

        l_params = self.learner.v_state.params

        delta = 0.5 * np.sum(steps)
        lip_l = lipschitz_l1_jax(l_params)

        l_grids = self.learner.v_state.apply_fn(l_params, grids)

        return np.max(l_grids) + delta * lip_l
