import argparse
import os
import sys

import jax.random
import pylab as pl
from gym import spaces
from tqdm import tqdm

from rsm_utils import lipschitz_l1_jax, triangular, pretty_time, pretty_number
from rl_environments import LDSEnv, InvertedPendulum, CollisionAvoidanceEnv, CollisionAvoidance3DEnv, vDroneEnv
from rsm_learner import RSMLearner
from rsm_verifier import RSMVerifier, get_n_for_bound_computation
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class RSMLoop:
    def __init__(
        self,
        learner,
        verifier,
        env,
        plot,
        soft_constraint,
        train_p=True,
    ):
        self.env = env
        self.learner = learner
        self.verifier = verifier
        self.train_p = train_p
        self.soft_constraint = soft_constraint

        os.makedirs("saved", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        os.makedirs("loop", exist_ok=True)
        self.plot = plot
        self.prefill_delta = 0
        self.iter = 0
        self.info = {}

    def learn(self):
        train_ds = self.verifier.train_buffer.as_tfds(batch_size=4096)
        current_delta = self.prefill_delta
        start_metrics = None
        num_epochs = (
            50 if self.iter > 0 else 200
        )  # in the first iteration we train a bit longer

        current_alpha = self.learner.alpha_min
        current_alpha += (
            1 / 2 ** self.iter * (self.learner.alpha_max - current_alpha)
            if self.iter < 5 else 0
        )

        # current_big_delta = self.verifier.get_big_delta()

        start_time = time.time()
        pbar = tqdm(total=num_epochs, unit="epochs")
        for epoch in range(num_epochs):
            # in the first 3 iterations we only train the RSM
            train_p = self.iter >= 3 and self.train_p

            # we always train the RSM
            train_v = True
            metrics = self.learner.train_epoch(
                train_ds, current_delta, current_alpha, train_v, train_p
            )
            if start_metrics is None:
                start_metrics = metrics
            pbar.update(1)
            pbar.set_description_str(
                f"Train [v={train_v}, p={train_p}]: loss={metrics['loss']:0.3g}, dec_loss={metrics['dec_loss']:0.3g}, violations={metrics['train_violations']:0.3g}"
            )
        pbar.close()
        self.info["ds_size"] = len(self.verifier.train_buffer)

        training_time = pretty_time(time.time() - start_time)

        print(
            f"Trained on {pretty_number(len(self.verifier.train_buffer))} samples, start_loss={start_metrics['loss']:0.3g}, end_loss={metrics['loss']:0.3g}, start_violations={start_metrics['train_violations']:0.3g}, end_violations={metrics['train_violations']:0.3g} in {training_time}"
        )

    def check_decrease_condition(self):
        K_f = self.env.lipschitz_constant
        K_p = lipschitz_l1_jax(self.learner.p_state.params).item()
        K_l = lipschitz_l1_jax(self.learner.v_state.params).item()

        lipschitz_k = K_l * K_f * (1 + K_p) + K_l
        lipschitz_k = float(lipschitz_k)
        self.log(lipschitz_k=lipschitz_k)
        self.log(K_p=K_p)
        self.log(K_f=K_f)
        self.log(K_l=K_l)
        violations, hard_violations, max_decrease, max_hard = self.verifier.check_dec_cond(
            lipschitz_k
        )
        self.log(max_decrease=max_decrease)
        self.log(max_hard=max_hard)

        if violations == 0:
            return True, max_decrease
        if hard_violations == 0 and self.iter > 4 and self.iter % 2 == 0:
            print("Refining grid")
            if self.env.observation_space.shape[0] == 2:
                # in 2D -> double grid size
                self.verifier.grid_size *= 2
            elif self.env.observation_space.shape[0] == 3:
                # in 3D -> increase grid size by 50%
                if int(1.5 * self.verifier.grid_size) <= 160:
                    self.verifier.grid_size = int(1.5 * self.verifier.grid_size)
            else:
                # increase grid size by 30%
                self.verifier.grid_size = int(1.3 * self.verifier.grid_size)

        return False, max_decrease

    def verify(self):

        dec_sat, max_decrease = self.check_decrease_condition()

        n = get_n_for_bound_computation(self.env.observation_dim)
        _, ub_init = self.verifier.compute_bound_init(n)
        lb_unsafe, _ = self.verifier.compute_bound_unsafe(n)
        lb_domain, _ = self.verifier.compute_bound_domain(n)
        self.log(ub_init=ub_init)
        self.log(lb_unsafe=lb_unsafe)
        self.log(lb_domain=lb_domain)

        if dec_sat:
            print("Decrease condition fulfilled!")
            self.learner.save(f"saved/{self.env.name}_loop.jax")
            print("[SAVED]")

            if not self.verifier.stability_check:

                if lb_unsafe < ub_init:
                    print(
                        "WARNING: RSM is lower at unsafe than in init. No Reach-avoid guarantees can be obtained."
                    )

                # normalize to lb_domain -> 0
                ub_init = ub_init - lb_domain
                lb_unsafe = lb_unsafe - lb_domain

                # normalize to ub_init -> 1
                lb_unsafe = lb_unsafe / ub_init
                actual_reach_prob = 1 - 1 / np.clip(lb_unsafe, 1e-9, None)
                self.log(actual_reach_prob=actual_reach_prob)
                if self.soft_constraint or actual_reach_prob >= self.verifier.reach_prob:
                    return actual_reach_prob
            else:
                lip_l = lipschitz_l1_jax(self.learner.v_state.params).item()
                big_d = self.verifier.get_big_delta()
                self.log(big_d=big_d)
                if lb_unsafe <= 1 + lip_l * big_d:
                    print(
                        "{Unsafe states are not greater than the bound. the actual lb: ",
                        lb_unsafe,
                        ", the desired lower bound: ",
                        1 + lip_l * self.verifier.get_big_delta(),
                        ", lip_v: ",
                        lip_l,
                        ", big delta: ",
                        self.verifier.get_big_delta(),
                        "}"
                    )
                    return None
                p = (1 + lip_l * big_d - lb_domain) / (lb_unsafe - lb_domain)
                self.log(p=p)
                if self.plot:
                    self.plot_stability_time_contour(p, -max_decrease, big_d, lb_domain, f"plots/{self.env.name}_contour_lines.pdf")
                return 1
        return None

    def log(self, **kwargs):
        for k, v in kwargs.items():
            self.info[k] = v

    def run(self, timeout):

        start_time = time.time()
        last_saved = time.time()
        self.prefill_delta = self.verifier.prefill_train_buffer()
        while True:
            runtime = time.time() - start_time
            self.log(runtime=runtime)
            self.log(iter=self.iter)

            if runtime > timeout:
                print("Timeout!")
                self.learner.save(f"saved/{self.env.name}_loop.jax")
                return False
            if time.time() - last_saved > 60 * 60:
                # save every hour
                last_saved = time.time()
                self.learner.save(f"saved/{self.env.name}_loop.jax")
                print("[SAVED]")

            print(
                f"\n#### Iteration {self.iter} (runtime: {pretty_time(runtime)}) #####"
            )
            self.learn()

            actual_reach_prob = self.verify()

            print("Log=", str(self.info))
            sys.stdout.flush()

            if actual_reach_prob is not None:
                print(
                    f"Probability of reaching the target safely is at least {actual_reach_prob * 100:0.3f}% (higher is better)"
                )
                runtime = time.time() - start_time
                self.log(runtime=runtime)
                print("runtime: ", runtime)
                return True

            if self.plot:
                if self.iter == 0:
                    self.plot_l(f"loop/{self.env.name}_{self.iter:04d}", plot_rsm=False, save=True)
                else:
                    self.plot_l(f"loop/{self.env.name}_{self.iter:04d}", plot_hard=True)
                    # self.plot_l(f"loop/{self.env.name}_{self.iter:04d}_nohard.png", False)
            self.iter += 1

    def rollout(self):
        rng = np.random.default_rng().integers(0, 10000)
        rng = jax.random.PRNGKey(rng)
        safe = np.array([0.2, 0.2], np.float32)
        space = spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            dtype=np.float32,
        )

        state = space.sample()
        trace = [np.array(state)]
        for i in range(100):
            action = self.learner.p_state.apply_fn(self.learner.p_state.params, state)
            next_state = self.env.next(state, action)
            rng, seed = jax.random.split(rng)
            noise = triangular(rng, (self.env.observation_dim,))
            noise = noise * self.env.noise
            state = next_state + noise
            trace.append(np.array(state))
        return np.stack(trace, axis=0)

    def plot_l(self, filename, form="png", plot_rsm=True, plot_hard=False, plot_kernel=False, save=False):
        if self.env.observation_dim > 3:
            return

        grid, _, _ = self.verifier.get_unfiltered_grid(n=50)
        l = self.learner.v_state.apply_fn(self.learner.v_state.params, grid).flatten()
        l = np.array(l)
        # l = np.clip(l, 0, 5)
        if save:
            lb_domain, _ = self.verifier.compute_bound_domain(
                get_n_for_bound_computation(self.env.observation_dim))
            np.savez(filename + ".npz", grid=grid, l=l, lb_domain=lb_domain)
        for j in range(self.env.observation_dim):
            for i in range(j):
                if plot_rsm:
                    sns.set()
                    fig, ax = plt.subplots(figsize=(6, 6))
                    sc = ax.scatter(grid[:, i], grid[:, j], marker="s", c=l, zorder=1, alpha=0.7)
                    fig.colorbar(sc)
                    ax.set_title(f"L at iter {self.iter} for {self.env.name}")

                    terminals_x, terminals_y = [], []
                    for rol in range(20):
                        trace = self.rollout()
                        ax.plot(
                            trace[:, i],
                            trace[:, j],
                            color=sns.color_palette()[0],
                            zorder=2,
                            alpha=0.3,
                        )
                        ax.scatter(
                            trace[:, i],
                            trace[:, j],
                            color=sns.color_palette()[0],
                            zorder=2,
                            marker=".",
                        )
                        terminals_x.append(float(trace[-1, i]))
                        terminals_y.append(float(trace[-1, j]))
                    ax.scatter(terminals_x, terminals_y, color="white", marker="x", zorder=5)
                    if self.verifier.hard_constraint_violation_buffer is not None and plot_hard:
                        self.verifier.hard_constraint_violation_buffer = np.asarray(self.verifier.hard_constraint_violation_buffer)
                        self.verifier.hard_constraint_violation_buffer = np.reshape(self.verifier.hard_constraint_violation_buffer, (-1, self.env.observation_dim))
                        print(
                            "self.verifier.hard_constraint_violation_buffer: ",
                            self.verifier.hard_constraint_violation_buffer[0:10],
                        )
                        ax.scatter(
                            self.verifier.hard_constraint_violation_buffer[:, i],
                            self.verifier.hard_constraint_violation_buffer[:, j],
                            color="green",
                            marker="s",
                            alpha=0.7,
                            zorder=6,
                        )
                    if self.verifier._debug_violations is not None:
                        ax.scatter(
                            self.verifier._debug_violations[:, i],
                            self.verifier._debug_violations[:, j],
                            color="cyan",
                            marker="s",
                            alpha=0.7,
                            zorder=6,
                        )
                    if not self.verifier.stability_check:
                        for init in self.env.init_spaces:
                            x = [
                                init.low[i],
                                init.high[i],
                                init.high[i],
                                init.low[i],
                                init.low[i],
                            ]
                            y = [
                                init.low[j],
                                init.low[j],
                                init.high[j],
                                init.high[j],
                                init.low[j],
                            ]
                            ax.plot(x, y, color="cyan", alpha=0.5, zorder=7)
                    for unsafe in self.env.unsafe_spaces:
                        x = [
                            unsafe.low[i],
                            unsafe.high[i],
                            unsafe.high[i],
                            unsafe.low[i],
                            unsafe.low[i],
                        ]
                        y = [
                            unsafe.low[j],
                            unsafe.low[j],
                            unsafe.high[j],
                            unsafe.high[j],
                            unsafe.low[j],
                        ]
                        ax.plot(x, y, color="magenta", alpha=0.5, zorder=7)
                    for target_space in self.env.target_spaces:
                        x = [
                            target_space.low[i],
                            target_space.high[i],
                            target_space.high[i],
                            target_space.low[i],
                            target_space.low[i],
                        ]
                        y = [
                            target_space.low[j],
                            target_space.low[j],
                            target_space.high[j],
                            target_space.high[j],
                            target_space.low[j],
                        ]
                        ax.plot(x, y, color="green", alpha=0.5, zorder=7)

                    ax.set_xlim(
                        [self.env.observation_space.low[i], self.env.observation_space.high[i]]
                    )
                    ax.set_ylim(
                        [self.env.observation_space.low[j], self.env.observation_space.high[j]]
                    )
                    fig.tight_layout()
                    fig.savefig(filename + "_dims_" + str(i) + "_" + str(j) + "." + form)
                    plt.close(fig)

                if plot_kernel:
                    sns.set()
                    fig, ax = plt.subplots(figsize=(6, 6))

                    kernel = grid[l < 1]

                    ax.scatter(
                        kernel[:, i],
                        kernel[:, j],
                        s=50,
                        color="green",
                        marker="s",
                        alpha=0.7,
                        zorder=1,
                    )

                    terminals_x, terminals_y = [], []
                    for rol in range(20):
                        trace = self.rollout()
                        ax.plot(
                            trace[:, i],
                            trace[:, j],
                            color=sns.color_palette()[0],
                            zorder=2,
                            alpha=0.3,
                        )
                        ax.scatter(
                            trace[:, i],
                            trace[:, j],
                            color=sns.color_palette()[0],
                            zorder=2,
                            marker=".",
                        )
                        terminals_x.append(float(trace[-1, i]))
                        terminals_y.append(float(trace[-1, j]))
                    ax.scatter(terminals_x, terminals_y, color="red", marker="x", zorder=5)

                    ax.set_xlim(
                        [self.env.observation_space.low[i], self.env.observation_space.high[i]]
                    )
                    ax.set_ylim(
                        [self.env.observation_space.low[j], self.env.observation_space.high[j]]
                    )
                    plt.xlabel('x1')
                    plt.ylabel('x2')
                    fig.tight_layout()
                    fig.savefig(filename + "_dims_" + str(i) + "_" + str(j) + "_kernel." + form)
                    plt.close(fig)

                # if plot_rsm_3d:
                #     sns.set()
                #     fig, axes = plt.subplots(ncols=2, figsize=(9, 4), subplot_kw={"projection": "3d"})
                #
                #     palette = sns.color_palette("icefire", as_cmap=True)
                #     surf = axes[0].plot_surface(
                #         grid[:, 0].reshape((50, 50)),
                #         grid[:, 1].reshape((50, 50)),
                #         l.reshape((50, 50)) - np.min(l),
                #         linewidth=0,
                #         antialiased=False,
                #         cmap=palette,
                #         alpha=1.0,
                #     )
                #     fig.tight_layout()
                #     fig.subplots_adjust(right=0.8)
                #     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                #     fig.colorbar(surf, cax=cbar_ax, shrink=0.5, aspect=5)
                #     # fig.colorbar(surf, shrink=0.7, aspect=5)
                #     # fig.colorbar(surf, shrink=0.5, aspect=5)
                #     axes[0].view_init(elev=30.0, azim=105)
                #     axes[1].view_init(elev=30.0, azim=105)
                #     # fig.colorbar(palette)
                #     axes[0].set_title(f"Iteration 1")
                #     axes[1].set_title(f"Iteration 2")
                #     # fig.tight_layout()
                #     fig.set_facecolor("white")
                #     axes[0].set_facecolor("white")
                #     axes[1].set_facecolor("white")
                #     fig.savefig(filename + "_dims_" + str(i) + "_" + str(j) + "_rsm_3d." + form)
                #     plt.close(fig)



    def plot_stability_time_contour(self, p, eps, big_d, lb_domain, filename):
        if self.env.observation_dim > 2:
            return
        m_d = self.verifier.get_m_d(big_d)
        n = 100

        states, _, _ = self.verifier.get_unfiltered_grid(n=n)
        l = self.learner.v_state.apply_fn(self.learner.v_state.params, states).flatten()
        stab_exp = np.array(((l - lb_domain) + (p / (1 - p)) * (m_d - lb_domain)) / eps)



        plt.figure(figsize=(6, 6))

        contours = plt.contour(np.reshape(states[:, 0], (n, n)), np.reshape(states[:, 1], (n, n)), np.reshape(stab_exp, (n, n)))
        plt.clabel(contours, inline=1, fontsize=12)

        plt.xlabel('x1')
        plt.ylabel('x2')

        plt.savefig(filename)
        pl.close()


def interpret_batch_size_arg(cmd):
    """
    Converts a string with multiplications into an integer
    e.g., "2*8*1" -> 16
    """
    parts = cmd.split("*")
    bs = 1
    for p in parts:
        bs *= int(p)
    return bs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="lds0")
    parser.add_argument("--timeout", default=600, type=int)  # in minutes
    parser.add_argument("--reach_prob", default=0.8, type=float)
    parser.add_argument("--eps", default=0.1, type=float)
    parser.add_argument("--lip_lambda", default=0.001, type=float)
    parser.add_argument("--p_lip", default=3.0, type=float)
    parser.add_argument("--v_lip", default=8.0, type=float)
    parser.add_argument("--hidden", default=128, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--batch_size", default="512")
    parser.add_argument("--ppo_iters", default=150, type=int)
    parser.add_argument("--policy", default="policies/lds0_zero.jax")
    parser.add_argument("--debug_k0", action="store_true")
    parser.add_argument("--gen_plot", action="store_true")
    parser.add_argument("--no_refinement", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--skip_ppo", action="store_true")
    parser.add_argument("--continue_ppo", action="store_true")
    parser.add_argument("--only_ppo", action="store_true")
    parser.add_argument("--small_mem", action="store_true")
    parser.add_argument("--stability_check", action="store_true")
    parser.add_argument("--epsilon_as_tau", action="store_true")
    parser.add_argument("--continue_rsm", type=int, default=0)
    parser.add_argument("--train_p", type=int, default=1)
    parser.add_argument("--fail_check_fast", type=int, default=0)
    parser.add_argument("--soft_constraint", type=int, default=1)
    parser.add_argument("--grid_factor", default=1.0, type=float)
    args = parser.parse_args()

    if args.env.startswith("lds"):
        env = LDSEnv()
        env.name = args.env
    elif args.env.startswith("pend"):
        env = InvertedPendulum()
        env.name = args.env
    elif args.env.startswith("cavoid"):
        env = CollisionAvoidanceEnv()
        env.name = args.env
    elif args.env.startswith("ca3d"):
        env = CollisionAvoidance3DEnv()
        env.name = args.env
    elif args.env.startswith("vdrone"):
        env = vDroneEnv()
        env.name = args.env
    else:
        raise ValueError(f"Unknown environment '{args.env}'")

    os.makedirs("checkpoints", exist_ok=True)
    learner = RSMLearner(
        [args.hidden for i in range(args.num_layers)],
        [128, 128],
        env,
        p_lip=args.p_lip,
        v_lip=args.v_lip,
        lip_lambda=args.lip_lambda,
        eps=args.eps,
        reach_prob=args.reach_prob,
        stability_check=args.stability_check,
        epsilon_as_tau=args.epsilon_as_tau,
    )
    if args.skip_ppo or args.continue_ppo:
        learner.load(f"checkpoints/{args.env}_ppo.jax")

    if not args.skip_ppo:
        learner.pretrain_policy(
            args.ppo_iters, lip_start=0.05 / 10, lip_end=0.05, save_every=10
        )
        learner.save(f"checkpoints/{args.env}_ppo.jax")
        print("[SAVED]")

    verifier = RSMVerifier(
        learner,
        env,
        batch_size=interpret_batch_size_arg(args.batch_size),
        reach_prob=args.reach_prob,
        fail_check_fast=bool(args.fail_check_fast),
        grid_factor=args.grid_factor,
        stability_check=args.stability_check,
    )

    if args.continue_rsm > 0:
        learner.load(f"saved/{args.env}_loop.jax")
        verifier.grid_size *= args.continue_rsm

    loop = RSMLoop(
        learner,
        verifier,
        env,
        plot=args.plot,
        train_p=bool(args.train_p),
        soft_constraint=bool(args.soft_constraint),
    )

    # print("Sampling reward of the policy")
    txt_return = learner.evaluate_rl()

    # loop.plot_l(f"plots/{args.env}_start")
    with open("ppo_results.txt", "a") as f:
        f.write(f"{args.env}: {txt_return}\n")

    if args.only_ppo:
        with open("ppo_results.txt", "a") as f:
            f.write(f"{args.env}: {txt_return}\n")
        import sys

        sys.exit(0)

    sat = loop.run(args.timeout * 60)
    # loop.plot_l(f"plots/{args.env}_end.png", False)
    # loop.plot_l(f"plots/{args.env}_end", plot_rsm=False, plot_kernel=True, form="pdf", save=True)
    # loop.plot_l(f"plots/{args.env}_end")
    # loop.plot_l(f"plots/{args.env}_end_nohard.png", False)

    os.makedirs("study_results", exist_ok=True)
    env_name = args.env.split("_")
    if len(env_name) > 2:
        env_name = env_name[0] + "_" + env_name[1]
    else:
        env_name = args.env
    with open(f"study_results/info_{env_name}.log", "a") as f:
        cmd_line = " ".join(sys.argv)
        f.write(f"{cmd_line}\n")
        f.write("    args=" + str(vars(args)) + "\n")
        f.write("    return =" + txt_return + "\n")
        f.write("    info=" + str(loop.info) + "\n")
        f.write("    sat=" + str(sat) + "\n")
        f.write("\n\n")

    # with open("info.log", "a") as f:
    #     f.write("args=" + str(vars(args)) + "\n")
    #     f.write("sat=" + str(sat) + "\n")
    #     f.write("info=" + str(loop.info) + "\n\n\n")
