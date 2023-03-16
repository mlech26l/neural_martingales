import argparse
import os
import sys

import jax.random
from gym import spaces
from tqdm import tqdm

from klax import lipschitz_l1_jax, triangular
from rl_environments import LDSEnv, InvertedPendulum, CollisionAvoidanceEnv
from rsm_learner import RSMLearner
from rsm_verifier import RSMVerifier
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
        lip_factor,
        plot,
        jitter_grid,
        soft_constraint,
        train_p=True,
    ):
        self.env = env
        self.learner = learner
        self.verifier = verifier
        self.train_p = train_p
        self.soft_constraint = soft_constraint
        self.jitter_grid = jitter_grid

        os.makedirs("saved", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        os.makedirs("loop", exist_ok=True)
        self.lip_factor = lip_factor
        self.plot = plot
        self.prefill_delta = 0
        self.iter = 0
        self.info = {}

    def train_until_zero_loss(self):
        # if len(self.verifier.train_buffer) == 0:
        #     return
        if self.jitter_grid:
            if self.env.observation_space.shape[0] == 2:
                n = 300
            elif self.env.observation_space.shape[0] == 3:
                n = 100
            else:
                n = 20
            train_ds, stepsize = self.verifier.get_domain_jitter_grid(n)
            current_delta = stepsize
        else:
            train_ds = self.verifier.train_buffer.as_tfds(batch_size=4096)
            current_delta = self.prefill_delta
        start_metrics = None
        num_epochs = 200 if self.iter == 0 else 200

        start_time = time.time()
        pbar = tqdm(total=num_epochs)
        for epoch in range(num_epochs):
            # train_p = epoch % 5 == 0
            # train_l = not train_p
            train_p = self.iter >= 3
            # self.iter >= 10
            # if self.env.observation_space.shape[0] == 2

            # self.iter >= 10
            # if self.env.observation_space.shape[0] == 2
            # else self.iter >= 3
            if not self.train_p:
                train_p = False
            # train_p = False
            train_l = True
            metrics = self.learner.train_epoch(
                train_ds, current_delta, self.lip_factor, train_l, train_p
            )
            if start_metrics is None:
                start_metrics = metrics
            pbar.update(1)
            pbar.set_description_str(
                f"Train [l={train_l}, p={train_p}]: loss={metrics['loss']:0.3g}, dec_loss={metrics['dec_loss']:0.3g}, violations={metrics['train_violations']:0.3g}"
            )
        pbar.n = num_epochs
        pbar.refresh()
        pbar.close()
        self.info["ds_size"] = len(self.verifier.train_buffer)

        elapsed = time.time() - start_time
        if elapsed > 60:
            elapsed = f"{elapsed/60:0.1f} minutes"
        else:
            elapsed = f"{elapsed:0.1f} seconds"

        print(
            f"Trained on {len(self.verifier.train_buffer)} samples, start_loss={start_metrics['loss']:0.3g}, end_loss={metrics['loss']:0.3g}, start_violations={start_metrics['train_violations']:0.3g}, end_violations={metrics['train_violations']:0.3g} in {elapsed}"
        )

    def run(self, timeout):

        start_time = time.time()
        last_saved = time.time()
        self.prefill_delta = self.verifier.prefill_train_buffer()
        while True:
            runtime = time.time() - start_time
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
                f"\n#### Iteration {self.iter} ({runtime // 60:0.0f}:{runtime % 60:02.0f} elapsed) #####"
            )
            self.train_until_zero_loss()
            K_f = self.env.lipschitz_constant
            K_p = lipschitz_l1_jax(self.learner.p_state.params).item()
            K_l = lipschitz_l1_jax(self.learner.l_state.params).item()
            # n = 100 if self.env.reach_space.shape[0] == 2 else 20
            # K_p = self.verifier.compute_lipschitz_bound_on_domain(
            #     self.learner.p_ibp, self.learner.p_state.params, n
            # ).item()
            # K_l = self.verifier.compute_lipschitz_bound_on_domain(
            #     self.learner.l_ibp, self.learner.l_state.params, n
            # ).item()
            lipschitz_k = K_l * K_f * (1 + K_p) + K_l
            lipschitz_k = float(lipschitz_k)
            self.info["lipschitz_k"] = lipschitz_k
            self.info["K_p"] = K_p
            self.info["K_f"] = K_f
            self.info["K_l"] = K_l
            self.info["iter"] = self.iter
            self.info["runtime"] = runtime
            sat, hard_violations, info = self.verifier.check_dec_cond(lipschitz_k)
            for k, v in info.items():
                self.info[k] = v
            print("info=", str(self.info), flush=True)
            if sat:
                print("Decrease condition fulfilled!")
                self.learner.save(f"saved/{self.env.name}_loop.jax")
                print("[SAVED]")
                if self.env.observation_space.shape[0] == 2:
                    n = 200
                elif self.env.observation_space.shape[0] == 3:
                    n = 100
                else:
                    n = 50
                _, ub_init = self.verifier.compute_bound_init(n)
                lb_unsafe, _ = self.verifier.compute_bound_unsafe(n)
                domain_min, _ = self.verifier.compute_bound_domain(n)
                print(f"Init   max = {ub_init:0.6g}")
                print(f"Unsafe min = {lb_unsafe:0.6g}")
                print(f"domain min = {domain_min:0.6g}")
                self.info["ub_init"] = ub_init
                self.info["lb_unsafe"] = lb_unsafe
                self.info["domain_min"] = domain_min

                bound_correct = True
                if lb_unsafe < ub_init:
                    bound_correct = False
                    print(
                        "RSM is lower at unsafe than in init. No probabilistic guarantees can be obtained."
                    )
                # normalize to min = 0
                ub_init = ub_init - domain_min
                lb_unsafe = lb_unsafe - domain_min
                # normalize to init=1
                lb_unsafe = lb_unsafe / ub_init
                actual_reach_prob = 1 - 1 / np.clip(lb_unsafe, 1e-9, None)
                self.info["actual_reach_prob"] = actual_reach_prob
                if not bound_correct:
                    self.info["actual_reach_prob"] = "UNSAFE"
                print(
                    f"Probability of reaching the target safely is at least {actual_reach_prob*100:0.3f}% (higher is better)"
                )
                if (
                    self.soft_constraint
                    or actual_reach_prob >= self.verifier.reach_prob
                ):
                    return True
            if hard_violations == 0 and self.iter > 4 and self.iter % 2 == 0:
                print("Refining grid")
                if self.env.reach_space.shape[0] == 2:
                    self.verifier.grid_size *= 2
                elif self.env.reach_space.shape[0] == 3:
                    self.verifier.grid_size *= int(1.5 * self.verifier.grid_size)
                else:
                    self.verifier.grid_size = int(1.35 * self.verifier.grid_size)
                # state_grid = self.verifier.get_filtered_grid(n=self.verifier.grid_size)

            sys.stdout.flush()
            if self.plot:
                self.plot_l(f"loop/{self.env.name}_{self.iter:04d}.png")
            self.iter += 1

    def rollout(self):
        rng = np.random.default_rng().integers(0, 10000)
        rng = jax.random.PRNGKey(rng)
        safe = np.array([0.2, 0.2], np.float32)
        space = spaces.Box(
            low=self.env.reach_space.low,
            high=self.env.reach_space.high,
            dtype=np.float32,
        )

        state = space.sample()
        trace = [np.array(state)]
        for i in range(100):
            action = self.learner.p_state.apply_fn(self.learner.p_state.params, state)
            next_state = self.env.next(state, action)
            rng, seed = jax.random.split(rng)
            noise = triangular(rng, (self.env.observation_space.shape[0],))
            noise = noise * self.env.noise
            state = next_state + noise
            trace.append(np.array(state))
        return np.stack(trace, axis=0)

    def plot_l(self, filename):
        if self.env.observation_space.shape[0] > 2:
            return
        grid, _, _ = self.verifier.get_unfiltered_grid(n=50)
        l = self.learner.l_state.apply_fn(self.learner.l_state.params, grid).flatten()
        l = np.array(l)
        # l = np.clip(l, 0, 5)
        # np.savez(f"plots/{env.name}.npz", grid=grid, l=l)
        sns.set()
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(grid[:, 0], grid[:, 1], marker="s", c=l, zorder=1, alpha=0.7)
        fig.colorbar(sc)
        ax.set_title(f"L at iter {self.iter} for {self.env.name}")

        terminals_x, terminals_y = [], []
        for i in range(20):
            trace = self.rollout()
            ax.plot(
                trace[:, 0],
                trace[:, 1],
                color=sns.color_palette()[0],
                zorder=2,
                alpha=0.3,
            )
            ax.scatter(
                trace[:, 0],
                trace[:, 1],
                color=sns.color_palette()[0],
                zorder=2,
                marker=".",
            )
            terminals_x.append(float(trace[-1, 0]))
            terminals_y.append(float(trace[-1, 1]))
        ax.scatter(terminals_x, terminals_y, color="white", marker="x", zorder=5)
        if self.verifier.hard_constraint_violation_buffer is not None:
            print(
                "self.verifier.hard_constraint_violation_buffer: ",
                self.verifier.hard_constraint_violation_buffer[0:10],
            )
            ax.scatter(
                self.verifier.hard_constraint_violation_buffer[:, 0],
                self.verifier.hard_constraint_violation_buffer[:, 1],
                color="green",
                marker="s",
                alpha=0.7,
                zorder=6,
            )
        if self.verifier._debug_violations is not None:
            ax.scatter(
                self.verifier._debug_violations[:, 0],
                self.verifier._debug_violations[:, 1],
                color="cyan",
                marker="s",
                alpha=0.7,
                zorder=6,
            )
        for init in self.env.init_spaces:
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
        for unsafe in self.env.unsafe_spaces:
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
            ax.plot(x, y, color="magenta", alpha=0.5, zorder=7)
        x = [
            self.env.safe_space.low[0],
            self.env.safe_space.high[0],
            self.env.safe_space.high[0],
            self.env.safe_space.low[0],
            self.env.safe_space.low[0],
        ]
        y = [
            self.env.safe_space.low[1],
            self.env.safe_space.low[1],
            self.env.safe_space.high[1],
            self.env.safe_space.high[1],
            self.env.safe_space.low[1],
        ]
        ax.plot(x, y, color="green", alpha=0.5, zorder=7)

        # if len(self.learner._debug_unsafe) > 0:
        #     init_samples = np.concatenate(self.learner._debug_init, axis=0)
        #     unsafe_samples = np.concatenate(self.learner._debug_unsafe, axis=0)
        #     ax.scatter(
        #         unsafe_samples[:, 0],
        #         unsafe_samples[:, 1],
        #         color="red",
        #         marker="x",
        #         alpha=0.1,
        #         zorder=7,
        #     )
        #     ax.scatter(
        #         init_samples[:, 0],
        #         init_samples[:, 1],
        #         color="green",
        #         marker="x",
        #         alpha=0.1,
        #         zorder=7,
        #     )
        #     self.learner._debug_init = []
        #     self.learner._debug_unsafe = []
        # print(f"Terminals x={terminals_x}, y={terminals_y}")
        ax.set_xlim([self.env.reach_space.low[0], self.env.reach_space.high[0]])
        ax.set_ylim([self.env.reach_space.low[1], self.env.reach_space.high[1]])
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="lds0")
    parser.add_argument("--timeout", default=60, type=int)  # in minutes
    parser.add_argument("--eps", default=0.05, type=float)
    parser.add_argument("--reach_prob", default=0.8, type=float)
    parser.add_argument("--lip", default=0.01, type=float)
    parser.add_argument("--p_lip", default=4.0, type=float)
    parser.add_argument("--l_lip", default=4.0, type=float)
    parser.add_argument("--hidden", default=128, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--ppo_iters", default=50, type=int)
    parser.add_argument("--policy", default="policies/lds0_zero.jax")
    parser.add_argument("--debug_k0", action="store_true")
    parser.add_argument("--gen_plot", action="store_true")
    parser.add_argument("--no_refinement", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--skip_ppo", action="store_true")
    parser.add_argument("--continue_ppo", action="store_true")
    parser.add_argument("--only_ppo", action="store_true")
    parser.add_argument("--small_mem", action="store_true")
    parser.add_argument("--continue_rsm", type=int, default=0)
    parser.add_argument("--train_p", type=int, default=1)
    parser.add_argument("--square_l_output", type=int, default=1)
    parser.add_argument("--fail_check_fast", type=int, default=0)
    parser.add_argument("--jitter_grid", type=int, default=0)
    parser.add_argument("--soft_constraint", type=int, default=1)
    parser.add_argument("--gamma_decrease", default=1.0, type=float)
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
    else:
        raise ValueError(f"Unknown environment '{args.env}'")

    os.makedirs("checkpoints", exist_ok=True)
    learner = RSMLearner(
        [args.hidden for i in range(args.num_layers)],
        [128, 128],
        env,
        p_lip=args.p_lip,
        l_lip=args.l_lip,
        eps=args.eps,
        gamma_decrease=args.gamma_decrease,
        reach_prob=args.reach_prob,
        square_l_output=int(args.square_l_output),
        # square_l_output=int(args.square_l_output),
    )
    if args.skip_ppo or args.continue_ppo:
        learner.load(f"checkpoints/{args.env}_ppo.jax")

    if not args.skip_ppo:
        learner.pretrain_policy(args.ppo_iters, lip_start=0.05 / 10, lip_end=0.05)
        learner.save(f"checkpoints/{args.env}_ppo.jax")
        print("[SAVED]")

    verifier = RSMVerifier(
        learner,
        env,
        batch_size=args.batch_size,
        reach_prob=args.reach_prob,
        fail_check_fast=bool(args.fail_check_fast),
        grid_factor=args.grid_factor,
        small_mem=args.small_mem,
    )

    if args.continue_rsm > 0:
        learner.load(f"saved/{args.env}_loop.jax")
        verifier.grid_size *= args.continue_rsm

    loop = RSMLoop(
        learner,
        verifier,
        env,
        lip_factor=args.lip,
        plot=args.plot,
        train_p=bool(args.train_p),
        jitter_grid=bool(args.jitter_grid),
        soft_constraint=bool(args.soft_constraint),
    )

    # verifier.debug_cavoid()
    # unsafe_min = verifier.compute_lower_bound_unsafe(200)
    # print("unsafe min: ", unsafe_min)

    txt_return = learner.evaluate_rl()
    loop.plot_l(f"plots/{args.env}_start.png")
    if args.only_ppo:
        with open("ppo_results.txt", "a") as f:
            f.write(f"{args.env}: {txt_return}\n")
        import sys

        sys.exit(0)
    sat = loop.run(args.timeout * 60)
    loop.plot_l(f"plots/{args.env}_end.png")

    with open("info.log", "a") as f:
        f.write("args=" + str(vars(args)) + "\n")
        f.write("sat=" + str(sat) + "\n")
        f.write("info=" + str(loop.info) + "\n\n\n")
