"""One Lab5 training run: the TA's DQNAgent, plus what the template leaves out (extra, not required, except where noted).

The TA's dqn.py changes only inside its TODOs. Everything else lives here, in a subclass, so that
`git diff` against the template stays small.

Usage (the real commands are in experiments.sh):
    python run_experiment.py --name task1 --env CartPole-v1 --replay-start-size 1000 ...
    python run_experiment.py --name task3_r1 --env ALE/Pong-v5 --resume-from runs/task3/resume.pt ...

Files a run writes in runs/<name>/:
    env.json          Python, PyTorch, CUDA, GPU, package versions, git commit, every setting, before training starts
    log.csv           one row per evaluation: environment steps, updates, 20-game average, min, max, epsilon
    checkpoints.csv   the SHA-256 of every .pt file, recorded as the run goes
    snapshot_<n>.pt   the weights at exactly n environment steps (spec 4.2.2, Task 3)
    first19_<n>.pt    Pong: the first model whose 20-game average reaches 19, at n environment steps
    best_model.pt     the best 20-game average so far (saved by the TA's run())
    model_ep<n>.pt    the TA's save every 100 games
    resume.pt         weights, optimizer and counters, to restart after a crash
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from collections import deque

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch
import wandb

from dqn import DQNAgent, PrioritizedReplayBuffer

RESUME_EVERY = 50000  # environment steps between two saves of resume.pt


class PassThroughPreprocessor:
    """Stands in for AtariPreprocessor in CartPole, where the state is already 4 numbers.

    Required: cv2.cvtColor raises an error on a 4-number state. run() and evaluate() call self.preprocessor
    in both games, so this keeps those TA functions unchanged.
    """

    def reset(self, obs):
        return obs

    def step(self, obs):
        return obs


class ReplayMemory:
    """The replay memory: a short n-step queue in front of a uniform ring or the TA's PrioritizedReplayBuffer.

    Required by spec 2.1 (uniform replay) and spec 2.3 (PER and multi-step). The TA's run() calls
    self.memory.append(...), so this class keeps that line unchanged. Fundamentals pages 5, 8 and 9.
    """

    def __init__(
        self,
        capacity,
        gamma,
        n_step,
        per,
        alpha,
        beta,
        beta_end,
        beta_steps,
        priority_clip=0.0,
        point_ends_target=False,
        augment_pad=0,
    ):
        self.gamma = gamma
        self.augment_pad = augment_pad  # 0 = off. Pixels a training frame may move by (DrQ random shift). Extra, G1
        self.priority_clip = priority_clip  # 0 = no limit. Extra, Task 3 report E5
        self.point_ends_target = point_ends_target  # False = only the end of a game ends a target. Extra, report E7
        self.n_step = n_step  # 1 = plain DQN
        self.per = per  # False = uniform sampling
        self.n_queue = deque()  # transitions waiting for their next n_step rewards
        self.max_error = 1.0  # priority of a new transition: the largest error so far (page 8, Step 5)
        self.beta_start = beta
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.samples_drawn = 0
        # deque(maxlen=...) drops the oldest item by itself once it is full.
        self.buffer = PrioritizedReplayBuffer(capacity, alpha, beta) if per else deque(maxlen=capacity)

    def __len__(self):
        """How many transitions are stored. The TA's train() calls len(self.memory)."""
        return len(self.buffer.buffer) if self.per else len(self.buffer)

    def append(self, transition):
        """Take one (state, action, reward, next_state, done) from run(), and store it once its window is full."""
        if self.point_ends_target and transition[2] != 0:
            # Pong: after a point the ball is served again. *transition[:4] unpacks (s, a, r, s'); done becomes True.
            transition = (*transition[:4], True)
        self.n_queue.append(transition)
        done = transition[4]
        if done:
            # End of a game: every waiting transition is stored now, with a shorter window (page 9, Step 3).
            while self.n_queue:
                self.emit()
        elif len(self.n_queue) == self.n_step:
            self.emit()

    def emit(self):
        """Store the oldest waiting transition, with its reward replaced by the n-step sum (page 9, Step 4)."""
        # enumerate gives (0, item0), (1, item1), ... item[2] is the reward of that step.
        g = sum(self.gamma**k * item[2] for k, item in enumerate(self.n_queue))  # CORE: n-step return
        state, action = self.n_queue[0][0], self.n_queue[0][1]  # [0] = the oldest step in the window
        next_state, done = self.n_queue[-1][3], self.n_queue[-1][4]  # [-1] = the newest step in the window
        self.n_queue.popleft()
        if self.per:
            self.buffer.add((state, action, g, next_state, done), self.max_error)
        else:
            self.buffer.append((state, action, g, next_state, done))

    def sample(self, batch_size):
        """Return (batch, indices, weights). Without PER the indices are None and every weight is 1."""
        if self.per:
            # Raise beta from its start value to beta_end (page 8, Step 6). Extra, not required.
            fraction = min(1.0, self.samples_drawn / self.beta_steps)
            self.buffer.beta = self.beta_start + (self.beta_end - self.beta_start) * fraction
            self.samples_drawn += 1
            batch, indices, weights = self.buffer.sample(batch_size)
        else:
            batch = random.sample(self.buffer, batch_size)  # CORE: uniform sampling
            indices, weights = None, np.ones(batch_size, dtype=np.float32)
        if self.augment_pad > 0:
            batch = self.random_shift(batch)
        return batch, indices, weights

    def random_shift(self, batch):
        """Move each state and next state by up to augment_pad pixels, in training only (DrQ, Task 3 report G1).

        The memory keeps the real frames; only the copies in this batch move. The 4 stacked frames of one state
        move together, and a state and its next state move independently, as in DrQ.
        """
        p = self.augment_pad
        states, actions, rewards, next_states, dones = zip(*batch)
        frames = np.stack(states + next_states)  # (2B, 4, 84, 84): all states, then all next states
        # mode="edge" repeats the border pixels into the new margin: (2B, 4, 84 + 2p, 84 + 2p)
        padded = np.pad(frames, ((0, 0), (0, 0), (p, p), (p, p)), mode="edge")
        size = frames.shape[-1]  # 84
        offsets = np.random.randint(0, 2 * p + 1, size=(len(frames), 2))  # (2B, 2) row and column of each crop
        moved = [padded[k, :, dy : dy + size, dx : dx + size] for k, (dy, dx) in enumerate(offsets)]
        b = len(batch)
        return list(zip(moved[:b], actions, rewards, moved[b:], dones))

    def update_priorities(self, indices, errors):
        """Pass the new error sizes to the PER buffer. It does nothing with uniform sampling."""
        if self.per:
            if self.priority_clip > 0:
                # The loss in train() keeps the full error. Only the sampling chances see this limit (report E5).
                errors = np.minimum(errors, self.priority_clip)  # (B)
            self.max_error = max(self.max_error, float(errors.max()))
            self.buffer.update_priorities(indices, errors)


class LabAgent(DQNAgent):
    """The TA's DQNAgent, with the setup its __init__ does not do, and a 20-game evaluate()."""

    def __init__(self, args):
        super().__init__(env_name=args.env, args=args)
        is_pong = "Pong" in args.env
        if not is_pong:
            self.preprocessor = PassThroughPreprocessor()
        self.best_reward = -21 if is_pong else 0  # the TA's comment in __init__: 0 for CartPole, -21 for Pong
        self.memory = ReplayMemory(
            args.memory_size,
            args.discount_factor,
            args.n_step,
            args.per,
            args.per_alpha,
            args.per_beta,
            args.per_beta_end,
            args.per_beta_steps,
            args.priority_clip,
            args.point_ends_target,
            args.augment_pad,
        )
        # The TA's __init__ uses Adam's default eps, 1e-8. Replacing it here keeps dqn.py unchanged (report E3).
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=args.lr, eps=args.adam_eps)
        self.train_every = args.train_every
        self.train_calls = 0  # how many times run() has asked for an update
        self.first_19_saved = not is_pong  # the target of 19 exists only in Pong (spec 4.2.2)
        self.double = args.double
        self.n_step = args.n_step
        self.max_env_steps = args.max_env_steps
        self.snapshot_steps = set(int(s) for s in args.snapshot_steps.split(",") if s)
        self.last_loss = 0.0  # logged by run() before the first update
        self.last_q_mean = 0.0
        self.last_error_mean = 0.0
        self.log_path = os.path.join(self.save_dir, "log.csv")
        self.checkpoints_path = os.path.join(self.save_dir, "checkpoints.csv")
        self.seen_files = {}  # file name -> (modification time, size), to find new or changed checkpoints
        self.last_time = time.time()
        self.last_count = 0
        write_csv_row(self.log_path, ["env_step", "updates", "mean", "min", "max", "epsilon", "seconds"])
        write_csv_row(self.checkpoints_path, ["utc_time", "env_step", "file", "bytes", "sha256"])
        self.start_time = time.time()
        self.env.reset(seed=args.seed)  # later reset() calls continue from this seed
        if args.resume_from:
            self.load_resume(args.resume_from)

    def train(self):
        """Do the TA's update on every train_every-th call. run() calls this once per environment step (report E2).

        Epsilon decays inside the update, so with --train-every 4 the decay per update must be 4 times stronger
        to keep the same schedule in environment steps: 0.99998 ** 4 = 0.99992.
        """
        self.train_calls += 1
        if self.train_calls % self.train_every == 0:
            super().train()

    def evaluate(self):
        """Average score over 20 games, seeds 0 to 19, with no random actions (spec 4.2.2).

        Replaces the TA's one-game evaluate(), so that best_model.pt and the logged Eval Reward are not luck.
        It only runs between games, because it reuses self.preprocessor.
        """
        scores = []
        for seed in range(20):
            obs, _ = self.test_env.reset(seed=seed)
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            while not done:
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax().item()
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = self.preprocessor.step(next_obs)
            scores.append(total_reward)
        mean = float(np.mean(scores))
        seconds = round(time.time() - self.start_time)
        row = [self.env_count, self.train_count, mean, min(scores), max(scores), round(self.epsilon, 4), seconds]
        write_csv_row(self.log_path, row)
        if mean >= 19 and not self.first_19_saved:
            # Keep the first model that reaches 19: the grade depends on how early that is (spec 4.2.2), and
            # best_model.pt is overwritten by every later, higher score.
            torch.save(self.q_net.state_dict(), os.path.join(self.save_dir, f"first19_{self.env_count}.pt"))
            self.first_19_saved = True
        self.record_new_checkpoints()  # picks up the best_model.pt that run() saved after the last evaluation
        return mean

    def save_files_at_this_step(self):
        """Called by run() every 1000 environment steps: snapshots, the resume file, and a speed line."""
        if self.env_count in self.snapshot_steps:  # spec 4.2.2, Task 3
            torch.save(self.q_net.state_dict(), os.path.join(self.save_dir, f"snapshot_{self.env_count}.pt"))
            self.record_new_checkpoints()
        if self.env_count % RESUME_EVERY == 0:
            self.save_resume()
        if self.env_count % 10000 == 0:
            now = time.time()
            speed = (self.env_count - self.last_count) / (now - self.last_time)
            hours_left = (self.max_env_steps - self.env_count) / speed / 3600
            print(f"[Speed] {speed:.0f} env steps/s, about {hours_left:.1f} h left, SC: {self.env_count}", flush=True)
            wandb.log({"Env Step Count": self.env_count, "Env Steps Per Second": speed})
            self.last_time, self.last_count = now, self.env_count

    def record_new_checkpoints(self):
        """Add the SHA-256 of every new or changed .pt file to checkpoints.csv (hash at training time)."""
        for name in sorted(os.listdir(self.save_dir)):
            path = os.path.join(self.save_dir, name)
            if not name.endswith(".pt") or name == "resume.pt":
                continue
            stamp = (os.path.getmtime(path), os.path.getsize(path))
            if self.seen_files.get(name) != stamp:
                self.seen_files[name] = stamp
                utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                write_csv_row(self.checkpoints_path, [utc, self.env_count, name, stamp[1], sha256_of(path)])

    def save_resume(self):
        """Save what a restart needs. The replay memory is not saved: it would take gigabytes, and it refills."""
        state = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "env_count": self.env_count,
            "train_count": self.train_count,
            "epsilon": self.epsilon,
            "best_reward": self.best_reward,
        }
        path = os.path.join(self.save_dir, "resume.pt")
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)  # a crash during the save cannot damage the previous file

    def load_resume(self, path):
        """Continue from resume.pt. Training waits until the memory refills to replay_start_size."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(state["q_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.env_count = state["env_count"]
        self.train_count = state["train_count"]
        self.epsilon = state["epsilon"]
        self.best_reward = state["best_reward"]
        self.last_count = self.env_count
        print(f"Resumed from {path} at {self.env_count} environment steps", flush=True)


def write_csv_row(path, row):
    """Append one row to a CSV file, and flush it, so that a crash loses nothing."""
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def sha256_of(path):
    """Return the SHA-256 of a file, as a hex string."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # read 1 MB at a time until f.read returns b""
            digest.update(chunk)
    return digest.hexdigest()


def record_environment(args):
    """Write env.json before training: versions, GPU, git commit and every setting."""

    def git(*command):
        return subprocess.run(["git", *command], capture_output=True, text=True).stdout.strip()

    cuda = torch.cuda.is_available()
    info = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gymnasium": gym.__version__,
        "ale_py": ale_py.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "wandb": wandb.__version__,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "*.py")),
        "command": " ".join(sys.argv),
        "args": vars(args),
    }
    with open(os.path.join(args.save_dir, "env.json"), "w") as f:
        json.dump(info, f, indent=2)


def parse_args():
    """The TA's settings keep their names and defaults. The rest is new."""
    parser = argparse.ArgumentParser(description="One Lab5 training run")
    # The TA's settings
    parser.add_argument("--batch-size", type=int, default=32)  # default 32
    parser.add_argument("--memory-size", type=int, default=100000)  # default 100000
    parser.add_argument("--lr", type=float, default=0.0001)  # default 0.0001
    parser.add_argument("--discount-factor", type=float, default=0.99)  # default 0.99
    parser.add_argument("--epsilon-start", type=float, default=1.0)  # default 1.0
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)  # default 0.999999, per update
    parser.add_argument("--epsilon-min", type=float, default=0.05)  # default 0.05
    parser.add_argument("--target-update-frequency", type=int, default=1000)  # default 1000 updates
    parser.add_argument("--replay-start-size", type=int, default=50000)  # default 50000
    parser.add_argument("--max-episode-steps", type=int, default=10000)  # default 10000
    parser.add_argument("--train-per-step", type=int, default=1)  # default 1
    # New: which run
    parser.add_argument("--name", type=str, required=True)  # no default: the folder runs/<name>
    parser.add_argument("--env", type=str, default="CartPole-v1")  # default CartPole-v1; or ALE/Pong-v5
    parser.add_argument("--seed", type=int, default=0)  # default 0
    parser.add_argument("--max-env-steps", type=int, default=1000000)  # default 1000000; a multiple of 1000
    parser.add_argument("--snapshot-steps", type=str, default="")  # default none; comma list, multiples of 1000
    parser.add_argument("--resume-from", type=str, default="")  # default none; path to a resume.pt
    parser.add_argument("--wandb-mode", type=str, default="online")  # default online; or offline, disabled
    # New: the Task 3 techniques (spec 2.3). All off by default, which is vanilla DQN.
    parser.add_argument("--double", action="store_true")  # default off
    parser.add_argument("--per", action="store_true")  # default off
    parser.add_argument("--n-step", type=int, default=1)  # default 1
    parser.add_argument("--per-alpha", type=float, default=0.6)  # default 0.6, the TA's default
    parser.add_argument("--per-beta", type=float, default=0.4)  # default 0.4, the TA's default
    parser.add_argument("--per-beta-end", type=float, default=1.0)  # default 1.0; extra, not required
    parser.add_argument("--per-beta-steps", type=int, default=1000000)  # default 1000000 updates; extra
    # New: the Task 3 experiments. Extra, not required. The defaults change nothing.
    parser.add_argument("--train-every", type=int, default=1)  # default 1; one update per this many env steps (E2, E6)
    parser.add_argument("--adam-eps", type=float, default=1e-8)  # default 1e-8, Adam's own default (E3)
    parser.add_argument("--priority-clip", type=float, default=0.0)  # default 0, no limit (E5)
    parser.add_argument("--point-ends-target", action="store_true")  # default off (E7)
    parser.add_argument("--augment-pad", type=int, default=0)  # default 0, off; DrQ uses 4 (report G1)
    args = parser.parse_args()
    if args.max_env_steps % 1000 != 0:
        parser.error("--max-env-steps must be a multiple of 1000: run() only checks it every 1000 steps")
    for step in filter(None, args.snapshot_steps.split(",")):
        if int(step) % 1000 != 0:
            parser.error("--snapshot-steps must be multiples of 1000: run() only checks them every 1000 steps")
    return args


def main():
    args = parse_args()
    args.save_dir = os.path.join("runs", args.name)
    if os.path.exists(args.save_dir):
        sys.exit(f"{args.save_dir} already exists. Use a new --name, so that no earlier result is overwritten.")
    os.makedirs(args.save_dir)
    record_environment(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    project = "DLP-Lab5-DQN-Pong" if "Pong" in args.env else "DLP-Lab5-DQN-CartPole"
    wandb.init(project=project, name=args.name, config=vars(args), save_code=True, mode=args.wandb_mode)
    # Spec 4.1: the x-axis must be environment steps. W&B's default x-axis only counts log calls.
    wandb.define_metric("Env Step Count")
    wandb.define_metric("*", step_metric="Env Step Count")

    agent = LabAgent(args)
    agent.run(episodes=10**9)  # the TODO in run() stops it at --max-env-steps
    agent.record_new_checkpoints()
    wandb.finish()
    print(f"[Done] {args.name}: {agent.env_count} env steps, best 20-game average {agent.best_reward}", flush=True)


if __name__ == "__main__":
    main()
