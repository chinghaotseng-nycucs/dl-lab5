"""Smoke test for Lab5: checks the code against the numbers on the fundamentals pages, then runs everything briefly.

Extra, not required. Usage: python smoke_test.py
It takes a few minutes on a laptop. It needs no login: W&B runs in offline mode. Its run folders are deleted at the end.
"""

import difflib
import os
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

os.environ["WANDB_SILENT"] = "true"

import dqn  # noqa: E402
import run_experiment  # noqa: E402
from dqn import DQN, DQNAgent, PrioritizedReplayBuffer  # noqa: E402
from run_experiment import ReplayMemory  # noqa: E402

TEMPLATE = os.path.join("..", "material", "dqn.py")
SMOKE_NAMES = ["_smoke_cartpole", "_smoke_cartpole_full", "_smoke_pong", "_smoke_pong_full"]
failures = []


def check(name, condition, detail=""):
    """Print one result line, and remember a failure."""
    print(f"{'ok  ' if condition else 'FAIL'} {name}{'  ' + str(detail) if detail != '' else ''}", flush=True)
    if not condition:
        failures.append(name)


def close(a, b, tolerance=1e-3):
    """True when two numbers or two lists agree to the rounding used on the pages."""
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tolerance)


def test_only_todos_changed():
    """dqn.py may differ from the template only inside the TODOs, and on the lines the TA's NOTEs say to enable."""
    if not os.path.exists(TEMPLATE):
        print("skip the template comparison: ../material/dqn.py is not here")
        return
    old = open(TEMPLATE).read().split("\n")
    new = open("dqn.py").read().split("\n")
    bad = []
    inside = False  # are we between a YOUR CODE HERE marker and its END marker?
    enable = 0  # how many lines after a NOTE may be enabled
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    changed_old = set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            changed_old.update(range(i1, max(i2, i1 + 1)))
    for i, line in enumerate(old):
        if "YOUR CODE HERE" in line:
            inside = True
        if "NOTE: Enable this part after" in line:
            enable = 6
        elif 'NOTE: Enable this part if "loss"' in line:
            enable = 2
        elif enable > 0:
            enable -= 1
            continue
        if i in changed_old and not inside:
            bad.append(line.strip())
        if "END OF YOUR CODE" in line:
            inside = False
    pending_forward = bad == ["return self.network(x)"]
    if pending_forward:
        print("note: DQN.forward differs from the template (x / self.input_scale). You approved this line.")
    check("dqn.py changed only inside the TODOs and the NOTE lines", bad == [] or pending_forward, bad or "")


def test_network_shapes():
    """Page 3: parameter counts and output shapes. test_model.py must be able to load the Pong weights."""
    from test_model import DQN as TA_DQN

    cartpole, pong = DQN(2), DQN(6)
    check("page 3: CartPole parameters", sum(p.numel() for p in cartpole.parameters()) == 4610)
    check("page 3: Pong parameters", sum(p.numel() for p in pong.parameters()) == 1687206)
    check("page 3: CartPole output", cartpole.network(torch.zeros(32, 4)).shape == (32, 2))
    check("page 3: Pong output", pong.network(torch.zeros(32, 4, 84, 84)).shape == (32, 6))
    ta = TA_DQN(4, 6)
    same = {k: v.shape for k, v in ta.state_dict().items()} == {k: v.shape for k, v in pong.state_dict().items()}
    check("page 3: Pong weights have the same names and shapes as test_model.py", same)
    x = torch.full((1, 4, 84, 84), 107.0)
    scaled = torch.equal(pong(x), pong.network(x / 255.0))
    check("page 4: DQN.forward divides Pong pixels by 255", scaled, "" if scaled else "not approved or applied yet")
    y = torch.tensor([[0.1, 0.0, -0.2, 0.3]])
    check("page 4: DQN.forward leaves CartPole states alone", torch.equal(cartpole(y), cartpole.network(y)))


def test_per():
    """Page 8: priorities, chances, weights, the ring, and the update."""
    buffer = PrioritizedReplayBuffer(4, alpha=0.6, beta=0.4)
    for i, error in enumerate([2.0, -0.5, 0.1, 0.0]):
        buffer.add(f"t{i}", error)
    check("page 8 Step 1: priorities", close(buffer.priorities, [2.01, 0.51, 0.11, 0.01]))
    scaled = buffer.priorities[:4].astype(np.float64) ** 0.6
    check("page 8 Step 2: chances", close(scaled / scaled.sum(), [0.604, 0.2653, 0.1057, 0.0251]))
    np.random.seed(0)
    draws = np.concatenate([buffer.sample(32)[1] for _ in range(3000)])
    check(
        "page 8 Step 2: draw frequency",
        close(np.bincount(draws, minlength=4) / draws.size, scaled / scaled.sum(), 0.01),
    )

    buffer = PrioritizedReplayBuffer(3, alpha=0.6, beta=0.4)
    for i, error in enumerate([2.0, 0.5, 0.1]):
        buffer.add(f"t{i}", error)
    check("page 8 Step 5: position wraps", buffer.pos == 0)
    buffer.add("t3", 2.0)
    check("page 8 Step 5: the oldest is replaced", buffer.buffer == ["t3", "t1", "t2"] and buffer.pos == 1)
    np.random.seed(1)
    batch, indices, weights = buffer.sample(2)
    p = np.array([2.01, 0.51, 0.11]) ** 0.6 / (np.array([2.01, 0.51, 0.11]) ** 0.6).sum()  # the exact P(i)
    expected = (3 * p[indices]) ** -0.4
    check(
        "page 8 Step 5: sample",
        batch == [buffer.buffer[i] for i in indices] and close(weights, expected / expected.max(), 1e-5),
    )
    buffer.update_priorities([0, 1], [0.2, 0.3])
    check("page 8 Step 5: update", close(buffer.priorities, [0.21, 0.31, 0.11]))
    check("page 8: stored p ** alpha follows the update", close(buffer.scaled, buffer.priorities**0.6, 1e-9))
    test_per_fast_sampling()


def test_per_fast_sampling():
    """The roulette-wheel sample(): the same chances and IS weights as the old np.random.choice version."""
    np.random.seed(2)
    buffer = PrioritizedReplayBuffer(50, alpha=0.6, beta=0.4)
    for i in range(70):  # 70 > 50, so the ring wraps and 20 places are overwritten
        buffer.add(i, np.random.exponential())
    buffer.update_priorities(np.arange(0, 50, 3), np.random.exponential(size=17) * 5)
    old_scaled = buffer.priorities.astype(np.float64) ** 0.6  # what the old sample() computed each time
    check("PER fast: stored p ** alpha after wrap and update", close(buffer.scaled, old_scaled, 1e-12))
    old_chances = old_scaled / old_scaled.sum()
    draws = np.concatenate([buffer.sample(32)[1] for _ in range(20000)])  # 640,000 draws
    frequency = np.bincount(draws, minlength=50) / draws.size
    check(
        "PER fast: draw frequency = old chances (640k draws)",
        np.abs(frequency - old_chances).max() < 0.002,
        f"largest gap {np.abs(frequency - old_chances).max():.5f}",
    )
    buffer.beta = 0.7
    batch, indices, weights = buffer.sample(32)
    old_weights = (50 * old_chances[indices]) ** -0.7
    check("PER fast: IS weights = old formula", close(weights, old_weights / old_weights.max(), 1e-5))
    check("PER fast: batch matches indices", batch == [buffer.buffer[i] for i in indices])

    big = PrioritizedReplayBuffer(500000, alpha=0.6, beta=0.4)
    for i in range(1000):
        big.add(i, 1.0)
    big.buffer = list(range(500000))  # fill the rest by hand: add() 500k times would take long
    big.priorities[:] = np.random.random(500000).astype(np.float32) + 0.01
    big.scaled[:] = big.priorities.astype(np.float64) ** 0.6
    start = time.time()
    for _ in range(200):
        indices = big.sample(32)[1]
    ms = (time.time() - start) / 200 * 1000
    check("PER fast: sampling at 500,000 entries", indices.shape == (32,) and indices.max() < 500000, f"{ms:.2f} ms")
    print(f"       PER sample() at 500,000 entries: {ms:.2f} ms per batch")


def test_resets_and_weight_decay():
    """--reset-every, --reset-keep and --weight-decay (Task 3 report H)."""
    torch.manual_seed(0)
    agent = SimpleNamespace(
        num_actions=6,
        device=torch.device("cpu"),
        lr=5e-5,
        adam_eps=1.5e-4,
        weight_decay=0.1,
        reset_keep=0.8,
        resets_done=0,
        env_count=123,
        train_count=40000,
    )
    agent.q_net = DQN(6)
    agent.target_net = DQN(6)
    agent.make_optimizer = lambda: run_experiment.LabAgent.make_optimizer(agent)
    agent.optimizer = agent.make_optimizer()
    check("H: --weight-decay uses AdamW", type(agent.optimizer).__name__ == "AdamW")
    old = {k: v.clone() for k, v in agent.q_net.state_dict().items()}
    agent.q_net(torch.rand(2, 4, 84, 84) * 255).sum().backward()
    agent.optimizer.step()
    run_experiment.LabAgent.reset_network(agent)
    new = agent.q_net.state_dict()
    head_changed = all(not torch.equal(old[k], new[k]) for k in old if k.startswith(("network.7.", "network.9.")))
    # A kept conv weight is 0.8 * old + 0.2 * random, so it stays close to the old one but not equal.
    gap = (new["network.0.weight"] - old["network.0.weight"]).abs().mean() / old["network.0.weight"].abs().mean()
    check("H: the Linear layers start again", head_changed)
    check("H: conv layers keep 80 percent", 0.05 < gap.item() < 0.6, f"relative change {gap.item():.2f}")
    check(
        "H: target copies the reset network",
        all(torch.equal(new[k], agent.target_net.state_dict()[k]) for k in new),
    )
    check("H: optimizer starts again", len(agent.optimizer.state) == 0 and agent.resets_done == 1)
    agent.weight_decay = 0.0
    check("H off: plain Adam", type(agent.make_optimizer()).__name__ == "Adam")


def test_random_shift():
    """--augment-pad: each training copy moves by at most pad pixels; the stored frames do not change (G1)."""
    np.random.seed(3)
    memory = ReplayMemory(100, 0.99, 1, False, 0.6, 0.4, 1.0, 1000, augment_pad=4)
    states = [np.random.randint(0, 256, size=(4, 84, 84), dtype=np.uint8) for _ in range(9)]
    for k in range(8):
        memory.append((states[k], 1, 0.0, states[k + 1], False))
    stored = [s.copy() for s in states]
    batch, _, _ = memory.sample(8)
    shapes_ok = all(b[0].shape == (4, 84, 84) and b[0].dtype == np.uint8 and b[3].shape == (4, 84, 84) for b in batch)
    check("G1: shifted frames keep shape and type", shapes_ok)
    check("G1: stored frames unchanged", all((a == b).all() for a, b in zip(states, stored)))

    def offset_of(moved, original):
        padded = np.pad(original, ((0, 0), (4, 4), (4, 4)), mode="edge")
        for dy in range(9):
            for dx in range(9):
                if (padded[:, dy : dy + 84, dx : dx + 84] == moved).all():
                    return dy - 4, dx - 4
        return None

    # Find each moved state among the originals and check the shift is within 4 pixels.
    ok = True
    pairs = []
    for b in batch:
        hits = [offset_of(b[0], s) for s in states[:8]]
        hit = next((h for h in hits if h is not None), None)
        ok = ok and hit is not None and max(abs(hit[0]), abs(hit[1])) <= 4
        pairs.append(hit)
    check("G1: each state is an original moved by at most 4 pixels", ok, str(pairs[:3]))
    plain = ReplayMemory(100, 0.99, 1, False, 0.6, 0.4, 1.0, 1000)
    for k in range(8):
        plain.append((states[k], 1, 0.0, states[k + 1], False))
    batch, _, _ = plain.sample(8)
    check("G1 off: frames returned unchanged", all(any((b[0] == s).all() for s in states) for b in batch))


def test_n_step():
    """Page 9, Step 4: the stored rewards, next states and done flags of one game."""
    memory = ReplayMemory(10, 0.99, 3, False, 0.6, 0.4, 1.0, 1000)
    rewards = [0, 0, 1, 0, 2]
    for t, r in enumerate(rewards):
        memory.append((f"s{t}", 0, r, f"s{t + 1}", t == 4))
    stored = list(memory.buffer)
    check("page 9 Step 4: sums", close([x[2] for x in stored], [0.9801, 0.99, 2.9602, 1.98, 2.0]))
    check("page 9 Step 4: next states", [x[3] for x in stored] == ["s3", "s4", "s5", "s5", "s5"])
    check("page 9 Step 4: done flags", [x[4] for x in stored] == [False, False, True, True, True])
    plain = ReplayMemory(10, 0.99, 1, False, 0.6, 0.4, 1.0, 1000)
    plain.append(("s0", 1, 5.0, "s1", False))
    check("page 9: --n-step 1 stores the transition as it is", list(plain.buffer) == [("s0", 1, 5.0, "s1", False)])


class TableNet(nn.Module):
    """A fake network: it returns a row of a fixed table, chosen by the integer state."""

    def __init__(self, rows):
        super().__init__()
        self.table = nn.Parameter(torch.tensor(rows, dtype=torch.float32))

    def forward(self, x):
        return self.table[x.long()]


class FixedMemory:
    """A fake memory: it always returns the same batch."""

    def __init__(self, batch, weights):
        self.batch, self.weights = batch, np.array(weights, dtype=np.float32)
        self.errors = None

    def __len__(self):
        return 10**6

    def sample(self, batch_size):
        return self.batch, None, self.weights

    def update_priorities(self, indices, errors):
        self.errors = errors


def train_once(q_rows, target_rows, batch, weights=None, gamma=0.99, double=False, n_step=1):
    """Run the real DQNAgent.train() once on fixed tables, and return the agent."""
    agent = DQNAgent.__new__(DQNAgent)  # build the object without creating environments
    agent.q_net, agent.target_net = TableNet(q_rows), TableNet(target_rows)
    agent.optimizer = torch.optim.Adam(agent.q_net.parameters(), lr=0.01)
    agent.memory = FixedMemory(batch, weights if weights is not None else [1.0] * len(batch))
    agent.device = torch.device("cpu")
    agent.batch_size, agent.gamma, agent.double, agent.n_step = len(batch), gamma, double, n_step
    agent.replay_start_size, agent.train_count, agent.target_update_frequency = 0, 0, 10**9
    agent.epsilon, agent.epsilon_decay, agent.epsilon_min = 1.0, 1.0, 0.05
    agent.train()
    return agent


def test_train():
    """Pages 6, 7, 8 and 9: the real train() on the worked examples."""
    q_rows = [[1, 2], [0.5, -1], [0.2, 0.3]]
    target_rows = [[1.5, 0.5], [0, 1], [2, 3]]
    batch = [(0, 1, 1.0, 0, False), (1, 0, 1.0, 1, False), (2, 0, -1.0, 2, True)]
    agent = train_once(q_rows, target_rows, batch)
    check("page 6 Step 3: mean loss", close(agent.last_loss, 1.2984))
    check("page 6 Step 2: errors", close(agent.memory.errors, [0.485, 1.49, 1.2]))

    # Page 7, Step 3. State 0 has prediction 0. Next state 1 has the values from the page.
    q_rows = [[0, 0, 0], [1, 2, 1.5]]
    target_rows = [[0, 0, 0], [1.2, 0.8, 1.4]]
    batch = [(0, 0, 1.0, 1, False)]
    check("page 7 Step 3: vanilla target", close(train_once(q_rows, target_rows, batch).memory.errors, [2.386]))
    double = train_once(q_rows, target_rows, batch, double=True)
    check("page 7 Step 3: Double DQN target", close(double.memory.errors, [1.792]))

    # Page 9, Step 5: the reward field already holds the n-step sum.
    batch = [(0, 0, 0.9801, 1, False)]
    full = train_once(q_rows, target_rows, batch, double=True, n_step=3)
    check("page 9 Step 5: Task 3 target", close(full.memory.errors, [1.7563]))

    # Page 8, Step 4: predictions 0, targets 2 and 0.5, done flags set.
    q_rows, target_rows = [[0, 0]], [[0, 0]]
    batch = [(0, 0, 2.0, 0, True), (0, 0, 0.5, 0, True)]
    weighted = train_once(q_rows, target_rows, batch, weights=[0.28005, 0.3892])
    check("page 8 Step 4: weighted loss", close(weighted.last_loss, 0.6087))


def run(*command):
    """Run a command, and return its text output. A non-zero exit counts as a failure."""
    done = subprocess.run([sys.executable, *command], capture_output=True, text=True)
    if done.returncode != 0:
        print(done.stdout[-1500:], done.stderr[-1500:])
    check(" ".join(command)[:90], done.returncode == 0)
    return done.stdout


def test_experiment_flags():
    """The Task 3 report's flags: each does what the report says, and the defaults change nothing."""
    # E7: a reward flushes the window, and both stored transitions end their targets.
    memory = ReplayMemory(100, 0.99, 3, False, 0.6, 0.4, 1.0, 100, point_ends_target=True)
    memory.append(("s0", 0, 0.0, "s1", False))
    check("E7: no reward, nothing stored yet", len(memory) == 0)
    memory.append(("s1", 0, 1.0, "s2", False))
    stored = list(memory.buffer)
    check("E7: a point stores both", len(stored) == 2 and stored[0][4] and stored[1][4])
    check("E7: the return still discounts", close(stored[0][2], 0.99) and close(stored[1][2], 1.0))
    # The same steps without the flag stay in the queue.
    memory = ReplayMemory(100, 0.99, 3, False, 0.6, 0.4, 1.0, 100)
    memory.append(("s0", 0, 0.0, "s1", False))
    memory.append(("s1", 0, 1.0, "s2", False))
    check("E7 off: a point stores nothing", len(memory) == 0)
    # E5: the priority is limited, and a new transition can no longer get a larger one.
    memory = ReplayMemory(100, 0.99, 1, True, 0.6, 0.4, 1.0, 100, priority_clip=1.0)
    memory.append(("s0", 0, 0.0, "s1", True))
    memory.update_priorities(np.array([0]), np.array([5.0]))
    check("E5: priority limited to 1 + 0.01", close(memory.buffer.priorities[0], 1.01))
    check("E5: max_error limited to 1", memory.max_error == 1.0)
    memory = ReplayMemory(100, 0.99, 1, True, 0.6, 0.4, 1.0, 100)
    memory.append(("s0", 0, 0.0, "s1", True))
    memory.update_priorities(np.array([0]), np.array([5.0]))
    check(
        "E5 off: priority unchanged by the limit", close(memory.buffer.priorities[0], 5.01) and memory.max_error == 5.0
    )
    # E2 and E3, in a real short run: 2000 steps, of which about 200 fill the memory, so about 450 updates.
    run(
        "run_experiment.py",
        *"--env CartPole-v1 --wandb-mode offline --replay-start-size 200 --memory-size 2000".split(),
        *"--max-env-steps 2000 --train-every 4 --adam-eps 0.00015 --name _smoke_flags".split(),
    )
    agent_args = run_experiment.json.load(open(os.path.join("runs", "_smoke_flags", "env.json")))["args"]
    check("E3: the setting is recorded", agent_args["adam_eps"] == 0.00015)
    rows = open(os.path.join("runs", "_smoke_flags", "log.csv")).read().strip().split("\n")[1:]
    last = rows[-1].split(",")
    steps, updates = int(last[0]), int(last[1])
    check(
        "E2: about 1 update per 4 steps", abs(updates - (steps - 200) / 4) <= 2, f"{updates} updates at {steps} steps"
    )


def test_real_runs():
    """Short real runs of both games, vanilla and with all three techniques, plus the evaluation commands."""
    small = "--wandb-mode offline --replay-start-size 200 --memory-size 2000 --max-env-steps 1000".split()
    cartpole = ["run_experiment.py", "--env", "CartPole-v1", "--snapshot-steps", "1000", *small]
    pong = ["run_experiment.py", "--env", "ALE/Pong-v5", "--snapshot-steps", "1000", *small]
    run(*cartpole, "--name", "_smoke_cartpole")
    run(*cartpole, "--name", "_smoke_cartpole_full", "--double", "--per", "--n-step", "3")
    run(*pong, "--name", "_smoke_pong")
    run(*pong, "--name", "_smoke_pong_full", "--double", "--per", "--n-step", "3")

    for name in SMOKE_NAMES:
        folder = os.path.join("runs", name)
        files = set(os.listdir(folder))
        wanted = {"env.json", "log.csv", "checkpoints.csv", "snapshot_1000.pt"}
        check(f"{name}: files written", wanted <= files, sorted(wanted - files) or "")
        hashes = open(os.path.join(folder, "checkpoints.csv")).read()
        real = run_experiment.sha256_of(os.path.join(folder, "snapshot_1000.pt"))
        check(f"{name}: the snapshot hash is recorded", real in hashes)

    # The evaluation commands, in the spec's format, and with the TA's own test script.
    out = run("test_model_task1.py", "--model_path", "runs/_smoke_cartpole/snapshot_1000.pt", "--episodes", "2")
    check("Figure 4 format", "Environment steps: 1000, seed: 0, eval reward:" in out and "Average reward:" in out)
    out = run("test_model_task3.py", "--model_path", "runs/_smoke_pong_full/snapshot_1000.pt", "--episodes", "1")
    mine = float(out.split("eval reward: ")[1].split()[0])
    ta_out = run(
        "test_model.py",
        "--model-path",
        "runs/_smoke_pong_full/snapshot_1000.pt",
        "--episodes",
        "1",
        "--seed",
        "0",
        "--output-dir",
        "runs/_smoke_pong_full/videos",
    )
    ta = float(ta_out.split("total reward ")[1].split()[0])
    pixels = torch.full((1, 4, 84, 84), 107.0)
    model = DQN(6)
    if torch.equal(model(pixels), model.network(pixels / 255.0)):
        check("test_model.py loads the snapshot and gives the same score on seed 0", mine == ta, f"{mine} and {ta}")
    else:
        print(f"note: scores {mine} and {ta} differ only by chance until DQN.forward divides by 255")


def main():
    random.seed(0)
    os.makedirs("runs", exist_ok=True)
    for name in [*SMOKE_NAMES, "_smoke_flags"]:
        shutil.rmtree(os.path.join("runs", name), ignore_errors=True)
    test_only_todos_changed()
    test_network_shapes()
    test_per()
    test_random_shift()
    test_resets_and_weight_decay()
    test_n_step()
    test_train()
    test_experiment_flags()
    test_real_runs()
    for name in [*SMOKE_NAMES, "_smoke_flags"]:
        shutil.rmtree(os.path.join("runs", name), ignore_errors=True)
    print()
    print(f"{len(failures)} failed: {failures}" if failures else "All checks passed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
