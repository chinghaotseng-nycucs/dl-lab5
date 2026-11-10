# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DQN(nn.Module):
    """
    Design the architecture of your deep Q network
    - Input size is the same as the state dimension; the output size is the same as the number of actions
    - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
    - Feel free to add any member variables/functions whenever needed
    """

    def __init__(self, num_actions):
        super(DQN, self).__init__()
        # An example:
        # self.network = nn.Sequential(
        #    nn.Linear(input_dim, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, num_actions)
        # )
        ########## YOUR CODE HERE (5~10 lines) ##########
        # The template passes only num_actions, so it also tells the two games apart:
        # CartPole-v1 has 2 actions, ALE/Pong-v5 has 6 (env.action_space.n).
        # B = batch size in the shapes below.
        if num_actions == 2:
            # CartPole (Task 1): the state is 4 numbers. The size is my choice (spec 2.1).
            self.input_scale = 1.0  # the state is not a picture, so forward() must not scale it
            self.network = nn.Sequential(  # CORE: network layers
                nn.Linear(4, 64),  # (B, 4) -> (B, 64)
                nn.ReLU(),
                nn.Linear(64, 64),  # (B, 64) -> (B, 64)
                nn.ReLU(),
                nn.Linear(64, num_actions),  # (B, 64) -> (B, 2), one Q-value per action
            )
        else:
            # Pong (Tasks 2 and 3): the same layers, in the same order, as test_model.py.
            # test_model.py loads weights by name (network.0, network.2, ...), so the order must match.
            self.input_scale = 255.0  # pixels are stored as 0-255, and the network must see 0-1
            # width multiplies every layer's size (Task 3 report W). 1 = the same network as test_model.py.
            # DQN.width is set by run_experiment.py and eval_snapshot.py, so DQN(num_actions) keeps its arguments.
            w = getattr(DQN, "width", 1)
            self.network = nn.Sequential(  # CORE: network layers
                nn.Conv2d(4, 32 * w, kernel_size=8, stride=4),  # (B, 4, 84, 84) -> (B, 32w, 20, 20)
                nn.ReLU(),
                nn.Conv2d(32 * w, 64 * w, kernel_size=4, stride=2),  # -> (B, 64w, 9, 9)
                nn.ReLU(),
                nn.Conv2d(64 * w, 64 * w, kernel_size=3, stride=1),  # -> (B, 64w, 7, 7)
                nn.ReLU(),
                nn.Flatten(),  # -> (B, 3136w); 64 * 7 * 7 = 3136
                nn.Linear(64 * w * 7 * 7, 512 * w),  # -> (B, 512w)
                nn.ReLU(),
                nn.Linear(512 * w, num_actions),  # -> (B, 6), one Q-value per action
            )
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        return self.network(
            x / self.input_scale
        )  # CORE: scale pixels to 0-1. [MY WORK] was self.network(x); input_scale is 255 for Pong, 1 for CartPole, so training matches test_model.py


class AtariPreprocessor:
    """
    Preprocesing the state input of DQN for Atari
    """

    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
    Prioritizing the samples in the replay memory by the Bellman error
    See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ##########
        # Fundamentals page 8, Step 5. 0.01 = the spec's epsilon: a zero error can still be drawn (spec section 1).
        priority = abs(error) + 0.01  # CORE: priority
        if not hasattr(self, "scaled"):
            # Made on the first add, because __init__ is outside the TODOs. scaled[i] = priority ** alpha, kept
            # up to date here and in update_priorities(), so sample() does not redo the power over the whole buffer.
            self.scaled = np.zeros((self.capacity,), dtype=np.float64)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)  # still filling up
        else:
            self.buffer[self.pos] = transition  # full: overwrite the oldest
        self.priorities[self.pos] = priority
        self.scaled[self.pos] = float(self.priorities[self.pos]) ** self.alpha  # the spec's p_i ** alpha
        self.pos = (self.pos + 1) % self.capacity  # ring: the position wraps back to 0
        ########## END OF YOUR CODE (for Task 3) ##########
        return

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########
        # Fundamentals page 8, Steps 2 and 3. B = batch_size.
        n = len(self.buffer)  # how many transitions are stored (the spec's N)
        scaled = self.scaled[:n]  # (n) p_i ** alpha, stored by add() and update_priorities(); [:n] = the filled part
        # A roulette wheel: transition i owns a segment of length scaled[i] on the line from 0 to the total.
        cumulative = np.cumsum(scaled)  # (n) running total [s0, s0+s1, ...]; cumulative[-1] = the total
        darts = np.random.random(batch_size) * cumulative[-1]  # (B) random points on [0, total)
        # Binary search: the first i whose running total passes the point, so i is drawn with chance scaled[i] / total.
        indices = np.minimum(np.searchsorted(cumulative, darts, side="right"), n - 1)  # CORE: sampling chance; (B)
        chances = scaled[indices] / cumulative[-1]  # (B) the spec's P(i) of the drawn transitions
        weights = (n * chances) ** (-self.beta)  # CORE: IS weights; (B)
        weights = weights / weights.max()  # normalize, so that a weight can only shrink a step (spec section 1)
        batch = [self.buffer[i] for i in indices]  # list comprehension: the B chosen transitions
        return batch, indices, weights.astype(np.float32)
        ########## END OF YOUR CODE (for Task 3) ##########
        return

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########
        # Fundamentals page 8, Step 5. zip walks the two lists side by side.
        for i, error in zip(indices, errors):
            self.priorities[i] = abs(error) + 0.01  # the same 0.01 as in add()
            self.scaled[i] = float(self.priorities[i]) ** self.alpha  # keep the stored p_i ** alpha in step
        ########## END OF YOUR CODE (for Task 3) ##########
        return


class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.q_net = DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = 0  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()

            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_state = self.preprocessor.step(next_obs)
                self.memory.append((state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:
                    print(
                        f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}"
                    )
                    wandb.log(
                        {
                            "Episode": ep,
                            "Step Count": step_count,
                            "Env Step Count": self.env_count,
                            "Update Count": self.train_count,
                            "Epsilon": self.epsilon,
                        }
                    )
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed
                    # Spec 4.1, "how you use Weight & Bias". The self.last_ values are set at the end of train().
                    wandb.log(
                        {
                            "Env Step Count": self.env_count,
                            "Loss": self.last_loss,
                            "Q Mean": self.last_q_mean,
                            "Bellman Error Mean": self.last_error_mean,
                        }
                    )
                    # env_count is a multiple of 1000 here, and so are 600k, 1M, ... (spec 4.2.2, Task 3).
                    self.save_files_at_this_step()  # in run_experiment.py
                    # Task 3 needs an exact step count, and the for loop above only counts games.
                    if self.env_count >= self.max_env_steps:  # max_env_steps: in run_experiment.py
                        return
                    ########## END OF YOUR CODE ##########
            print(
                f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}"
            )
            wandb.log(
                {
                    "Episode": ep,
                    "Total Reward": total_reward,
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Epsilon": self.epsilon,
                }
            )
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed

            ########## END OF YOUR CODE ##########
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log(
                    {"Env Step Count": self.env_count, "Update Count": self.train_count, "Eval Reward": eval_reward}
                )

    def evaluate(self):
        obs, _ = self.test_env.reset()
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

        return total_reward

    def train(self):

        if len(self.memory) < self.replay_start_size:
            return

        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        # self.memory draws uniformly, or by priority with --per (ReplayMemory in run_experiment.py).
        batch, indices, weights = self.memory.sample(self.batch_size)
        # zip(*batch): a list of B 5-tuples -> 5 tuples of length B (array of structs -> struct of arrays).
        states, actions, rewards, next_states, dones = zip(*batch)
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # CORE: Q-value of the taken action

        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates
        # Step numbers follow fundamentals page 6. Every vector below has shape (B), B = batch size.
        # Step 1: the target. no_grad: it is a fixed label, so no gradient flows through it.
        with torch.no_grad():
            if self.double:
                # Double DQN (page 7): the main network chooses, and the target network gives the value.
                next_actions = self.q_net(next_states).argmax(1)  # (B, num_actions) -> (B)
                next_q = (
                    self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                )  # CORE: Double DQN target
            else:
                next_q = self.target_net(next_states).max(1)[0]  # .max(1)[0]: the largest value in each row
            # rewards already hold the n-step sum, and gamma ** n_step is plain gamma when n_step is 1 (page 9).
            targets = rewards + self.gamma**self.n_step * (1 - dones) * next_q  # CORE: Bellman target
        # Step 2: the Bellman error.
        errors = targets - q_values  # CORE: Bellman error and loss
        # Step 3: the loss. Without --per all weights are 1, so this is plain MSE (Mean Squared Error).
        # The spec's formula (1) is half the sum. This mean differs from it only by the constant B / 2.
        weights = torch.from_numpy(weights).to(self.device)  # numpy array -> tensor, or the product below fails
        loss = (weights * errors.pow(2)).mean()  # CORE: weighted loss; one number
        # Step 5: one update.
        self.optimizer.zero_grad()  # backward() adds to the old gradients, so clear them first
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)  # my choice: limits rare large gradients
        self.optimizer.step()
        # PER (page 8, Step 5): write the new error sizes back. This does nothing without --per.
        self.memory.update_priorities(indices, errors.detach().abs().cpu().numpy())
        # Kept for the wandb log in run(). .item(): a one-number tensor -> a Python number.
        self.last_loss = loss.item()
        self.last_q_mean = q_values.mean().item()
        self.last_error_mean = errors.abs().mean().item()
        ########## END OF YOUR CODE ##########

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(
                f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run()
