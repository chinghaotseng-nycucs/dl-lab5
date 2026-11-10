"""Evaluate one snapshot on 20 games, seeds 0 to 19 (spec 4.2.2), and print it like the spec's Figure 4.

Extra, not required as a file: the spec only asks for a runnable command. test_model_task1.py, test_model_task2.py
and test_model_task3.py call this, so that each task has one command:
    python test_model_task3.py --model_path LAB5_315551118_task3_2500000.pt
Add --video-dir videos to save one mp4 per game, for the demo video.
"""

import argparse
import os
import re

import gymnasium as gym
import imageio
import numpy as np
import torch

from dqn import DQN, AtariPreprocessor
from run_experiment import PassThroughPreprocessor


def steps_from_name(path):
    """Read the step count from a file name: LAB5_x_task3_600000.pt -> '600000'. Other names give 'best'."""
    match = re.search(r"_(\d+)\.pt$", os.path.basename(path))  # \d+ = one or more digits at the end of the name
    return match.group(1) if match else "best"


def evaluate_snapshot(env_name):
    """Play the games and print one line per seed, then the average."""
    parser = argparse.ArgumentParser(description=f"Evaluate a snapshot on {env_name}")
    parser.add_argument("--model_path", "--model-path", type=str, required=True)  # the spec's example uses --model_path
    parser.add_argument("--episodes", type=int, default=20)  # default 20 (spec 4.2.2)
    parser.add_argument("--first-seed", type=int, default=0)  # default 0: seeds 0 to 19 (spec 4.2.2)
    parser.add_argument("--video-dir", type=str, default="")  # default none: no videos
    parser.add_argument("--env-steps", type=str, default="")  # default: read from the file name, else "best"
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name, render_mode="rgb_array")
    preprocessor = AtariPreprocessor() if "Pong" in env_name else PassThroughPreprocessor()
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    if "Pong" in env_name:
        # TA announcement 11/10: the network may differ from test_model.py. The width is read from the weights.
        DQN.width = state_dict["network.0.weight"].shape[0] // 32
    model = DQN(env.action_space.n).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)

    steps = args.env_steps or steps_from_name(args.model_path)  # TA announcement 11/10: print the training steps
    scores = []
    for seed in range(args.first_seed, args.first_seed + args.episodes):
        obs, _ = env.reset(seed=seed)  # the same seed repeats the same game, so the TAs can reproduce it
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0
        frames = []
        while not done:
            if args.video_dir:
                frames.append(env.render())
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()  # no random actions in evaluation
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocessor.step(next_obs)
        scores.append(total_reward)
        print(f"Environment steps: {steps}, seed: {seed}, eval reward: {total_reward:g}", flush=True)
        if args.video_dir:
            with imageio.get_writer(os.path.join(args.video_dir, f"seed{seed}.mp4"), fps=30) as video:
                for frame in frames:
                    video.append_data(frame)
    print(f"Average reward: {np.mean(scores):g}")
