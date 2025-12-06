#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估已训练的 VEB-RL 模型，对战官方 demo 蓝方智能体。
"""

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch

# 计算目录：PROJECT_ROOT = qyPython，CODE_ROOT = 仓库根（fly 的父目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CODE_ROOT = os.path.dirname(PROJECT_ROOT)
DEFAULT_CHECKPOINT_REL = os.path.join("..", "checkpoints", "veb", "best_individual.npz")

sys.path.insert(0, PROJECT_ROOT)

from reward import RewardConfig
from veb_battle_env import VEBBattleEnvironment
from veb_individual import VEBIndividual
from env.agent.demo.demo_auto_agent import DemoAutoAgent


def resolve_device(device_arg: str) -> str:
    norm = device_arg.lower()
    if norm == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if norm == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，自动切换到 CPU")
        return "cpu"
    if norm not in {"cpu", "cuda"}:
        print(f"未识别的设备参数 {device_arg}，使用 CPU")
        return "cpu"
    return norm


def resolve_checkpoint_path(path_arg: str) -> str:
    """Convert checkpoint path to absolute; treat相对路径基于仓库根目录."""
    if os.path.isabs(path_arg):
        return path_arg
    repo_relative = os.path.normpath(os.path.join(CODE_ROOT, path_arg))
    if os.path.exists(repo_relative):
        return repo_relative
    return os.path.abspath(path_arg)


def load_best_individual(path: str, device: str) -> Tuple[VEBIndividual, torch.nn.Module]:
    data = np.load(path, allow_pickle=True)
    state_dim = int(data["state_dim"]) if "state_dim" in data else 230
    action_dim = int(data["action_dim"]) if "action_dim" in data else 60
    hidden_dims = tuple(int(x) for x in data["hidden_dims"]) if "hidden_dims" in data else (256, 256)
    use_dueling = bool(data["use_dueling"]) if "use_dueling" in data else False

    individual = VEBIndividual(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        use_dueling=use_dueling,
    )
    individual.q_weights = data["q_weights"]
    if "target_weights" in data:
        individual.target_weights = data["target_weights"]

    q_net = individual.create_q_network(device)
    q_net.eval()
    return individual, q_net


def run_episode(env: VEBBattleEnvironment, q_net: torch.nn.Module, device: str, max_steps: int) -> dict:
    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    info = {}

    while not done and steps < max_steps:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = int(q_net(state_tensor).argmax(dim=-1).item())

        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1

    return {"steps": steps, "total_reward": total_reward, "info": info}


def main():
    parser = argparse.ArgumentParser(description="Evaluate VEB-RL model against demo agent")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_REL,
                        help=f"Checkpoint path (default: {DEFAULT_CHECKPOINT_REL})，默认相对于仓库根目录")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of evaluation episodes (default: 1)")
    parser.add_argument("--max-episode-steps", type=int, default=1000,
                        help="Max steps per episode (default: 1000)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto/cpu/cuda (default: auto)")
    parser.add_argument("--no-opponent", action="store_true",
                        help="Run without demo opponent (default: False)")
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)
    print(f"使用设备: {device}")
    individual, q_net = load_best_individual(checkpoint_path, device)
    print(f"已加载模型: state_dim={individual.state_dim}, action_dim={individual.action_dim}, hidden_dims={individual.hidden_dims}")

    reward_config = RewardConfig()
    opponent = None if args.no_opponent else DemoAutoAgent("blue", "blue_demo")

    env = VEBBattleEnvironment(
        reward_config=reward_config,
        state_dim=individual.state_dim,
        action_dim=individual.action_dim,
        device=device,
        max_episode_steps=args.max_episode_steps,
        opponent_agent=opponent,
    )

    for idx in range(1, args.episodes + 1):
        print(f"\n===== 开始评估 Episode {idx} =====")
        result = run_episode(env, q_net, device, args.max_episode_steps)
        info = result["info"]
        print(f"Episode {idx} 结束: steps={result['steps']} total_reward={result['total_reward']:.2f}")
        if info.get("victory") is not None:
            outcome = "胜利" if info["victory"] else "失败"
            print(f"结果: {outcome}")
        if "summary" in info:
            summary = info["summary"]
            stats = summary.get("stats", {})
            print(f"统计: 击毁敌机(有人/无人) {stats.get('enemy_manned_killed', 0)}/{stats.get('enemy_uav_killed', 0)} | "
                  f"己方损失 {stats.get('own_manned_lost', 0)}/{stats.get('own_uav_lost', 0)}")
            print(f"中心占领比例: {stats.get('center_control_ratio', 0.0):.2%}")

    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
