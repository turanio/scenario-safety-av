import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    data = np.load(args.artifact)

    scenario_id = str(data["scenario_id"])
    target_actor_id = str(data["target_actor_id"])

    positions = data["positions"]
    probabilities = data["probabilities"]

    ego_history = data["ego_history_positions"]
    ego_future = data["ego_future_positions"]
    target_history = data["target_history_positions"]
    target_future = data["target_future_positions"]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    plt.figure(figsize=(8, 7))

    plt.plot(ego_history[:, 0], ego_history[:, 1], linewidth=2, label="Ego history")
    plt.plot(ego_future[:, 0], ego_future[:, 1], linewidth=2, linestyle="--", label="Ego future")

    plt.plot(target_history[:, 0], target_history[:, 1], linewidth=2, label="Target history")
    plt.plot(target_future[:, 0], target_future[:, 1], linewidth=2, linestyle="--", label="Target ground truth future")

    top1_mode = int(np.argmax(probabilities))

    for mode_idx in range(positions.shape[0]):
        label = f"QCNet mode {mode_idx} p={probabilities[mode_idx]:.2f}"
        linewidth = 2.5 if mode_idx == top1_mode else 1.2
        alpha = 1.0 if mode_idx == top1_mode else 0.65
        plt.plot(
            positions[mode_idx, :, 0],
            positions[mode_idx, :, 1],
            linewidth=linewidth,
            alpha=alpha,
            label=label,
        )

    plt.scatter(ego_future[0, 0], ego_future[0, 1], marker="o", s=50, label="Ego current")
    plt.scatter(target_history[-1, 0], target_history[-1, 1], marker="x", s=60, label="Target current")

    plt.axis("equal")
    plt.grid(True, alpha=0.3)

    title = args.title or f"QCNet multimodal predictions\nScenario {scenario_id}, target {target_actor_id}"
    plt.title(title)
    plt.xlabel("x position, m")
    plt.ylabel("y position, m")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
