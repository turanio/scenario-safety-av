from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    summary_path = Path("results/qcnet_server_500/qcnet_server_500_probability_threshold_summary.csv")
    out_dir = Path("results/qcnet_server_500/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)

    # Plot 1: brake counts
    plt.figure(figsize=(8, 5))
    plt.plot(df["probability_threshold"], df["probability_aware_brake_count"], marker="o", label="Probability-aware")
    plt.plot(df["probability_threshold"], df["top1_brake_count"], marker="o", label="Top-1")
    plt.plot(df["probability_threshold"], df["worst_case_brake_count"], marker="o", label="Worst-case")
    plt.plot(df["probability_threshold"], df["ground_truth_near_miss_count"], marker="o", label="Ground truth near miss")
    plt.xlabel("Probability threshold")
    plt.ylabel("Brake / near-miss count")
    plt.title("Safety-filter decisions across probability thresholds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "probability_threshold_brake_counts.png", dpi=200)
    plt.close()

    # Plot 2: hidden risk vs missed worst-case
    plt.figure(figsize=(8, 5))
    plt.plot(df["probability_threshold"], df["hidden_risk_detected_count"], marker="o", label="Hidden-risk detections")
    plt.plot(df["probability_threshold"], df["missed_worst_case_brake_count"], marker="o", label="Missed worst-case brake cases")
    plt.xlabel("Probability threshold")
    plt.ylabel("Scenario count")
    plt.title("Hidden-risk detection versus missed worst-case risk")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "probability_threshold_tradeoff.png", dpi=200)
    plt.close()

    # Plot 3: mean eligible modes
    plt.figure(figsize=(8, 5))
    plt.plot(df["probability_threshold"], df["mean_eligible_mode_count"], marker="o")
    plt.xlabel("Probability threshold")
    plt.ylabel("Mean eligible mode count")
    plt.title("Number of retained prediction modes across thresholds")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "probability_threshold_mean_eligible_modes.png", dpi=200)
    plt.close()

    print("Created figures:")
    for path in sorted(out_dir.glob("probability_threshold_*.png")):
        print(path)


if __name__ == "__main__":
    main()
