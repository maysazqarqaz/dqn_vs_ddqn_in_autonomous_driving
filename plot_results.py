import csv
import argparse
import matplotlib.pyplot as plt

SINGLE_LOG   = "training_log.csv"

def load_single(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "ep_reward": float(row["reward"]),
                "avg50":     float(row["avg50"]),
            })
    return rows


def plot(rows, title, save_path):
    episodes = list(range(1, len(rows) + 1))
    rewards  = [r["ep_reward"] for r in rows]
    avg50s   = [r["avg50"]     for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax1.plot(episodes, rewards, alpha=0.3, color="steelblue", linewidth=0.8)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Episode Reward")
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, avg50s, color="crimson", linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Avg-50 reward")
    ax2.set_title("Avg-50 Reward Trend")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",  default=None, help="Path to CSV log file")
    parser.add_argument("--out",  default=None, help="Output image path")
    args = parser.parse_args()

    path = args.log or SINGLE_LOG
    rows = load_single(path)
    out  = args.out or "training_results_single.png"
    title = "Training — Episode Reward"

    print(f"Loaded {len(rows)} episodes from {path}")
    plot(rows, title, out)
