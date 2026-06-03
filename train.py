import os
import csv
import glob
import argparse
import numpy as np

from carla_env import CarlaEnv
from agent import DQNAgent, DDQNAgent


# Hyperparameters
GAMMA = 0.99
LR = 5e-4
BATCH_SIZE = 32
NUM_FRAMES = 8
EPSILON_START = 1.0
EPSILON_MIN = 0.0001
BUFFER_SIZE = 400_000
MIN_REPLAY_SIZE = 100_000

AGENT_CONFIGS = {
    "dqn": {
        "class": DQNAgent,
        "carla_port": 3000,
        "tm_port": 9000,
        "model_dir": "models_dqn",
        "log_file": "training_log_dqn.csv",
        "buffer_dir": "replay_buffer_dqn",
    },
    "ddqn": {
        "class": DDQNAgent,
        "carla_port": 2000,
        "tm_port": 8000,
        "model_dir": "models",
        "log_file": "training_log_ddqn.csv",
        "buffer_dir": "replay_buffer",
    },
}


def parse_args(default_agent="dqn"):
    parser = argparse.ArgumentParser(description="Train a DQN or DDQN agent in CARLA")
    parser.add_argument("--agent", choices=sorted(AGENT_CONFIGS), default=default_agent)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--carla-port", type=int)
    parser.add_argument("--tm-port", type=int)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--model-dir", type=str)
    parser.add_argument("--log-file", type=str)
    parser.add_argument("--buffer-dir", type=str)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = AGENT_CONFIGS[args.agent]
    args.carla_port = args.carla_port if args.carla_port is not None else config["carla_port"]
    args.tm_port = args.tm_port if args.tm_port is not None else config["tm_port"]
    args.model_dir = args.model_dir or config["model_dir"]
    args.log_file = args.log_file or config["log_file"]
    args.buffer_dir = args.buffer_dir or config["buffer_dir"]
    return args


def main(default_agent="dqn"):
    args = parse_args(default_agent)
    config = AGENT_CONFIGS[args.agent]
    agent_class = config["class"]
    checkpoint_prefix = args.agent

    os.makedirs(args.model_dir, exist_ok=True)

    env = CarlaEnv(port=args.carla_port, tm_port=args.tm_port)

    agent = agent_class(
        num_actions=env.action_space,
        num_frames=NUM_FRAMES,
        gamma=GAMMA,
        lr=LR,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        min_replay_size=MIN_REPLAY_SIZE,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        episodes=args.episodes,
        buffer_dir=args.buffer_dir,
    )

    episode_rewards = []
    best_avg50 = float("-inf")
    total_steps = 0

    if not args.no_resume:
        checkpoints = sorted(glob.glob(os.path.join(args.model_dir, f"{checkpoint_prefix}_ep*.pth")))
        if checkpoints:
            latest = checkpoints[-1]
            agent.load(latest)
            print(f"Resumed from {latest} (episode {agent.episode_count}, eps {agent.epsilon:.4f})")

        # Restore running state from existing log so avg50, best_avg50,
        # and total_steps are all consistent with the previous run.
        if os.path.isfile(args.log_file):
            with open(args.log_file, newline="") as f:
                prior_rows = list(csv.reader(f))[1:]  # skip header
            if prior_rows:
                episode_rewards = [float(r[3]) for r in prior_rows[-50:]]
                best_avg50 = max(float(r[4]) for r in prior_rows)
                total_steps = int(prior_rows[-1][2])
                print(
                    f"Restored log state: total_steps={total_steps}, "
                    f"best_avg50={best_avg50:.0f}, "
                    f"seeding avg50 with last {len(episode_rewards)} rewards"
                )

    log_exists = os.path.isfile(args.log_file)
    log_file = open(args.log_file, "a", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(["episode", "steps", "total_steps", "reward", "avg50", "loss", "epsilon"])

    try:
        episode = agent.episode_count + 1
        while episode <= args.episodes:
            state = env.reset()
            done = False
            ep_reward = 0
            ep_steps = 0
            losses = []

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)

                agent.store(state, action, float(reward), next_state, done)
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss.item() if hasattr(loss, "item") else loss)

                state = next_state
                ep_reward += reward
                ep_steps += 1

            total_steps += ep_steps

            agent.end_episode()

            episode_rewards.append(ep_reward)

            avg50 = np.mean(episode_rewards[-50:])
            avg_loss = np.mean(losses) if losses else 0.0

            print(
                f"Ep {episode:4d}/{args.episodes} | "
                f"Steps: {ep_steps:4d} | "
                f"TotalSteps: {total_steps:7d} | "
                f"Reward: {ep_reward:6.0f} | "
                f"Avg50: {avg50:6.0f} | "
                f"Loss: {avg_loss:.6f} | "
                f"Eps: {agent.epsilon:.4f}"
            )

            log_writer.writerow([
                episode,
                ep_steps,
                total_steps,
                ep_reward,
                round(avg50, 1),
                avg_loss,
                round(agent.epsilon, 4),
            ])
            log_file.flush()

            if avg50 > best_avg50:
                best_avg50 = avg50
                agent.save(os.path.join(args.model_dir, f"{checkpoint_prefix}_best.pth"))
                print(f"  -> New best avg50: {best_avg50:.0f}")

            if episode % args.save_every == 0:
                path = os.path.join(args.model_dir, f"{checkpoint_prefix}_ep{episode}.pth")
                agent.save(path)
                print(f"  -> Saved {path}")

            episode += 1

    finally:
        log_file.close()
        env.destroy()


if __name__ == "__main__":
    main()
