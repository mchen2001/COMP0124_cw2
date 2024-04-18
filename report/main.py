import matplotlib.pyplot as plt
import os

# Assuming Env class is defined somewhere else and imported properly
from env import Env


def main():
    game_types = ['basic', 'intermediate', 'hard']
    task_volumes = [10, 50, 100]
    plots_dir = "./plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    for task_volume in task_volumes:
        for game_type in game_types:
            random_cumulative_rewards = []
            adaptive_cumulative_rewards = []
            dqn_cumulative_rewards = []
            ppo_cumulative_rewards = []
            for i in range(100):
                env = Env(game_type, task_volume, ['random', 'adaptive', 'dqn', 'ppo'])
                rewards, actions = env.simulate_game(verbose=False)
                random_cumulative_rewards.append(rewards['Random'])
                adaptive_cumulative_rewards.append(rewards['Adaptive'])
                dqn_cumulative_rewards.append(rewards['DQN'])
                ppo_cumulative_rewards.append(rewards['PPO'])

            plt.figure(figsize=(10, 6))
            plt.plot(random_cumulative_rewards, label='Random')
            plt.plot(adaptive_cumulative_rewards, label='Adaptive')
            plt.plot(dqn_cumulative_rewards, label='DQN')
            plt.plot(ppo_cumulative_rewards, label='PPO')
            plt.xlabel('Iteration')
            plt.ylabel(f'Agents Cumulative Rewards')
            plt.title(f'Cumulative Rewards Comparison - {game_type.capitalize()} Game Task Volume = {task_volume}')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{plots_dir}/{game_type}_game_rewards_plot_task_volume_{task_volume}.png")
            plt.close()


if __name__ == "__main__":
    main()
