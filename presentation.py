import random
import matplotlib.pyplot as plt
import numpy as np

# Define tasks and game configurations
tasks = {
    "easy": {"probability": 1.0, "payoff": 1},
    "medium": {"probability": 0.7, "payoff": 3},
    "hard": {"probability": 0.3, "payoff": 6}
}

game_configs = {
    "basic": {"easy": 4, "medium": 1, "hard": 1},
    "intermediate": {"easy": 2, "medium": 2, "hard": 2},
    "hard": {"easy": 1, "medium": 1, "hard": 4}
}

# Function to generate tasks based on game type and volume level
def generate_tasks(game_type, volume_level):
    config = game_configs[game_type]
    task_list = []
    for task_type, count in config.items():
        task_list.extend([task_type] * count * volume_level)
    random.shuffle(task_list)
    return task_list

# Function to simulate the game with given strategies
def simulate_game(task_list, *strategy_funcs):
    num_players = len(strategy_funcs)
    scores = [0] * num_players
    remaining_tasks = task_list.copy()
    action_history = {player_id: [] for player_id in range(1, num_players + 1)}

    while remaining_tasks:
        opponent_scores = [[scores[j] for j in range(num_players) if j != i] for i in range(num_players)]
        chosen_tasks = [strategy(remaining_tasks, scores[player_id], opponent_scores[player_id])
                        if strategy.__name__ == "adaptive_strategy"
                        else strategy(remaining_tasks, scores[player_id])
                        for player_id, strategy in enumerate(strategy_funcs)]

        unique_tasks = set(chosen_tasks)

        for task in unique_tasks:
            if task is None or task not in remaining_tasks:
                continue

            players_choosing_task = [i for i, chosen_task in enumerate(chosen_tasks) if chosen_task == task]

            if len(players_choosing_task) > 1:
                winner = random.choice(players_choosing_task)
                success = random.random() <= tasks[task]["probability"]
                scores[winner] += tasks[task]["payoff"] if success else 0
                action_history[winner + 1].append((task, 'Success' if success else 'Failure'))
                for player in players_choosing_task:
                    if player != winner:
                        action_history[player + 1].append((task, 'Conflict'))
            else:
                player = players_choosing_task[0]
                success = random.random() <= tasks[task]["probability"]
                scores[player] += tasks[task]["payoff"] if success else 0
                action_history[player + 1].append((task, 'Success' if success else 'Failure'))

            remaining_tasks.remove(task)

    return scores, action_history

# Define strategy functions
def risk_averse(available_tasks, current_score):
    return max(available_tasks, key=lambda task: tasks[task]["probability"], default=None)

def opportunistic_strategy(available_tasks, current_score):
    return max(available_tasks, key=lambda task: tasks[task]["payoff"], default=None)

def random_strategy(available_tasks, current_score):
    return random.choice(available_tasks) if available_tasks else None

def balanced_strategy(available_tasks, current_score):
    return max(available_tasks, key=lambda task: tasks[task]["probability"] * tasks[task]["payoff"], default=None)

def adaptive_strategy(available_tasks, current_score, opponent_scores):
    if current_score > max(opponent_scores, default=0):
        return risk_averse(available_tasks, current_score)
    elif current_score < min(opponent_scores, default=0):
        return opportunistic_strategy(available_tasks, current_score)
    else:
        return balanced_strategy(available_tasks, current_score)

# Function to run the game and produce comprehensive visuals for different volumes and game modes
def run_game_and_produce_multiple_visuals(agents, strategies, game_modes, volumes, iterations=200):
    for game_mode in game_modes:
        for volume in volumes:
            all_scores = np.zeros((len(agents), iterations))
            for i in range(iterations):
                task_list = generate_tasks(game_mode, volume)
                scores, _ = simulate_game(task_list, *agents)
                all_scores[:, i] = scores

            # Calculate and print expected rewards for each task
            expected_rewards = {task: info["probability"] * info["payoff"] for task, info in tasks.items()}
            print(f"Game Mode: {game_mode}, Volume: {volume}")
            print(f"Expected Rewards: {expected_rewards}")

            # Visualizations
            fig, axs = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 2]})

            # Boxplot for reward distribution
            axs[0].boxplot(all_scores.T, labels=strategies, showmeans=True)
            axs[0].set_title(f"Reward Distribution by Strategy in {game_mode.capitalize()} Mode with Volume {volume}")
            axs[0].set_ylabel("Reward")

            # Line plot for average reward trends
            axs[1].set_title(f"Reward Trends by Strategy over {iterations} Iterations")
            axs[1].set_xlabel("Iteration")
            axs[1].set_ylabel("Reward")
            for i, strategy in enumerate(strategies):
                # Calculate the average score up to each iteration for the strategy
                average_scores = np.cumsum(all_scores[i, :]) / (np.arange(iterations) + 1)
                axs[1].plot(average_scores, label=strategy)
            axs[1].legend()

            plt.tight_layout()
            plt.savefig(f"reward_distribution_{game_mode}_volume_{volume}.png")
            plt.show()
            print(f"Finished simulation for game mode: {game_mode}, volume: {volume}")


agents = [risk_averse, opportunistic_strategy, random_strategy, balanced_strategy, adaptive_strategy]
strategies = ["Risk Averse", "Opportunistic", "Random", "Balanced", "Adaptive"]
game_mode = ["basic", "intermediate", "hard"]
volume_level = [6, 12]

run_game_and_produce_multiple_visuals(agents, strategies, game_mode, volume_level)
