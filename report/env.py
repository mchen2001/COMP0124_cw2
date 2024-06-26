import random

from game import *
from agent import *
from utils import *


class Env:
    def __init__(self, game_type, task_volume, agent_list, gain=0.1):  # agent list is a list of str
        self.game = None
        self.game_type = game_type
        self.task_volume = task_volume
        self.gain = gain
        self.agents = []
        self.agent_list = agent_list
        self.reset()

    def reset(self):
        # This part of code remains largely unchanged
        self.game = Game(self.game_type, self.task_volume, self.gain)
        self.agents = []
        tasks = list(self.game.get_game_tasks())
        for idx, agent_type in enumerate(self.agent_list):
            if agent_type == 'adaptive':
                agent = AdaptiveAgent(idx, tasks)
                agent.initialize_other_agents_rewards(list(range(len(self.agent_list))))
                self.agents.append(agent)
            elif agent_type == 'random':
                self.agents.append(RandomAgent(idx, tasks))
            elif agent_type == 'dqn':
                # Initialize DQN agent with the size of the game's tasks and actions
                agent = DQNAgent(idx, tasks)  # Assuming three types of tasks
                self.agents.append(agent)
            elif agent_type == 'ppo':
                agent = PPOAgent(idx, tasks)
                self.agents.append(agent)

            else:
                raise Exception("Unknown agent type")

    def step(self, agent_actions):
        task_agent = {}
        agent_reward = {}
        for agent, task in agent_actions.items():
            task_idx = task.idx
            if task_idx not in task_agent:
                task_agent[task_idx] = [agent]
            else:
                task_agent[task_idx].append(agent)
        for task_idx, agents in task_agent.items():
            task = self.game.get_task_by_idx(task_idx)
            payoff = task.do_task()
            if payoff > 0:
                for agent in agents:
                    agent.update_reward(payoff)
                    agent_reward[agent.idx] = payoff
                task.done = True
            else:
                for agent in agents:
                    agent_reward[agent.idx] = 0

        self.game.update_tasks()
        for agent in self.agents:
            if hasattr(agent, 'update_tasks'):
                agent.update_tasks(self.game.get_game_tasks())

        done = not any(not task.is_completed() for task in self.game.get_game_tasks())
        return agent_reward, done

    def simulate_game(self, verbose=True):
        self.reset()
        agent_types = ['Random', 'Adaptive', 'DQN', 'PPO']  # Adjust this list according to the order in `agent_list`
        print("Agents:", {idx: f"{type}" for idx, type in enumerate(agent_types)})
        action_history = {agent.idx: [] for agent in self.agents}
        cumulative_rewards = {agent.idx: 0 for agent in self.agents}

        while any(not task.is_completed() for task in self.game.get_game_tasks()):
            print_with_verbosity(
                f"Tasks available: {[task.idx for task in self.game.get_game_tasks() if not task.is_completed()]}",
                verbose)
            task_agent_map = {}
            for agent in self.agents:
                task = agent.act(verbose)
                if task:
                    if task.idx not in task_agent_map:
                        task_agent_map[task.idx] = []
                    task_agent_map[task.idx].append(agent)
                    action_history[agent.idx].append(f"Task{task.idx}")

            for task_idx, agents in task_agent_map.items():
                task = self.game.get_task_by_idx(task_idx)
                payoff = task.do_task()
                if payoff > 0:
                    for agent in agents:
                        agent.update_reward(payoff)
                        cumulative_rewards[agent.idx] += payoff
                        print_with_verbosity(f"Task {task_idx} completed by {agent_types[agent.idx]}, Reward: {payoff}",
                                             verbose)
                else:
                    for agent in agents:
                        print_with_verbosity(f"Task {task_idx} not completed by {agent_types[agent.idx]}",
                                             verbose)

            rewards_by_name = {agent_types[agent.idx]: cumulative_rewards[agent.idx] for agent in self.agents}
            print_with_verbosity(f"Cumulative Rewards after this round: {rewards_by_name}",
                                 verbose)
            self.game.update_tasks()
            for agent in self.agents:
                if hasattr(agent, 'update_tasks'):
                    agent.update_tasks(self.game.get_game_tasks())

        final_rewards_by_name = {agent_types[agent.idx]: cumulative_rewards[agent.idx] for agent in self.agents}
        return final_rewards_by_name, action_history
