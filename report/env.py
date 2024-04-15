import random

from game import *
from agent import *


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
                agent = DQNAgent(idx, tasks, len(tasks), 3)  # Assuming three types of tasks
                self.agents.append(agent)
            else:
                raise Exception("Unknown agent type")

    def step(self, agent_actions):
        task_agent = {}
        for agent, task in agent_actions.items():
            task_idx = task.idx
            if task_idx not in task_agent:
                task_agent[task_idx] = [agent]
            else:
                task_agent[task_idx].append(agent)

        agent_reward = {agent.idx: agent.reward for agent in self.agents}  # Initialize rewards from current agent state

        for idx, agents in task_agent.items():
            task = self.game.get_task_by_idx(idx)
            payoff = task.do_task()

            if payoff == 0:
                for agent in agents:
                    print(f"Task {task.idx} not completed by agent {agent.idx}")
            else:
                # Randomly select an agent who attempted the task to receive the reward
                agent = random.choice(agents)
                agent.update_reward(payoff)
                agent_reward[agent.idx] += payoff
                print(f"Task {task.idx} completed by agent {agent.idx}, Reward: {payoff}")
                task.done = True

        self.game.update_tasks()

        for agent in self.agents:
            agent.update_tasks([task for task in self.game.get_game_tasks()])

        return agent_reward


    def simulate_game(self):
        self.reset()
        print("Agents:", [agent.idx for agent in self.agents])
        action_history = {agent.idx: [] for agent in self.agents}
        cumulative_rewards = {agent.idx: 0 for agent in self.agents}  # Initialize cumulative rewards to zero

        while any(not task.is_completed() for task in self.game.get_game_tasks()):
            print(f"Tasks: {[task.idx for task in self.game.get_game_tasks() if not task.is_completed()]}")
            agent_actions = {}
            for agent in self.agents:
                task = agent.act()
                if task:
                    agent_actions[agent] = task

            for agent in agent_actions:
                task = agent_actions[agent]
                payoff = task.do_task()
                if hasattr(agent, 'update_reward'):
                    agent.update_reward(payoff)
                action_history[agent.idx].append(f"Task {task.idx}")
                if task.is_completed():
                    print(f"Task {task.idx} completed by agent {agent.idx}")
                else:
                    print(f"Task {task.idx} not completed by agent {agent.idx}")
                cumulative_rewards[agent.idx] += payoff  # Update cumulative rewards

            # Print cumulative rewards after each round
            print(f"Cumulative Rewards after this round: {cumulative_rewards}")

            # Update tasks for the next round
            self.game.update_tasks()
            for agent in self.agents:
                if hasattr(agent, 'update_tasks'):
                    agent.update_tasks(self.game.get_game_tasks())

        return cumulative_rewards, action_history

