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
        self.game = Game(self.game_type, self.task_volume, self.gain)
        self.agents = []
        tasks = self.game.tasks
        for idx, agent in enumerate(self.agent_list):
            if agent == 'adaptive':
                aa = AdaptiveAgent(idx, tasks)
                aa.initialize_other_agents_rewards(list(range(len(self.agent_list))))
                self.agents.append(aa)
            elif agent == 'random':
                self.agents.append(RandomAgent(idx, tasks))
            else:
                Exception("Place holder for other agents")

    def step(self, agent_actions):
        task_agent = {}
        for agent in agent_actions:
            if agent_actions[agent].idx not in task_agent:
                task_agent[agent_actions[agent].idx] = [agent]
            else:
                task_agent[agent_actions[agent].idx].append(agent)

        agent_reward = {}
        for idx in task_agent:
            task = self.game.get_task_by_idx(idx)
            payoff = task.do_task()
            agent = random.choice(task_agent[idx])
            if payoff == 0:
                # print(task.idx)
                # print([t.idx for t in agent.available_tasks])
                print(f"Task {task.idx} not completed by agent {agent.idx}")
                # agent.remove_task(task)
            else:
                agent.update_reward(payoff)
                agent_reward[agent.idx] = payoff
                print(f"Task {task.idx} completed by agent {agent.idx}")
                task.done = True

        self.game.update_tasks()
        for agent in self.agents:
            if isinstance(agent, AdaptiveAgent):
                agent.record_other_rewards(agent_reward)
            agent.update_tasks(self.game.tasks)
            # agent.available_tasks = list(set(agent.available_tasks) & set(self.game.tasks))
        # for agent in self.agents:
        #     agent.update_available_tasks(self.game.get_game_tasks())

    def simulate_game(self):
        self.reset()
        print("agents:", [agent.idx for agent in self.agents])
        action_history = {agent.idx: [] for agent in self.agents}
        agent_reward = {agent.idx: [0] for agent in self.agents}
        agent_tasks = self.game.tasks
        while self.game.tasks and agent_tasks:
            print(f"Tasks: {[task.idx for task in self.game.tasks]}")
            agent_actions = {}
            for agent in self.agents:
                act = agent.action()
                if act is not None:
                    agent_actions[agent] = act
            for agent in agent_actions:
                action_history[agent.idx].append(f"Task{agent_actions[agent].idx}")
            self.step(agent_actions)
            for agent in self.agents:
                agent_reward[agent.idx].append(agent.reward)
            agent_tasks = set()
            for agent in self.agents:
                agent_tasks.update(agent.available_tasks)

        return agent_reward, action_history


