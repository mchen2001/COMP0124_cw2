import random


class Agent:
    def __init__(self, tasks, idx):
        self.available_tasks = tasks
        self.idx = idx
        self.reward = 0

    def remove_task(self, task):
        self.available_tasks.remove(task)
        return self.available_tasks

    def get_available_task(self):
        return self.available_tasks

    def update_reward(self, payoff):
        self.reward += payoff


class RandomAgent(Agent):
    def __init__(self, tasks, idx):
        super().__init__(tasks, idx)

    def action(self):
        return random.choice(self.available_tasks)


class AdaptiveAgent(Agent):
    def __init__(self, tasks, idx, agents):
        super().__init__(tasks, idx)
        self.other_agents_rewards = {}
        for agent in agents:
            self.other_agents_rewards[agent] = 0

    def action(self):
        other_rewards = list(self.other_agents_rewards.values())
        easy_tasks = []
        medium_tasks = []
        hard_tasks = []

        for task in self.available_tasks:
            task_type = task.get_task_type()
            if task_type == 'easy':
                easy_tasks.append(task)
            elif task_type == 'medium':
                medium_tasks.append(task)
            else:
                hard_tasks.append(task)

        if self.reward >= max(other_rewards):
            if easy_tasks:
                return random.choice(easy_tasks)
            if medium_tasks:
                return random.choice(medium_tasks)
            return random.choice(hard_tasks)
        elif self.reward <= min(other_rewards):
            if hard_tasks:
                return random.choice(hard_tasks)
            if medium_tasks:
                return random.choice(medium_tasks)
            return random.choice(easy_tasks)
        else:
            if medium_tasks:
                return random.choice(medium_tasks)
            merged_tasks = easy_tasks + hard_tasks
            return random.choice(merged_tasks)

    def record_other_rewards(self, other_agent_reward_dict:dict):
        """ record other agents' reward """
        for agent in other_agent_reward_dict:
            self.other_agents_rewards[agent] += other_agent_reward_dict[agent]

