import random


class Agent:
    def __init__(self, idx, tasks):
        self.available_tasks = set(tasks)
        self.idx = idx
        self.reward = 0

    def update_tasks(self, tasks):
        agent_tasks = self.available_tasks.copy()
        self.available_tasks = agent_tasks & set(tasks)
        # print("!!!:", agent_tasks)
        # idx = [task.idx for task in tasks]
        # self.available_tasks = set([task for task in agent_tasks if task.idx in idx])

    # def remove_task(self, task):
    #     for t in self.available_tasks:
    #         if t.idx == task.idx:
    #             self.available_tasks.remove(t)
    # self.available_tasks.remove(task)
    # return self.available_tasks

    def get_available_task(self):
        return self.available_tasks

    def update_reward(self, payoff):
        self.reward += payoff

    # def update_available_tasks(self, available_tasks):
    #     # Assuming the agent has an attribute that tracks available task indices
    #     self.available_task_indices = [task.idx for task in available_tasks]

    def action(self):
        pass

    def __eq__(self, other):
        if not isinstance(other, Agent):
            return NotImplemented

        return self.idx == other.idx

    def __hash__(self):
        return hash(self.idx)


class RandomAgent(Agent):
    def __init__(self, idx, tasks):
        super().__init__(idx, tasks)

    def action(self):
        print(f"available task for agent {self.idx}: {[t.idx for t in self.available_tasks]}")
        if not self.available_tasks:
            return None
        a = random.choice(list(self.available_tasks))
        self.available_tasks.remove(a)
        print(f"agent {self.idx} chooses task {a.idx}")
        # print("1:", a.idx)
        # print("2:", [t.idx for t in self.available_tasks])
        return a


class AdaptiveAgent(Agent):
    def __init__(self, idx, tasks):
        super().__init__(idx, tasks)
        self.other_agents_rewards = {}

    def initialize_other_agents_rewards(self, agents):
        for agent in agents:
            self.other_agents_rewards[agent] = 0

    def action(self):
        print(f"available task for agent {self.idx}: {[t.idx for t in self.available_tasks]}")
        if not self.available_tasks:
            return None
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
                a = random.choice(easy_tasks)
            elif medium_tasks:
                a = random.choice(medium_tasks)
            else:
                a = random.choice(hard_tasks)
        elif self.reward <= min(other_rewards):
            if hard_tasks:
                a = random.choice(hard_tasks)
            elif medium_tasks:
                a = random.choice(medium_tasks)
            else:
                a = random.choice(easy_tasks)
        else:
            if medium_tasks:
                a = random.choice(medium_tasks)
            else:
                merged_tasks = easy_tasks + hard_tasks
                a = random.choice(merged_tasks)
        self.available_tasks.remove(a)
        print(f"agent {self.idx} chooses task {a.idx}")
        return a

    def record_other_rewards(self, other_agent_reward_dict: dict):
        """ record other agents' reward """
        # print(self.other_agents_rewards)
        for agent in other_agent_reward_dict:
            self.other_agents_rewards[agent] += other_agent_reward_dict[agent]
