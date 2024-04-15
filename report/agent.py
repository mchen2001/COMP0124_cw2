import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


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

    def act(self):
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

    def act(self):
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

    def act(self):
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


class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, idx, tasks, state_size, action_size, hidden_size=64, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, batch_size=20, memory_size=10000):
        self.idx = idx
        self.available_tasks = set(tasks)
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.reward = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self):
        if np.random.rand() <= self.epsilon:
            chosen_task = random.choice(list(self.available_tasks))
        else:
            state = [task.prob for task in self.available_tasks]  # Collecting state information
            state = torch.FloatTensor(state).unsqueeze(0)
            act_values = self.model(state)
            chosen_task = list(self.available_tasks)[torch.argmax(act_values).item()]
        
        self.available_tasks.remove(chosen_task)  # Ensure the chosen task is removed
        return chosen_task

    def update_reward(self, payoff):
        """Update the total reward for the agent."""
        self.reward += payoff
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        Q_expected = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        Q_targets_next = self.model(next_states).detach().max(1)[0]
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_tasks(self, tasks):
        self.available_tasks = set(tasks)