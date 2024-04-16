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

    def get_available_task(self):
        return self.available_tasks

    def update_reward(self, payoff):
        self.reward += payoff

    def act(self):
        pass

    def get_tasks_by_types(self):
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
        return easy_tasks, medium_tasks, hard_tasks

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
        chosen_task = random.choice(list(self.available_tasks))
        print(f"agent {self.idx} chooses task {chosen_task.idx}")
        return chosen_task


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

        easy_tasks, medium_tasks, hard_tasks = self.get_tasks_by_types()

        # Task choice logic remains the same
        chosen_task = None  # Default to None
        if self.reward >= max(other_rewards):
            if easy_tasks:
                chosen_task = random.choice(easy_tasks)
            elif medium_tasks:
                chosen_task = random.choice(medium_tasks)
            else:
                chosen_task = random.choice(hard_tasks)
        elif self.reward <= min(other_rewards):
            if hard_tasks:
                chosen_task = random.choice(hard_tasks)
            elif medium_tasks:
                chosen_task = random.choice(medium_tasks)
            else:
                chosen_task = random.choice(easy_tasks)
        else:
            if medium_tasks:
                chosen_task = random.choice(medium_tasks)
            else:
                merged_tasks = easy_tasks + hard_tasks
                chosen_task = random.choice(merged_tasks)
        # Do not remove the task here
        print(f"agent {self.idx} chooses task {chosen_task.idx}")
        return chosen_task

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
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2), output_size)
        )

    def forward(self, x):
        return self.net(x).flatten()


class DQNAgent(Agent):
    def __init__(self, idx, tasks, state_size, action_size, hidden_size=64, gamma=0.99, epsilon=0.95, epsilon_min=0.01,
                 epsilon_decay=0.995, learning_rate=0.001, batch_size=20, memory_size=10000):
        super().__init__(idx, tasks)
        # self.idx = idx
        # self.available_tasks = set(tasks)
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.model = DQN(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.reward = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self):
        print(f"available task for agent {self.idx}: {[t.idx for t in self.available_tasks]}")
        if np.random.rand() >= self.epsilon:
            chosen_task = random.choice(list(self.available_tasks))
        else:
            state_map = {"easy": 1,
                         "medium": 2,
                         "hard": 3}
            state = torch.zeros(self.state_size)
            for task in self.available_tasks:
                state_map[task.idx] = state_map[task.type]
            # state = [task.prob for task in self.available_tasks]  # Collecting state information
            state = torch.FloatTensor(state).unsqueeze(0)
            chosen_type_map = {0: "easy",
                               1: "medium",
                               2: "hard"}
            act_values = self.model(state)
            chosen_order = torch.argsort(act_values).flatten().tolist()

            easy_tasks, medium_tasks, hard_tasks = self.get_tasks_by_types()
            tasks_map = {"easy": easy_tasks,
                         "medium": medium_tasks,
                         "hard": hard_tasks}

            chosen_task = None
            for chosen_type in chosen_order:
                candidates = tasks_map[chosen_type_map[chosen_type]]
                if candidates:
                    chosen_task = random.choice(candidates)
                    break

        # self.available_tasks.remove(chosen_task)  # Ensure the chosen task is removed
        print(f"agent {self.idx} chooses task {chosen_task.idx}")
        return chosen_task

    # def update_reward(self, payoff):
    #     """Update the total reward for the agent."""
    #     self.reward += payoff

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

    # def update_tasks(self, tasks):
    #     self.available_tasks = set(tasks)


class PPONetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PPONetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_policy = nn.Linear(hidden_size, output_size)
        self.fc_value = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        policy = self.softmax(self.fc_policy(x))
        value = self.fc_value(x)
        return policy, value


class PPOAgent(Agent):
    def __init__(self, idx, tasks, input_size=3, output_size=3, hidden_size=32, learning_rate=1e-3):
        super(PPOAgent, self).__init__(idx, tasks)
        self.model = PPONetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.state = self.initialize_state()

    def initialize_state(self):
        """ Initialize the state based on the tasks available.
        """
        task_counts = {'easy': 0, 'medium': 0, 'hard': 0}
        for task in self.available_tasks:
            task_type = task.get_task_type()
            if task_type in task_counts:
                task_counts[task_type] += 1

        state_vector = [task_counts['easy'], task_counts['medium'], task_counts['hard']]
        return torch.tensor(state_vector, dtype=torch.float32)

    def act(self):
        """ Act based on the current state, choosing a task type and then a specific task. """
        policy, _ = self.model(self.state)
        task_type_idx = torch.multinomial(policy, 1).item()  # Select task type based on policy
        task_types = ['easy', 'medium', 'hard']
        selected_task_type = task_types[task_type_idx]

        available_tasks_of_type = [task for task in self.available_tasks
                                   if task.get_task_type() == selected_task_type]
        if available_tasks_of_type:
            chosen_task = random.choice(available_tasks_of_type)
            return chosen_task
        else:
            return None
