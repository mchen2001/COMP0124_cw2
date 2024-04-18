import random
import numpy as np


class Task:
    def __init__(self, task_type, idx, gain=0.1):
        self.type = task_type
        self.idx = idx
        if task_type == "easy":
            self.low = 0.8
            self.high = 1
            self.payoff = 1
        elif task_type == "medium":
            self.low = 0.5
            self.high = 0.7
            self.payoff = 3
        else:
            self.low = 0.1
            self.high = 0.3
            self.payoff = 6

        self.gain = gain  # prob gain if any agent fail in this task
        self.prob = random.uniform(self.low, self.high)  # randomly select a prob
        self.done = False
        

    def __eq__(self, other):
        if not isinstance(other, Task):
            return NotImplemented

        return self.idx == other.idx

    def __hash__(self):
        return hash(self.idx)

    def do_task(self):
        success = np.random.choice([True, False], p=[self.prob, 1 - self.prob])
        if success:
            self.done = True
            return self.payoff
        self.prob += self.gain
        self.prob = min(self.prob, 1)
        if self.prob >= 0.8:
            self.type = "easy"
        elif self.prob >= 0.5:
            self.type = "medium"
        return 0

    def get_task_type(self):
        return self.type

    def is_completed(self):
        return self.done


class Game:
    def __init__(self, game_type, volume_level, gain=0.1):
        self.type = game_type
        self.gain = gain
        self.volume_level = volume_level
        
        if game_type == "basic":
            self.config = {"easy": 4, "medium": 1, "hard": 1}
        elif game_type == "intermediate":
            self.config = {"easy": 2, "medium": 2, "hard": 2}
        else:
            self.config = {"easy": 1, "medium": 1, "hard": 4}

        self.tasks = set()
        self.tasks_idx_map = {}
        idx = 0
        for task_type, count in self.config.items():
            for _ in range(count * self.volume_level):
                task = Task(task_type, idx, self.gain)
                self.tasks_idx_map[idx] = task
                self.tasks.add(task)
                idx += 1
        # random.shuffle(self.tasks)

    def get_task_by_idx(self, idx):
        return self.tasks_idx_map[idx]

    def get_game_tasks(self):
        return self.tasks

    def update_tasks(self):
        completed_tasks = [task.idx for task in self.tasks if task.is_completed()]
        print(f"Completed tasks before update: {completed_tasks}")
        self.tasks = [task for task in self.tasks if not task.is_completed()]
        remaining_tasks = [task.idx for task in self.tasks]
        print(f"______________________________")
