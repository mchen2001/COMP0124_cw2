import random

from agent import DQNAgent, PPOAgent, DQN_MODEL_PATH, PPO_MODEL_PATH
from env import Env
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
"""
TODOs:

1. figure out why some loss values are None
2. training loss doesn't seem to be reduced during training
"""
# TODO:

def train_dqn(episodes, min_task_volume, max_task_volume):
    ls_1 = []
    ls_2 = []
    for episode in range(episodes):
        losses_1 = []
        losses_2 = []
        task_volume = np.random.randint(min_task_volume, max_task_volume)
        game_type = random.choice(["basic", "intermediate", "hard"])
        env = Env(game_type, task_volume, ["dqn", "dqn"])
        dqn_agent_1 = env.agents[0]
        dqn_agent_2 = env.agents[1]
        assert isinstance(dqn_agent_1, DQNAgent)
        assert isinstance(dqn_agent_2, DQNAgent)

        done = False

        while not done:
            # Agents choose tasks and get rewards from their actions
            dqn_state_1 = dqn_agent_1.get_state()
            dqn_state_2 = dqn_agent_2.get_state()
            task_type_map = {"easy": 0,
                             "medium": 1,
                             "hard": 2}
            dqn_action_1 = dqn_agent_1.act()
            dqn_action_2 = dqn_agent_2.act()
            agent_actions = {}
            if dqn_action_1:
                agent_actions[dqn_agent_1] = dqn_action_1
                dqn_action_1 = task_type_map[dqn_action_1.type]
            # else:
            #     dqn_action_1 = task_type_map[dqn_action_1]

            if dqn_action_2:
                agent_actions[dqn_agent_2] = dqn_action_2
                dqn_action_2 = task_type_map[dqn_action_2.type]
            # else:
            #     dqn_action_2 = task_type_map[dqn_action_2]

            reward, done = env.step(agent_actions)

            dqn_reward_1 = reward[dqn_agent_1.idx]
            dqn_reward_2 = reward[dqn_agent_2.idx]

            dqn_next_state_1 = dqn_agent_1.get_state()
            dqn_next_state_2 = dqn_agent_2.get_state()

            if dqn_action_1:
                dqn_agent_1.remember(state=dqn_state_1,
                                     action=dqn_action_1,
                                     reward=dqn_reward_1,
                                     next_state=dqn_next_state_1,
                                     done=done)

            if dqn_action_2:
                dqn_agent_2.remember(state=dqn_state_2,
                                     action=dqn_action_2,
                                     reward=dqn_reward_2,
                                     next_state=dqn_next_state_2,
                                     done=done)

            # Replay experiences to update DQN model
            # if len(dqn_agent.memory) >= dqn_agent.batch_size:
            loss_1 = dqn_agent_1.replay()
            # TODO: some losses are None
            if loss_1:
                losses_1.append(loss_1)

            loss_2 = dqn_agent_2.replay()
            if loss_2:
                losses_2.append(loss_2)

            # PPO update mechanism needs to be implemented here if applicable
            # Possibly ppo_agent.update_policy() or similar

            # Cumulative rewards (not shown in this snippet) can be tracked similarly to previous examples

        torch.save(dqn_agent_1.model, DQN_MODEL_PATH)
        print(f"Episode {episode}: Completed with volumes set to {task_volume}.")
        ls_1.append(np.mean(losses_1))
        # ls_2.extend(losses_2)
    # print(ls_1)
    # torch.save(dqn_agent_1.model, )
    plt.plot(ls_1)
    plt.show()

if __name__ == '__main__':
    train_dqn(1000,1,20)