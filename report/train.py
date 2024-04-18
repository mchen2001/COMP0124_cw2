import os
import random

from torch import nn

from agent import DQNAgent, PPOAgent, DQN_MODEL_PATH, PPO_MODEL_PATH
from env import Env
import matplotlib.pyplot as plt

import torch
from torch.distributions import Categorical

import numpy as np


def collect_data(num_of_data, min_task_volume, max_task_volume, epsilon=None):
    dqn_data = []
    ppo_data = []
    while len(dqn_data) < num_of_data:
        task_volume = np.random.randint(min_task_volume, max_task_volume)
        game_type = random.choice(["basic", "intermediate", "hard"])
        env = Env(game_type, task_volume, ["dqn", "ppo"])
        dqn_agent = env.agents[0]
        ppo_agent = env.agents[1]
        assert isinstance(dqn_agent, DQNAgent)
        assert isinstance(ppo_agent, PPOAgent)
        if epsilon:
            dqn_agent.epsilon = epsilon
        done = False

        while not done:
            dqn_state = dqn_agent.get_state()
            ppo_state = ppo_agent.update_state()
            task_type_map = {"easy": 0,
                             "medium": 1,
                             "hard": 2}
            dqn_action = dqn_agent.act(verbose=False)
            ppo_action = ppo_agent.act(verbose=False)
            agent_actions = {}
            if dqn_action:
                agent_actions[dqn_agent] = dqn_action
                dqn_action = task_type_map[dqn_action.type]

            if ppo_action:
                agent_actions[ppo_agent] = ppo_action
                ppo_action = task_type_map[ppo_action.type]

            reward, done = env.step(agent_actions)

            dqn_reward = reward[dqn_agent.idx]
            ppo_reward = reward[ppo_agent.idx]

            # ppo_value = ppo_agent.value_est
            ppo_log_prob = ppo_agent.log_prob

            dqn_next_state = dqn_agent.get_state()
            ppo_next_state = ppo_agent.update_state()

            if dqn_action:
                dqn_data.append((dqn_state, dqn_action, dqn_reward, dqn_next_state, done))

            if ppo_action and ppo_log_prob:
                ppo_data.append((ppo_state, ppo_action, ppo_reward, ppo_next_state, done, ppo_log_prob))
    return dqn_data, ppo_data


def train_dqn_ppo(episodes, min_task_volume, max_task_volume, data_points):
    if os.path.exists(DQN_MODEL_PATH):
        os.remove(DQN_MODEL_PATH)
    if os.path.exists(PPO_MODEL_PATH):
        os.remove(PPO_MODEL_PATH)
    dqn_agent = DQNAgent(0, [])
    ppo_agent = PPOAgent(1, [])

    ls_dqn = []
    actor_ls = []
    critic_ls = []
    epsilon = None
    for episode in range(episodes):
        dqn_data, ppo_data = collect_data(data_points, min_task_volume, max_task_volume, epsilon)
        losses_dqn = []
        losses_actor = []
        losses_critic = []

        # Train DQN Agent
        dqn_loss, epsilon = train_step_dqn(dqn_agent, dqn_data)
        if dqn_loss is not None:
            losses_dqn.append(dqn_loss)

        # Train PPO Agent
        actor_loss, critic_loss = train_step_ppo(ppo_agent, ppo_data)
        if actor_loss is not None:
            losses_actor.append(actor_loss)
        if critic_loss is not None:
            losses_critic.append(critic_loss)

        # Log and save models after each episode
        torch.save(dqn_agent.model.state_dict(), DQN_MODEL_PATH)
        torch.save(ppo_agent.model.state_dict(), PPO_MODEL_PATH)
        print(f"Episode {episode + 1}: Training completed on full dataset.")

        # Record average losses for the episode
        if losses_dqn:
            ls_dqn.append(np.mean(losses_dqn))
        if losses_actor:
            actor_ls.append(np.mean(losses_actor))
        if losses_critic:
            critic_ls.append(np.mean(losses_critic))

        torch.save(dqn_agent.model.state_dict(), DQN_MODEL_PATH)
        torch.save(ppo_agent.model.state_dict(), PPO_MODEL_PATH)
    # Plotting training losses
    plt.plot(ls_dqn, label='DQN Loss')
    plt.plot(actor_ls, label='Actor Loss')
    plt.plot(critic_ls, label='Critic Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.savefig("./plots/training")
    return epsilon


def train_step_dqn(agent, data):
    states, actions, rewards, next_states, dones = zip(*data)
    states = torch.stack(states, dim=0)
    next_states = torch.stack(next_states, dim=0)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)

    output = agent.model(states).squeeze(1)
    actions = actions.unsqueeze(-1)
    Q_expected = output.gather(1, actions).squeeze(1)
    Q_targets_next = agent.model(next_states).detach().max(1)[0]
    Q_targets = rewards.unsqueeze(1) + (agent.gamma * Q_targets_next * (1 - dones.unsqueeze(1)))
    Q_targets_selected = Q_targets.gather(1, actions).squeeze(1)
    loss = nn.MSELoss()(Q_expected, Q_targets_selected)

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    return loss.item(), agent.epsilon


def train_step_ppo(agent, data, clip_param=0.2, gamma=0.99):
    states, actions, rewards, next_states, dones, log_probs_old = zip(*data)
    states = torch.stack(states, dim=0)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.stack(next_states, dim=0)
    dones = torch.FloatTensor(dones)
    log_probs_old = torch.stack(log_probs_old, dim=0)

    # Get current outputs from the model
    action_probs, values = agent.model(states)
    values_next = agent.model(next_states)[1].detach()

    # Value target calculations
    td_target = rewards + gamma * values_next * (1 - dones)
    advantages = (td_target - values).detach()

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Get new log probabilities
    dist = Categorical(action_probs)
    log_probs = dist.log_prob(actions)

    # Calculate the ratio
    ratios = torch.exp(log_probs - log_probs_old)

    # Clipped loss for the policy
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()

    # Critic loss
    critic_loss = (td_target - values).pow(2).mean()

    # Total loss
    total_loss = actor_loss + 0.5 * critic_loss

    # Optimize the model
    agent.optimizer.zero_grad()
    total_loss.backward()
    agent.optimizer.step()

    return actor_loss.item(), critic_loss.item()


if __name__ == '__main__':
    epsilon = train_dqn_ppo(1000, 1, 100, 100)
    print(f"best epsilon for DQN is: {epsilon}")
