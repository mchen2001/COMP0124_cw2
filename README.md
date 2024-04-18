# COMP0124_CW2

## Project Overview

This project explores the dynamics of a competitive environment where multiple agents with different strategies compete to allocate tasks of varying difficulty levels. It is designed to study how different agent strategies affect their rewards and overall performance in a competitive setting. The agent strategies implemented are Random, Adaptive, Deep Q-Network (DQN), and Proximal Policy Optimization (PPO).

## Installation

Before running the project, ensure you have Python and pip installed on your system.

1. Clone the repository:
   ```sh
   git clone [URL_OF_YOUR_GITHUB_REPO]

2. Navigate to the project directory:
   ```sh
    cd [LOCAL_CLONED_REPO_NAME]

3. Install the dependencies:
   ```sh
    pip install -r requirements.txt

## Usage

Execute the main.py script to start the simulation:

This will simulate a competitive game environment where agents interact based on their unique strategies and learn over time to improve their task allocation and reward maximization.

## Project Structure

agent.py: Defines the agents with their strategies and learning mechanisms.

env.py: Contains the environment setup where agents perform and interact.

game.py: Implements the game logic, task definitions, and their management.

main.py: The entry point of the program to run the simulations.

train.py: Script for training the DQN and PPO agents using the simulation data.

utils.py: Helper functions and utilities used across the project.

report/save_model: Directory where trained models are saved for later use or evaluation.

## Features

Strategic Agents: Study and compare the behavior of different types of agents in a shared environment.

Environment Simulation: A well-defined task environment that simulates competition among agents.

Reinforcement Learning: Implementation of DQN and PPO agents, along with their training procedures.

Performance Evaluation: Analysis of how different strategies affect the rewards and task allocation efficiency.
