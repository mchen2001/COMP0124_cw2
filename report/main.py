from env import *

env = Env('basic', 3, ['random', 'adaptive', 'random'])
rewards, actions = env.simulate_game()
print(rewards, actions)