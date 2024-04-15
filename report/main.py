from env import Env

def main():
    env = Env('intermediate', 2, ['random', 'adaptive', 'dqn'])
    rewards, actions = env.simulate_game()
    print("Rewards:", rewards)
    print("Actions:", actions)

if __name__ == '__main__':
    main()
