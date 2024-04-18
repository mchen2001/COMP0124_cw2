from env import Env


def main():
    env = Env('hard', 1, ['random', 'adaptive', 'dqn', 'ppo'])
    rewards, actions = env.simulate_game()
    print("All tasks completed!")
    print("Final Rewards:", rewards)
    # print("Actions:", actions)


if __name__ == '__main__':
    main()
