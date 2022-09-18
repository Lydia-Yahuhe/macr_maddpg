import argparse

import numpy as np

from flightEnv.env import ConflictEnv


def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_episodes', default=int(1e3), type=int)

    parser.add_argument('--x', default=30, type=int)  # 7
    parser.add_argument('--A', default=1, type=int)  # 5
    parser.add_argument('--c_type', default='conc', type=str)  # 6
    parser.add_argument('--density', default=1, type=float)  # 8

    parser.add_argument('--render', default=False, type=bool)

    return parser.parse_args()


def choose_action(states, dim_act):
    return np.random.uniform(-1.0, 1.0, (states.shape[0], dim_act))


def push_to_memory(memory, *args):
    (states, actions, next_states, reward) = args

    n_agents = states.shape[0]
    if n_agents in memory.keys():
        memory[n_agents]['states'].append(states)
        memory[n_agents]['actions'].append(actions)
        memory[n_agents]['next_states'].append(next_states)
        memory[n_agents]['reward'].append(reward)
    else:
        memory[n_agents] = {'states': [states],
                            'actions': [actions],
                            'next_states': [next_states],
                            'reward': [reward]}


def save_npz(memory, episode):
    for n_agent, experience in memory.items():
        print(n_agent)
        np.savez('experience_{}_{}.npz'.format(episode, n_agent), **experience)
    memory.clear()


def train():
    args = args_parse()

    env = ConflictEnv(density=args.density, x=args.x, A=args.A, c_type=args.c_type)
    dim_act = env.action_space.n

    # 统计：每百回合的平均奖励、每百步的解脱率、每百回合的解脱率、每回合的步数
    rew_step, sr_step, sr_epi, step_epi = [], [], [], []

    # 变量：步数、回合数、回合内求解次数、回合内奖励和、是否更换新的场景
    episode, t, change = 1, 0, True

    memory = {}

    while True:
        states, done = env.reset(change=change), False

        # 如果states是None，则该回合的所有冲突都被成功解脱
        if states is not None:
            count = 0
            while True:
                actions = choose_action(states, dim_act)
                next_states, reward, done, info = env.step(actions)

                if args.render:
                    env.render(wait=1000, counter=str(t))

                push_to_memory(memory, states, actions, next_states, np.array([reward]))

                count += 1
                sr_step.append(float(done))
                rew_step.append(reward)
                print('[{:>2d} {:>2d} {:>5d} {:>+4.2f}]'.format(count, t, episode, reward))

                if done or count >= 10:
                    t += 1
                    break

        # 如果前个冲突成功解脱，则进入下一个冲突时刻，否则更换新的场景
        if not done:
            change = True
            episode += 1
            sr_epi.append(int(states is None))
            step_epi.append(t)
            t, rew = 0, 0.0
        else:
            change = False

        if change and episode % 100 == 0:
            save_npz(memory, episode)

        if episode >= args.max_episodes:
            break


if __name__ == '__main__':
    train()
