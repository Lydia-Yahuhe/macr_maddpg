import argparse

from flightEnv.env import ConflictEnv

from algo.maddpg_agent import MADDPG
from algo.misc import *


def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_episodes', default=int(1e5), type=int)
    parser.add_argument('--memory_length', default=int(5e4), type=int)
    parser.add_argument('--max_steps', help='meta-training iterations', default=int(1e6), type=int)
    parser.add_argument('--inner_iter', help='samples', default=5, type=int)  # 1
    parser.add_argument('--max_step_per_epi', default=1, type=int)  # 2
    parser.add_argument('--meta-step-size', help='meta-training step size', default=1.0, type=float)
    parser.add_argument('--meta-final', help='meta-training step size by the end', default=0.01, type=float)  # 3

    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.0, type=float)  # 4
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)  # 5
    parser.add_argument('--c_lr', default=0.0001, type=float)  # 6
    parser.add_argument('--batch_size', default=16, type=int)  # 7

    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument('--step_before_train', default=100, type=int)

    return parser.parse_args()


def train():
    args = args_parse()
    # th.manual_seed(args.seed)

    env = ConflictEnv(x=10, size=16, ratio=0.75)
    model = MADDPG(env.observation_space.shape[0], env.action_space.n, args)
    # model.load_model()

    # 每百回合的平均奖励、每百步的解脱率、每百回合的解脱率、每回合的步数
    rew_epi, rew_step, sr_step, sr_epi, step_epi = [], [], [], [], []

    episode, t, rew, change = 1, 0, 0.0, True
    for step in range(1, args.max_steps):
        states, done = env.reset(change=change), False

        # 如果states是None，则该回合的所有冲突都被成功解脱
        if states is not None:
            actions = model.choose_action(states, noisy=True)
            next_states, rewards, done, info = env.step(actions)
            # env.render(counter='{}_{}_{}'.format(t, step, episode))

            # replay buffer R
            obs = th.from_numpy(np.stack(states)).float().to(device)
            next_obs = th.from_numpy(np.stack(next_states)).float().to(device)
            rw_tensor = th.FloatTensor(np.array([sum(rewards)])).to(device)
            ac_tensor = th.FloatTensor(actions).to(device)
            model.memory.push(obs.data, ac_tensor, next_obs.data, rw_tensor)
            # states = next_states

            t += 1
            rew += min(rewards)
            sr_step.append(float(done))
            rew_step.append(min(rewards))
            print('{:>2d}, {:>6d}, {:>6d}'.format(t, step, episode), end='\t')
            print(['{:>+4.2f}'.format(rew) for rew in rewards])

            # 开始更新网络参数
            if step >= args.step_before_train:
                frac_done = step / (args.max_steps * 0.3)
                step_size = frac_done * args.meta_final + (1 - frac_done) * args.meta_step_size
                model.update(step, step_size)

        # 如果前个冲突成功解脱，则进入下一个冲突时刻，否则更换新的场景
        if not done:
            change = True
            episode += 1
            sr_epi.append(int(states is None))
            rew_epi.append(rew)
            step_epi.append(t)
            t, rew = 0, 0.0
        else:
            change = False

        if change and episode % 100 == 0:
            model.scalars("REW", {'t': np.mean(rew_step), 'e': np.mean(rew_epi)}, episode)
            model.scalars("SR", {'t': np.mean(sr_step), 'e': np.mean(sr_epi)}, episode)
            model.scalars("PAR", {'times': np.mean(step_epi), 'var': model.var}, episode)
            model.scalars('MEM', model.memory.counter(), episode)

            rew_epi, rew_step, sr_step, sr_epi, step_epi = [], [], [], [], []
            if episode % args.save_interval == 0:
                model.save_model()

        # 回合数超过设定最大值，则结束训练
        if episode >= args.max_episodes:
            break

    model.close()


if __name__ == '__main__':
    train()
