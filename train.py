import argparse

from flightEnv.env import ConflictEnv

from algo.maddpg_agent import MADDPG
from algo.misc import *


def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_episodes', default=int(1e5), type=int)
    parser.add_argument('--memory_length', default=int(1e4), type=int)
    parser.add_argument('--max_steps', default=int(1e6), type=int)

    parser.add_argument('--inner_iter', help='meta-learning parameter', default=5, type=int)  # 1
    parser.add_argument('--meta-step-size', help='meta-training step size', default=1.0, type=float)
    parser.add_argument('--meta-final', help='meta-training step size by the end', default=0.1, type=float)

    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.0, type=float)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)  # 2
    parser.add_argument('--c_lr', default=0.0001, type=float)  # 3
    parser.add_argument('--batch_size', default=32, type=int)  # 4

    parser.add_argument('--x', default=0, type=int)  # 7
    parser.add_argument('--A', default=1, type=int)  # 5
    parser.add_argument('--c_type', default='conc', type=str)  # 6
    parser.add_argument('--density', default=3, type=float)  # 8

    parser.add_argument("--load_path", default=None, type=str)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument('--episode_before_train', default=100, type=int)

    return parser.parse_args()


def make_exp_id(args):
    return 'train_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.inner_iter, args.a_lr, args.c_lr, args.batch_size,
                                                  args.A, args.c_type, args.x, args.density)


def train():
    args = args_parse()
    # th.manual_seed(args.seed)
    path = get_folder(make_exp_id(args))

    env = ConflictEnv(size=1, ratio=1.0,
                      density=args.density, x=args.x, A=args.A, c_type=args.c_type)

    model = MADDPG(env.observation_space.shape[0],
                   env.action_space.n,
                   args,
                   graph_path=path['graph_path'],
                   log_path=path['log_path'],
                   load_path=args.load_path)

    # 统计：每百回合的平均奖励、每百步的解脱率、每百回合的解脱率、每回合的步数
    rew_epi, rew_step, sr_step, sr_epi, step_epi = [], [], [], [], []

    # 变量：步数、回合数、回合内求解次数、回合内奖励和、是否更换新的场景
    step, episode, t, rew, change = 0, 1, 0, 0.0, True
    while True:
        states, done = env.reset(change=change), False

        # 如果states是None，则该回合的所有冲突都被成功解脱
        if states is not None:
            actions = model.choose_action(states, noisy=True)
            next_states, reward, done, info = env.step(actions)
            # env.render(counter='{}_{}_{}'.format(t, step, episode))

            # replay buffer R
            obs = th.from_numpy(np.stack(states)).float().to(device)
            next_obs = th.from_numpy(np.stack(next_states)).float().to(device)
            rw_tensor = th.FloatTensor(np.array([reward])).to(device)
            ac_tensor = th.FloatTensor(actions).to(device)
            model.memory.push(obs.data, ac_tensor, next_obs.data, rw_tensor)
            # states = next_states

            t += 1
            step += 1
            rew += reward
            sr_step.append(float(done))
            rew_step.append(reward)
            print('[{:>2d} {:>6d} {:>6d} {:>+4.2f}]'.format(t, step, episode, reward))

            # 开始更新网络参数
            if episode >= args.episode_before_train:
                frac_done = min(step / args.max_steps, 1.0)
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
                model.save_model(path['model_path'], episode)

        # 回合数超过设定最大值，则结束训练
        if episode >= args.max_episodes:
            break

    model.close()


if __name__ == '__main__':
    train()
