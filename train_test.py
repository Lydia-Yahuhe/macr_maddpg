import argparse
from tqdm import tqdm

from algo.maddpg_agent import MADDPG
from algo.misc import get_folder

from flightEnv.env import ConflictEnv


def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--memory_length', default=int(1e4), type=int)
    parser.add_argument('--max_steps', default=int(1e5), type=int)

    parser.add_argument('--inner_iter', help='meta-learning parameter', default=5, type=int)  # 1
    parser.add_argument('--meta-step-size', help='meta-training step size', default=1.0, type=float)
    parser.add_argument('--meta-final', help='meta-training step size by the end', default=0.1, type=float)

    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.0, type=float)
    parser.add_argument('--a_lr', default=0.0001, type=float)  # 2
    parser.add_argument('--c_lr', default=0.0001, type=float)  # 3
    parser.add_argument('--batch_size', default=256, type=int)  # 4

    parser.add_argument('--x', default=0, type=int)  # 7
    parser.add_argument('--A', default=1, type=int)  # 5
    parser.add_argument('--c_type', default='conc', type=str)  # 6
    parser.add_argument('--density', default=1, type=float)  # 8
    parser.add_argument('--suffix', default='1', type=str)  # 8

    parser.add_argument("--render", default=True, type=bool)
    parser.add_argument("--load_path", default=None, type=str)
    parser.add_argument("--save_interval", default=10000, type=int)

    return parser.parse_args()


def make_exp_id(args):
    return 'train_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.inner_iter, args.a_lr, args.c_lr, args.batch_size,
                                                     args.A, args.c_type, args.x, args.density, args.suffix)


def train():
    args = args_parse()
    # th.manual_seed(args.seed)

    env = ConflictEnv(density=args.density, x=args.x, A=args.A, c_type=args.c_type)

    path = get_folder(make_exp_id(args), allow_exist=True)
    model = MADDPG(env.observation_space.shape[0],
                   env.action_space.n,
                   args,
                   release=True,
                   graph_path=path['graph_path'],
                   log_path=path['log_path'],
                   load_path=args.load_path)

    for step in tqdm(range(args.max_steps+1), desc='Model is training...'):
        model.update(step, args.meta_final)

        if step % args.save_interval == 0:
            model.save_model(path['model_path'], step)

    model.close()


if __name__ == '__main__':
    train()

