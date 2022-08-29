import csv

from flightEnv import ConflictScene
from flightEnv.env import ConflictEnv

from algo.maddpg_agent import MADDPG
from algo.misc import *

from train import args_parse


def train():
    args = args_parse()

    size = 16
    env = ConflictEnv(size=size, ratio=1.0)
    model = MADDPG(env.observation_space.shape[0], env.action_space.n, args, record=False)
    model.load_model()

    # for x in [0, 20, 40, 60]:
    for x in [10]:
        solved_step, solved_epi = [], []
        for i, info in enumerate(env.train):
            print(i, end='\t')
            scene = ConflictScene(info, x=x)
            times = 0
            rew_epi = []

            while True:
                states = scene.next_point()
                # print(states)
                if states is None:
                    solved_epi.append(1.0)
                    break

                actions = model.choose_action(states, noisy=False)
                next_states, rewards, done, info = env.step(actions, scene=scene)
                times += 1

                solved_step.append(float(done))
                rew_epi.append(sum(rewards))
                print(times, done, sum(rewards))

                if not done:
                    solved_epi.append(0.0)
                    break

            with open('evaluate_{}_{}.csv'.format(x, size), 'a+', newline='') as f:
                csv.writer(f).writerow([i, solved_epi[-1], times, round(np.mean(rew_epi), 2)])

        print('x:', x)
        print('mean sr_epi:', np.mean(solved_epi))
        print('mean sr_step:', np.mean(solved_step))

        break

    model.close()


if __name__ == '__main__':
    train()
