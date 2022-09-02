import csv
import time
from copy import deepcopy

import cv2

from train import *
from flightSim.utils import border_func


standard_width = 1600
standard_height = 900


class NetLooker:
    def __init__(self, net, name='net', width=16000, height=9000, channel=3, ratio=0.5, **kwargs):
        self.net = net
        self.vars = net.state_dict()
        self.name = name

        self.width = width
        self.height = height
        self.channel = channel
        self.ratio = ratio

        self.__look_weights_and_biases(**kwargs)

    def __look_weights_and_biases(self, look_weight=False, look_bias=True, scale=10):
        # 创建一个的黑色画布，RGB(0,0,0)即黑色
        image = np.ones((self.height, self.width, self.channel), np.uint8) * 255

        length = len(self.vars)

        self.rows = length // 2 + 1
        c_start = int(self.height * 0.05)  # 最左侧节点圆心与左侧边界的距离
        r_interval = (self.height - 2 * c_start) // (self.rows - 1)  # 相邻层节点圆心的距离
        c_interval = int(self.width // max([var.shape[-1] for var in self.vars.values()]))  # 同层相邻节点圆心的距离
        self.radius = int(c_interval * 0.4)

        var_list, link_metrix, count = [], [], 0
        for i, (var_name, var) in enumerate(self.vars.items()):
            var_list.append(var)
            if i % 2 != 0 and i != length - 1:
                continue

            if count == 0:
                mode = 'input'
            elif count == self.rows - 1:
                mode = 'output'
            else:
                mode = 'hidden'

            dim_nodes = var.shape[-1]

            # 最上层节点圆心与上层边界的距离
            r_start = int(self.width // 2 - (dim_nodes - 1) / 2 * c_interval)

            coord_list = []
            for j in range(dim_nodes):
                coord = (r_start + j * c_interval, c_start + count * r_interval)
                coord_list.append(coord)
            link_metrix.append(coord_list)

            string = "{}({})".format(mode, dim_nodes)
            image = cv2.putText(image, string,
                                (self.width // 2, c_start + count * r_interval + 10 * self.radius),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                self.width/standard_width*self.ratio, (255.0, 0, 0),
                                thickness=self.radius // 2)
            count += 1

        for i in range(len(var_list) // 2):
            weights, biases = var_list[i * 2], var_list[i * 2 + 1]
            print(weights.shape, len(link_metrix[i]), len(link_metrix[i+1]))

            for j, coord1 in enumerate(link_metrix[i]):
                for k, coord2 in enumerate(link_metrix[i + 1]):
                    # bias的颜色（越大越长，负下正上）
                    if j == len(link_metrix[i+1]) - 1 and look_bias:
                        c_bias = 255 if abs(biases[k]*scale) >= 1.0 else 200
                        image = cv2.line(image,
                                         coord2,
                                         (coord2[0], coord2[1]-border_func(biases[k]*scale, d_type=int)*5*self.radius),
                                         (0, 0, c_bias),
                                         thickness=self.radius // 2)

                    # weight连线的颜色（带符号的权重越大，越接近黑色）
                    if look_weight:
                        c_weight = (1.0 - border_func(abs(weights[k, j]), d_type=float)) * 255.0
                        image = cv2.line(image,
                                         (coord1[0], coord1[1] + self.radius), (coord2[0], coord2[1] - self.radius),
                                         (c_weight, c_weight, c_weight))

        self.link_matrix = link_metrix
        self.image = image

    def __look_layer(self, image, values, matrix):
        assert len(matrix) == len(values)

        for i, v in enumerate(values):
            thick = self.radius // 2 if v < 0 else -1
            c_node = (1 - border_func(abs(v), d_type=float)) * 255.0
            image = cv2.circle(image, matrix[i], self.radius, (c_node, c_node, c_node), thickness=thick)
        return image

    def look(self, inputs, folder=None, suffix='actor'):
        ratio = self.ratio
        net = self.net

        image = deepcopy(self.image)

        result = th.from_numpy(np.stack(inputs)).float().to(device)
        for i in range(self.rows):
            if i == 0:
                result = result
            elif i == 1:
                result = F.relu(net.FC1(result))
            elif i == 2:
                result = F.relu(net.FC2(result))
            elif i == 3:
                result = F.relu(net.FC3(result))
            elif i == 4:
                result = F.relu(net.FC4(result))
            else:
                result = th.tanh(net.FC5(result))

            image = self.__look_layer(image, result[0, :], self.link_matrix[i])

        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)

        cv2.imshow(self.name, image)
        cv2.waitKey(1000)

        if folder is not None:
            cv2.imwrite(folder+'{}_{}.png'.format(self.name, suffix), image)

    def close(self):
        cv2.waitKey(1) & 0xFF
        cv2.destroyAllWindows()


def train():
    args = args_parse()

    env = ConflictEnv(x=args.x, A=args.A, c_type=args.c_type)

    graph_path = os.path.join(args.load_path, 'graph/')
    suffix = 400

    model = MADDPG(env.observation_space.shape[0],
                   env.action_space.n,
                   args,
                   load_path=[os.path.join(args.load_path, 'model/'), suffix])

    a_looker = NetLooker(net=model.actor, name='actor', look_weight=True)
    # c_looker = NetLooker(net=model.critic, name='critic')

    # 每百步的解脱率、每百回合的解脱率、每回合的步数
    rew_epi, rew_step, sr_step, sr_epi, step_epi = [], [], [], [], []

    # 回合数、回合内步数、回合内奖励和、是否更换新的场景
    episode, t, rew, change = 1, 0, 0.0, True
    while not env.eval_is_over():
        states, done = env.reset_for_eval(change=change), False

        # 如果states是None，则该回合的所有冲突都被成功解脱
        if states is not None:
            a_looker.look(states, folder=graph_path, suffix='{}_{}_{}'.format(suffix, episode, t))
            actions = model.choose_action(states, noisy=False)
            next_states, reward, done, info = env.step(actions)
            # env.render(counter='{}_{}_{}'.format(t, step, episode))
            # states = next_states

            t += 1
            rew += reward
            sr_step.append(float(done))
            rew_step.append(reward)
            print('{:>2d} {:>6d} {:>+4.2f}'.format(t, episode, reward))

        # 如果前个冲突成功解脱，则进入下一个冲突时刻，否则更换新的场景
        if not done:
            change = True
            sr_epi.append(int(states is None))
            step_epi.append(t)
            rew_epi.append(rew)
            print(episode, states is None, rew, t)
            t, rew = 0, 0.0
            episode += 1
        else:
            change = False

    a_looker.close()

    print('----------------------------')
    print('   sr_step:', np.mean(sr_step))
    print('sr_episode:', np.mean(sr_epi))
    print('  rew_step:', np.mean(rew_step))
    print('   rew_epi:', np.mean(rew_epi))
    print('  step_epi:', np.mean(step_epi))
    print('----------------------------')


"""
Episode: <=3000
----------------------------
   sr_step: 0.5897435897435898
sr_episode: 0.0
  rew_step: -0.04852887233002828
   rew_epi: -0.11828912630444392
  step_epi: 2.4375
----------------------------
Episode: 4000
----------------------------
   sr_step: 0.6341463414634146
sr_episode: 0.0
  rew_step: -0.02853248833065353
   rew_epi: -0.13031086259831984
  step_epi: 2.6666666666666665
----------------------------
"""
if __name__ == '__main__':
    train()
