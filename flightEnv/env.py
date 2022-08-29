from abc import ABC

import gym
from gym import spaces

from flightEnv.scenario import ConflictScene
from flightEnv.cmd import CmdCount, rew_for_cmd

from flightSim.load import load_and_split_data
from flightSim.render import *

base_img = cv2.imread('scripts/wuhan_base.jpg', cv2.IMREAD_COLOR)


def calc_reward(scene):
    conflict_acs = scene.conflict_acs

    if scene.result:
        rewards = rew_for_cmd(conflict_acs, scene.cmd_info)
        solved = True
    else:
        now = scene.now()
        conflicts = scene.fake_conflicts
        rewards = []
        for ac in conflict_acs:
            if len(conflicts[ac]) > 0:
                rewards.append(-1.0 * len(conflicts[ac]))
            else:
                rewards.append(-0.5)
        solved = False
    return solved, rewards


class ConflictEnv(gym.Env, ABC):
    def __init__(self, x=0, size=None, ratio=0.8):
        self.x = x

        self.train, self.test = load_and_split_data(size=size, ratio=ratio)

        self.action_space = spaces.Discrete(CmdCount)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(200,), dtype=np.float64)

        print('----------env----------')
        print('    train size: {:>6}'.format(len(self.train)))
        print(' validate size: {:>6}'.format(len(self.test)))
        print('  action shape: {}'.format((self.action_space.n,)))
        print('   state shape: {}'.format(self.observation_space.shape))
        print('-----------------------')

        self.scene = None
        self.size = len(self.train)
        self.video_out = cv2.VideoWriter('trained/scenario.avi',
                                         cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, length))

    def reset(self, mode='next_p'):
        if mode == 'next_s':
            self.scene = None

        while True:
            if self.scene is None:
                idx = np.random.randint(0, self.size)
                self.scene = ConflictScene(self.train[idx], x=self.x)

            states = self.scene.next_point()
            if states is None:
                self.scene = None
                continue

            return states

    def step(self, actions, scene=None):
        if scene is None:
            scene = self.scene

        next_states = scene.do_step(actions)
        solved, rewards = calc_reward(scene)
        return next_states, rewards, solved, {}

    def render(self, mode='human', wait=10, counter=''):
        if self.video_out is None:
            return

        image = copy.deepcopy(base_img)

        # 冲突信息
        conflict_info = {'>>> Conflict': str(self.scene.result)}
        for i, c in enumerate(self.scene.conflicts):
            conflict_info['real_' + str(i + 1)] = c.to_string()

        i = 0
        for a0, cs in self.scene.fake_conflicts.items():
            for c in cs:
                conflict_info['fake_' + str(i + 1)] = c.to_string()
                i += 1
        image = add_texts_on_base_map(conflict_info, image, (750, 80), color=(180, 238, 180))

        # 指令信息
        cmd_info = {'>>> Command': ''}
        for conflict_ac, cmd_list in self.scene.cmd_info.items():
            for i, cmd in enumerate(cmd_list):
                key = '{:>10s}_{}'.format(conflict_ac, i)
                cmd_info[key] = cmd.to_string()
        image = add_texts_on_base_map(cmd_info, image, (750, 300), color=(180, 238, 180))

        if self.scene.result:
            return

        now = self.scene.now()
        info = self.scene.info
        g_info = 'No.{}_{}_{}'.format(info['id'], len(info['fpl_list']), counter)
        points_dict = self.scene.tracks
        conflict_acs = self.scene.conflict_acs
        for t in sorted(points_dict.keys()):
            frame = copy.deepcopy(image)
            points = points_dict[t]

            # 全局信息
            global_info = {'>>> Information': g_info,
                           'Time': '{}({}), ac_en: {}, speed: x{}'.format(t, now, len(points), fps)}
            frame = add_texts_on_base_map(global_info, frame, (750, 30), color=(255, 255, 0))

            # 轨迹点
            frame, points_just_coord = add_points_on_base_map(points, frame, conflict_ac=conflict_acs)

            frame = cv2.resize(frame, (width, length))
            cv2.imshow(mode, frame)
            cv2.waitKey(wait)
            if cv2.waitKey(wait) == 113:
                self.close()
            else:
                self.video_out.write(frame)

    def close(self):
        if self.video_out is not None:
            self.video_out.release()
            cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()
            self.video_out = None
