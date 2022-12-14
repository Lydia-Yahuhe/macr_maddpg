import copy
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
        return True, min(rew_for_cmd(conflict_acs, scene.cmd_info))

    return False, -5.0


class ConflictEnv(gym.Env, ABC):
    def __init__(self, size=None, ratio=0.8, density=1.0, **kwargs):
        self.kwargs = kwargs

        self.train, self.test = load_and_split_data(size=size, ratio=ratio, density=density)

        self.action_space = spaces.Discrete(CmdCount)
        self.observation_space = spaces.Box(low=-1.0, high=+1.0, shape=(200,), dtype=np.float64)

        print('----------env------------')
        print('|   split ratio: {:<6.2f} |'.format(ratio))
        print('|    train size: {:<6} |'.format(len(self.train)))
        print('| validate size: {:<6} |'.format(len(self.test)))
        print('|  action shape: {}   |'.format((self.action_space.n,)))
        print('|   state shape: {} |'.format(self.observation_space.shape))
        print('-------------------------')

        self.scene = None
        self.video_out = cv2.VideoWriter('trained/scenario.avi',
                                         cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, length))

    def reset(self, change=True, test=False):
        if change:
            info = self.train.pop(0)
            self.scene = ConflictScene(info, **self.kwargs)
            if not test:
                self.train.append(info)

        return self.scene.next_point()

    def eval_is_over(self):
        return len(self.train) <= 0

    def step(self, actions, scene=None):
        if scene is None:
            scene = self.scene

        next_states = scene.do_step(actions)
        solved, reward = calc_reward(scene)
        return next_states, reward, solved, {}

    def render(self, mode='human', wait=1, counter=''):
        if self.video_out is None:
            return

        # ??????
        image = copy.deepcopy(base_img)

        # ?????????
        image = add_lines_on_base_map(self.scene.get_lines(), image, color=(106, 106, 255), display=False)

        # ????????????
        now = self.scene.now()
        conflict_acs = self.scene.conflict_acs
        info = self.scene.info
        global_info = {'>>> Information': 'No.{}_{}_{}'.format(info['id'], len(info['fpl_list']), counter)}
        image = add_texts_on_base_map(global_info, image, (750, 30), color=(128, 0, 128))

        # ????????????
        conflict_info = {'>>> Conflict': str(self.scene.result)}
        image = add_texts_on_base_map(conflict_info, image, (750, 80), color=(128, 0, 128))

        conflict_info = {'Real': ''}
        for i, c in enumerate(self.scene.conflicts):
            if i >= 4:
                conflict_info['r_n'] = '...'
                break
            conflict_info['r_' + str(i + 1)] = c.to_string()

        conflict_info['Fake'] = ''
        for i, c in enumerate(self.scene.fake_conflicts):
            if i >= 4:
                conflict_info['f_n'] = '...'
                break
            conflict_info['f_' + str(i + 1)] = c.to_string()
        image = add_texts_on_base_map(conflict_info, image, (750, 100), color=(0, 0, 0))

        # ????????????
        cmd_info = {'>>> Command & status': ''}
        ac_cmd_dict = {}
        for conflict_ac, cmd_list in self.scene.cmd_info.items():
            ret = {}
            for i, cmd in enumerate(cmd_list):
                ret.update(cmd.to_dict())
            ac_cmd_dict[conflict_ac] = ret
        image = add_texts_on_base_map(cmd_info, image, (750, 320), color=(128, 0, 128))

        points_dict = self.scene.tracks
        for t in sorted(points_dict.keys()):
            points = points_dict[t]
            frame = copy.deepcopy(image)

            # ??????????????????????????????????????????????????????
            texts = {'Time': '{}({}), ac_en: {}, speed: x{}'.format(t, now, len(points), 1000/wait)}
            frame = add_texts_on_base_map(texts, frame, (750, 50), color=(0, 0, 0))

            # ?????????
            frame = add_points_on_base_map(points, frame, conflict_ac=conflict_acs, ac_cmd_dict=ac_cmd_dict)

            # ??????????????????
            frame = cv2.resize(frame, (width, length))
            cv2.imshow(mode, frame)
            button = cv2.waitKey(wait)
            if button == 113:  # ???Q???????????????
                return
            elif button == 112:  # ???P???????????????
                wait = int(wait*0.1)
            elif button == 111:  # ???O???????????????
                wait *= 10
            else:
                self.video_out.write(frame)

    def close(self):
        if self.video_out is not None:
            self.video_out.release()
            cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()
            self.video_out = None
