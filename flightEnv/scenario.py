import math

import numpy as np

from flightEnv.agentSet import AircraftAgentSet
from flightEnv.cmd import parse_and_assign_cmd

from flightSim.utils import get_side_length, in_which_box, border_func
from flightSim.render import border_origin as border, cv2

duration = 1


class ConflictScene:
    def __init__(self, info, x=0, A=1, c_type='conc', limit=30, advance=300):
        self.info = info

        self.x = x  # 变量1：时间范围大小
        self.A = A  # 变量2：空域范围（4——1/4空域）
        self.c_type = c_type  # 冲突解脱类型（conc——同时解脱，pair——两两解脱）
        self.delta_T = limit  # 预留的计算时间
        self.advance = advance  # 冲突探测提前量

        self.side_length = get_side_length(border, A)

        self.agent_set = AircraftAgentSet(fpl_list=info['fpl_list'], candi=info['candi'])
        self.agent_set.do_step(info['clock'] - advance - duration, basic=True)
        self.ghost = AircraftAgentSet(other=self.agent_set)
        self.ghost.do_step(duration=advance)
        # print('>>>t0', info['clock'], self.agent_set.time, self.ghost.time)

        self.conflict_acs_seq, self.conflict_acs = [], []
        self.conflicts, self.fake_conflicts = [], []
        self.cmd_info = {}  # 指令信息
        self.result = False  # 解脱效果
        self.tracks = {}  # 飞行轨迹

        print('--------scenario------------', info['id'], x, A, c_type, limit, advance)

    def now(self):
        return self.agent_set.time

    def get_lines(self):  # 将武汉空域分为A块
        coords = []
        for i in range(int(math.sqrt(self.A)) + 1):
            coords.append([
                [border[0], border[2] + i * self.side_length[1]],
                [border[1], border[2] + i * self.side_length[1]]
            ])

            coords.append([
                [border[0] + i * self.side_length[0], border[2]],
                [border[0] + i * self.side_length[0], border[3]]
            ])
        return coords

    def __get_conflict_ac(self, conflicts):
        if len(conflicts) == 1:
            return [conflicts[0].id.split('-')]

        if self.c_type == 'pair':                      # 解脱方式为pair（两两解脱）
            conflict_acs, check = [], []
            for c in conflicts:
                two = []

                [a0, a1] = c.id.split('-')
                if a0 not in check:
                    two.append(a0)
                    check.append(a0)
                if a1 not in check:
                    two.append(a1)
                    check.append(a1)

                conflict_acs.append(two)
        elif self.c_type == 'conc':                    # 解脱方式为conc（同时解脱）
            conflict_acs = [[] for _ in range(self.A)]
            for c in conflicts:
                [a0, a1] = c.id.split('-')

                idx = in_which_box(c.pos0, border, self.side_length, self.A)
                conflict_acs[idx].append(a0)

                idx = in_which_box(c.pos1, border, self.side_length, self.A)
                conflict_acs[idx].append(a1)
        else:
            raise NotImplementedError

        return [list(set(lst)) for lst in conflict_acs if len(lst) > 0]

    def next_point(self):
        if len(self.conflict_acs_seq) > 0:
            self.conflict_acs = self.conflict_acs_seq.pop(0)
            assert len(self.conflict_acs) >= 0
            return self.get_states(a_set0=self.agent_set, a_set1=self.ghost)

        while True:
            self.agent_set.do_step(duration)
            self.ghost.do_step(duration)

            self.ghost.check_list = []
            conflicts = self.ghost.detect_conflict_list()
            if len(conflicts) <= 0:
                if self.agent_set.done():
                    return None
                continue

            if self.x > 0 and self.c_type == 'conc':
                ghost = AircraftAgentSet(other=self.ghost)

                while ghost.time < self.ghost.time + self.x:
                    ghost.do_step(duration)
                    conflicts += ghost.detect_conflict_list()

            self.conflicts = conflicts
            self.conflict_acs_seq = self.__get_conflict_ac(conflicts)
            self.conflict_acs = self.conflict_acs_seq.pop(0)

            return self.get_states(a_set0=self.agent_set, a_set1=self.ghost)

    def look_state(self, conflict_ac, state, scale_x=10, scale_y=30):
        width, height, *_ = np.array(state).shape

        image = np.ones(((height+2)*scale_y, (width+2)*scale_x), np.uint8) * 255
        print(conflict_ac, image.shape)
        for r, one_row in enumerate(state):
            print(one_row)
            for c, ele in enumerate(one_row):
                r_idx_min = (c+1)*scale_y
                r_idx_max = (c+2)*scale_y
                c_idx_min = (r+1) * scale_x
                c_idx_max = (r+2) * scale_x
                image[r_idx_min: r_idx_max, c_idx_min: c_idx_max] = border_func((ele + 1)/2, min_v=0.0, max_v=1.0) * 255

        cv2.imshow(conflict_ac, image)
        if cv2.waitKey(0) == 113:
            cv2.destroyAllWindows()
            return

    def get_states(self, a_set0, a_set1):
        state_0 = a_set0.get_states(self.conflict_acs)
        state_1 = a_set1.get_states(self.conflict_acs)

        # for i, conflict_ac in enumerate(self.conflict_acs):
        #     s_0 = np.array(state_0[i])
        #     s_1 = np.array(state_1[i])
        #     state = np.concatenate([s_0, s_1], axis=-1)
        #     self.look_state(conflict_ac, state)

        states = np.concatenate([state_0, state_1], axis=-1)
        return states.reshape(states.shape[0], -1)

        # return np.concatenate(
        #     [
        #         a_set0.get_states(self.conflict_acs),
        #         a_set1.get_states(self.conflict_acs)
        #     ],
        #     axis=1
        # )

    def __assign_cmd(self, now, actions, a_set):
        agents = a_set.agents
        assign_time = now + self.delta_T

        cmd_info = {}
        for i, conflict_ac in enumerate(self.conflict_acs):
            cmd_info[conflict_ac] = parse_and_assign_cmd(assign_time,
                                                         actions[i],
                                                         target=agents[conflict_ac])
        self.cmd_info = cmd_info

    def __check_cmd_effect(self, now, a_set):
        a_set_copy = AircraftAgentSet(other=a_set)
        # print('>>>t3', self.agent_set.time, a_set_copy.time, a_set.time, self.ghost.time)

        tracks, conflicts, is_solved = {}, [], True
        while a_set_copy.time < now + 2 * self.advance:
            tracks[a_set_copy.time] = a_set_copy.do_step(duration=5)

            if a_set_copy.time == now + self.advance:
                self.ghost = AircraftAgentSet(other=a_set_copy)

            if not a_set_copy.is_all_finished(self.conflict_acs):
                conflicts = a_set_copy.detect_conflict_list(search=self.conflict_acs)
                if len(conflicts) > 0:
                    is_solved = False
                    break

        # print('>>>t4', self.agent_set.time, a_set_copy.time, a_set.time, self.ghost.time)
        if is_solved:
            self.agent_set = a_set

        self.result = is_solved
        self.tracks = tracks
        self.fake_conflicts = conflicts

        return self.get_states(a_set0=self.ghost, a_set1=a_set_copy)

    def do_step(self, actions):
        a_set_copy = AircraftAgentSet(other=self.agent_set)
        # print('>>>t1', self.agent_set.time, a_set_copy.time, self.ghost.time)
        now = self.now()

        # 解析、分配动作
        self.__assign_cmd(now, actions, a_set=a_set_copy)
        # print('>>>t2', self.agent_set.time, a_set_copy.time, self.ghost.time)

        # 检查动作的解脱效果，并返回下一部状态
        return self.__check_cmd_effect(now, a_set=a_set_copy)
