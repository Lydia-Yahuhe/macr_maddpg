import math

import numpy as np

from flightEnv.agentSet import AircraftAgentSet
from flightEnv.cmd import parse_and_assign_cmd

from flightSim.utils import get_side_length, in_which_box
from flightSim.render import border_origin as border

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

    def get_states(self, a_set0, a_set1):
        return np.concatenate(
            [
                a_set0.get_states(self.conflict_acs),
                a_set1.get_states(self.conflict_acs)
            ],
            axis=1
        )

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

        tracks, is_solved, conflicts = {}, True, []
        while a_set_copy.time < now + 2 * self.advance:
            tracks[a_set.time] = a_set_copy.do_step(duration=5)

            if a_set_copy.time == now + self.advance:
                self.ghost = AircraftAgentSet(other=a_set_copy)

            conflicts = a_set_copy.detect_conflict_list(search=self.conflict_acs)
            if len(conflicts) > 0:
                is_solved = False
                break

        # print('>>>t4', self.agent_set.time, a_set_copy.time, a_set.time, self.ghost.time)
        if is_solved:
            self.agent_set = a_set

        self.tracks = tracks
        self.fake_conflicts = conflicts
        self.result = is_solved

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
