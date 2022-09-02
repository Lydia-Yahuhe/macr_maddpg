import math

import numpy as np

from flightEnv.agentSet import AircraftAgentSet
from flightEnv.cmd import int_2_cmd

from flightSim.utils import make_bbox, position_in_bbox, get_side_length, border_func
from flightSim.render import border_origin as border

duration = 1


class ConflictScene:
    def __init__(self, info, x=0, A=1, c_type='conc', limit=30, advance=300):
        self.info = info

        self.x = x  # 变量1：时间范围大小
        self.A = A  # 变量2：空域范围（4——1/4空域）
        self.side_length = get_side_length(border, A)
        self.c_type = c_type  # 冲突解脱类型（conc——同时解脱，pair——两两解脱）
        self.delta_T = limit  # 预留的计算时间
        self.advance = advance  # 冲突探测提前量

        self.entity = AircraftAgentSet(fpl_list=info['fpl_list'], candi=info['candi'])
        self.entity.do_step(info['clock'] - advance - duration, basic=True)

        self.agent_set = self.entity
        self.ghost = None

        self.conflict_acs_total = []
        self.conflict_acs = []
        self.conflicts = []
        self.cmd_info = {}
        self.result = False
        self.tracks = {}

        print('--------scenario------------', info['id'], x, A, c_type, limit, advance,
              info['clock'], self.entity.time)

    # def initialize(self):
    #     self.agent_set = AircraftAgentSet(other=self.entity)

    def now(self):
        return self.agent_set.time

    def get_lines(self):
        coords = []
        for i in range(int(math.sqrt(self.A)) + 1):
            coords.append([[border[0], border[2] + i * self.side_length[1]],
                           [border[1], border[2] + i * self.side_length[1]]])
            coords.append([[border[0] + i * self.side_length[0], border[2]],
                           [border[0] + i * self.side_length[0], border[3]]])
        return coords

    def __get_conflict_ac(self, conflicts):
        if len(conflicts) == 1:
            return [conflicts[0].id.split('-')]

        check = []

        if self.c_type == 'pair':
            conflict_acs = []
            for c in conflicts:
                two = []
                [a0, a1] = c.id.split('-')
                if a0 not in check:
                    two.append(a0)
                    check.append(a0)
                if a1 not in check:
                    two.append(a1)
                    check.append(a1)

                if len(two) > 0:
                    conflict_acs.append(two)
        else:
            conflict_acs = [[] for _ in range(self.A)]
            num = int(math.sqrt(self.A))
            for c in conflicts:
                [a0, a1] = c.id.split('-')
                if a0 not in check:
                    check.append(a0)
                    r = border_func(int((c.pos0[0] - border[0]) / self.side_length[0]),
                                    min_v=0, max_v=num - 1, d_type=int)
                    c = border_func(int((c.pos0[1] - border[2]) / self.side_length[1]),
                                    min_v=0, max_v=num - 1, d_type=int)
                    idx = int(r * math.sqrt(self.A) + c)
                    # print(a0, r, c, idx)
                    conflict_acs[idx].append(a0)
                if a1 not in check:
                    check.append(a1)
                    r = border_func(int((c.pos1[0] - border[0]) / self.side_length[0]),
                                    min_v=0, max_v=num - 1, d_type=int)
                    c = border_func(int((c.pos1[1] - border[2]) / self.side_length[1]),
                                    min_v=0, max_v=num - 1, d_type=int)
                    idx = int(r * math.sqrt(self.A) + c)
                    # print(a1, r, c, idx)
                    conflict_acs[idx].append(a1)
            conflict_acs = [lst for lst in conflict_acs if len(lst) > 0]

        return conflict_acs

    def next_point(self):
        if len(self.conflict_acs_total) > 0:
            self.conflict_acs = self.conflict_acs_total.pop(0)
            # print(self.conflict_acs, self.conflict_acs_total)
            assert len(self.conflict_acs) >= 0
            return self.get_states(a_set0=self.agent_set, a_set1=self.ghost)

        while True:
            self.agent_set.do_step(duration)
            # print('>>> t1', self.now())

            if self.ghost is None:
                self.ghost = AircraftAgentSet(other=self.agent_set)
                self.ghost.do_step(duration=self.advance)
            else:
                self.ghost.do_step(duration=duration)
            # print('>>> t2', self.ghost.time)

            self.ghost.check_list = []
            conflicts = self.ghost.detect_conflict_list()
            if len(conflicts) <= 0:
                if self.agent_set.done():
                    return None
                continue

            # print(len(conflicts), end='\t')
            if self.x > 0 and self.c_type == 'conc':
                ghost = AircraftAgentSet(other=self.ghost)

                while ghost.time < self.ghost.time + self.x:
                    ghost.do_step(duration=duration)
                    conflicts += ghost.detect_conflict_list()

            self.conflicts = conflicts
            self.conflict_acs_total = self.__get_conflict_ac(conflicts)
            # print(len(conflicts), self.conflict_acs_total)

            self.conflict_acs = self.conflict_acs_total.pop(0)
            # print(self.conflict_acs, self.conflict_acs_total)

            return self.get_states(a_set0=self.agent_set, a_set1=self.ghost)

    def get_states(self, a_set0, a_set1):
        return np.concatenate((self.__get_states(a_set=a_set0), self.__get_states(a_set=a_set1)), axis=1)

    def __get_states(self, a_set, length=25):
        agents = a_set.agents
        r_tree, ac_en = a_set.build_rt_index()

        states = []
        for conflict_ac in self.conflict_acs:
            a0 = agents[conflict_ac]
            bbox = make_bbox(a0.position, ext=(1.0, 1.0, 1500.0))
            status0 = a0.get_x_data()

            state_dict = {}
            for i in r_tree.intersection(bbox):
                a1 = agents[ac_en[i]]
                status = a1.get_x_data()
                ele = [2 * float(a1.id in self.conflict_acs) - 1.0,
                       status[0] - status0[0],
                       status[1] - status0[1],
                       (status[2] - status0[2]) / 1500.0]
                state_dict[position_in_bbox(bbox, status)] = ele

            state = [[0.0 for _ in range(4)] for _ in range(length)]
            j = 0
            for key in sorted(state_dict):
                state[min(length - 1, j)] = state_dict[key]
                j += 1
            states.append(np.concatenate(state))

        return states

    def do_step(self, actions):
        agent_set = AircraftAgentSet(other=self.agent_set)
        # print('>>> t3', agent_set.time)

        conflict_acs = self.conflict_acs
        now = agent_set.time

        # 解析、分配动作
        cmd_info = {}
        assign_time = now + self.delta_T
        for i, conflict_ac in enumerate(conflict_acs):
            agent = agent_set.agents[conflict_ac]
            # print(conflict_ac)

            cmd_list = []
            for cmd in int_2_cmd(assign_time, actions[i]):
                # print('\t\t', cmd)
                agent.assign_cmd(cmd)
                cmd_list.append(cmd)
            cmd_info[conflict_ac] = cmd_list
        self.cmd_info = cmd_info

        ghost = AircraftAgentSet(other=agent_set)
        # print('>>> t4', ghost.time)

        # 检查动作的解脱效果
        self.tracks, self.result = {}, True
        while ghost.time < now + 2 * self.advance:
            self.tracks[ghost.time] = ghost.do_step(duration=5)
            if ghost.time == now + self.advance:
                self.ghost = AircraftAgentSet(other=ghost)

            conflicts = ghost.detect_conflict_list(search=conflict_acs)
            if len(conflicts) > 0:
                self.result = False
                break
        # print('>>> t5', ghost.time)

        if self.result:
            self.agent_set = agent_set

        return self.get_states(a_set0=self.ghost, a_set1=ghost)
