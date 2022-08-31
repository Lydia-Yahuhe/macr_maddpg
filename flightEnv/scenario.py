import numpy as np

from flightEnv.agentSet import AircraftAgentSet
from flightEnv.cmd import int_2_cmd

from flightSim.utils import make_bbox, position_in_bbox

duration = 1


class ConflictScene:
    def __init__(self, info, x=0, limit=30, advance=300):
        self.info = info

        self.x = x  # 变量1：时间范围大小
        self.delta_T = limit  # 预留的计算时间
        self.advance = advance  # 冲突探测提前量

        self.entity = AircraftAgentSet(fpl_list=info['fpl_list'], candi=info['candi'])
        self.entity.do_step(info['clock']-advance-duration, basic=True)

        self.agent_set = self.entity
        self.ghost = None

        self.conflict_acs = []
        self.conflicts = []
        self.cmd_info = {}
        self.result = False
        self.tracks = {}

        print('******************************', info['id'], self.now(), duration, advance, info['clock'])

    # def initialize(self):
    #     self.agent_set = AircraftAgentSet(other=self.entity)

    def now(self):
        return self.agent_set.time

    def next_point(self):
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

            ghost = AircraftAgentSet(other=self.ghost)

            while ghost.time < self.ghost.time + self.x:
                ghost.do_step(duration=duration)
                conflicts += ghost.detect_conflict_list()

            conflict_acs = []
            for c in conflicts:
                conflict_acs += c.id.split('-')
            self.conflicts = conflicts
            self.conflict_acs = list(set(conflict_acs))

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

        conflict_acs = self.conflict_acs
        now = agent_set.time

        # 解析、分配动作
        cmd_info = {}
        assign_time = now + self.delta_T
        for i, conflict_ac in enumerate(conflict_acs):
            agent = agent_set.agents[conflict_ac]

            cmd_list = []
            for cmd in int_2_cmd(assign_time, actions[i]):
                agent.assign_cmd(cmd)
                cmd_list.append(cmd)
            cmd_info[conflict_ac] = cmd_list
        self.cmd_info = cmd_info

        ghost = AircraftAgentSet(other=agent_set)

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

        if self.result:
            self.agent_set = agent_set

        return self.get_states(a_set0=self.ghost, a_set1=ghost)
