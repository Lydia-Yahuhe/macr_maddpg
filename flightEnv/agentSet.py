from __future__ import annotations

import numpy as np
from rtree import index

from flightSim.model import Conflict
from flightSim.aircraft import AircraftAgent
from flightSim.utils import make_bbox, distance, position_in_bbox
from flightSim.visual import save_to_kml


class AircraftAgentSet:
    def __init__(self, fpl_list=None, candi=None, other=None):
        if fpl_list:
            self.candi = candi
            self.time, self.end = min(candi.keys()) - 1, max(candi.keys())
            self.agents = {fpl.id: AircraftAgent(fpl) for fpl in fpl_list}
            self.agent_id_candi, self.agent_id_en = [], []
            self.check_list = []
        else:
            self.candi = other.candi
            self.time, self.end = other.time, other.end
            self.agents = {a_id: agent.copy() for a_id, agent in other.agents.items()}
            self.agent_id_candi, self.agent_id_en = other.agent_id_candi[:], other.agent_id_en[:]
            self.check_list = other.check_list[:]

    def done(self):
        return self.time > self.end and len(self.agent_id_en) <= 1

    def __pre_do_step(self, clock):
        if clock in self.candi.keys():
            return self.agent_id_candi + self.candi[clock]
        return self.agent_id_candi

    def is_all_finished(self, acs):
        return sum([int(ac in self.agent_id_en) for ac in acs]) <= 0

    def do_step(self, duration=1, basic=False):
        now = self.time
        duration -= now * int(basic)

        tracks = []
        for i in range(duration):
            clock = now + i + 1

            agent_id_candi, agent_id_en, tracks = [], [], []
            for agent_id in self.__pre_do_step(clock):
                agent = self.agents[agent_id]

                if not agent.do_step(clock):
                    agent_id_candi.append(agent_id)

                    if agent.is_enroute():
                        agent_id_en.append(agent_id)
                        tracks.append([agent_id] + agent.get_x_data())

            self.agent_id_candi = agent_id_candi
            self.agent_id_en = agent_id_en

        self.time = now + duration
        return tracks

    def build_rt_index(self, ext=(0.0, 0.0, 0.0)):
        agents = self.agents
        idx = index.Index(properties=index.Property(dimension=3))

        for i, a_id in enumerate(self.agent_id_en):
            idx.insert(i, make_bbox(agents[a_id].position, ext=ext))

        return idx, self.agent_id_en

    def get_states(self, conflict_acs, length=25):
        agents = self.agents
        r_tree, ac_en = self.build_rt_index()

        states = []
        for conflict_ac in conflict_acs:
            a0 = agents[conflict_ac]
            pos0 = a0.position
            bbox = make_bbox(pos0, ext=(1.0, 1.0, 1500.0))

            state_dict = {}
            for i in r_tree.intersection(bbox):
                a1 = agents[ac_en[i]]
                pos1 = a1.position

                state_dict[position_in_bbox(bbox, pos1)] = [
                    2 * float(a1.id in conflict_acs) - 1.0,
                    (pos1[0] - pos0[0]) / 1.0,
                    (pos1[1] - pos0[1]) / 1.0,
                    (pos1[2] - pos0[2]) / 1500.0
                ]

            if len(state_dict) <= length:
                state = [[0.0 for _ in range(4)] for _ in range(length)]
                j = 0
                for key in sorted(state_dict.keys()):
                    state[j] = state_dict[key]
                    j += 1
            else:
                state = [state_dict[key] for key in sorted(state_dict.keys())]
                delta = len(state) - length
                if delta % 2 == 0:
                    state = state[int(delta/2):-int(delta/2)]
                else:
                    state = state[int((delta-1)/2):-int((delta+1)/2)]

            states.append(np.concatenate(state))
        return states

    def detect_conflict_list(self, search=None):
        agent_id_en = self.agent_id_en

        if len(agent_id_en) <= 1:
            return []

        if search is None:
            search = agent_id_en

        r_tree, _ = self.build_rt_index()

        conflicts, check_list = [], []
        for a0_id in search:
            a0 = self.agents[a0_id]
            if not a0.is_enroute():
                continue

            pos0 = a0.position
            bbox = make_bbox(pos0, (0.1, 0.1, 299))

            for i in r_tree.intersection(bbox):
                a1_id = agent_id_en[i]
                forward_str, backward_str = a0_id + '-' + a1_id, a1_id + '-' + a0_id

                if a0_id == a1_id or forward_str in check_list or forward_str in self.check_list:
                    continue

                a1 = self.agents[a1_id]
                pos1 = a1.position
                h_dist = distance(pos0, pos1)
                v_dist = abs(pos0[2] - pos1[2])
                if h_dist < 10000 and v_dist < 300.0:
                    self.check_list.append(forward_str)
                    self.check_list.append(backward_str)

                    conflicts.append(Conflict(id=forward_str, time=self.time, hDist=h_dist, vDist=v_dist,
                                              fpl0=a0.fpl, fpl1=a1.fpl, pos0=pos0, pos1=pos1))
                check_list.append(backward_str)

        return conflicts

    # def visual(self, save_path='agentSet', limit=None):
    #     tracks_real = {}
    #     tracks_plan = {}
    #     for a_id, agent in self.agents.items():
    #         if not agent.is_enroute():
    #             continue
    #
    #         if limit is not None and a_id not in limit:
    #             continue
    #
    #         tracks_real[a_id] = [tuple(track[:3]) for track in agent.tracks.values()]
    #         tracks_plan[a_id] = [(point.location.lng, point.location.lat, 8100.0)
    #                              for point in agent.fpl.routing.waypointList]
    #     save_to_kml(tracks_real, tracks_plan, save_path=save_path)
