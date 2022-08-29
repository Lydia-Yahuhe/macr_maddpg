from typing import Dict

from .aircrafttype import *

from flightSim.utils import distance_point2d, bearing_point2d, destination, move_point2d


@dataclass
class Point2D(object):
    lng: float = 0.0
    lat: float = 0.0

    def reset(self, other):
        self.lng = other.lng
        self.lat = other.lat

    def set(self, other):
        self.lng = other.lng
        self.lat = other.lat

    def toArray(self):
        return [self.lng, self.lat]

    def toTuple(self):
        return self.lng, self.lat

    def clear(self):
        self.lng = 0
        self.lat = 0

    def distance_to(self, other):
        return distance_point2d(self, other)

    def bearing(self, other):
        return bearing_point2d(self, other)

    def destination(self, course: float, dist: float):
        coords = destination(self.toTuple(), course, dist)
        return Point2D(lng=coords[0], lat=coords[1])

    def move(self, course: float, dist: float):
        move_point2d(self, course, dist)

    def copy(self):
        return Point2D(lng=self.lng, lat=self.lat)

    def __str__(self):
        return '<%.5f,%.5f>' % (self.lng, self.lat)

    def __repr(self):
        return '<%.5f,%.5f>' % (self.lng, self.lat)


@dataclass
class Waypoint:
    id: str
    location: Point2D

    def __str__(self):
        return '[%s, %s]' % (self.id, self.location)

    def __repr__(self):
        return '[%s, %s]' % (self.id, self.location)

    def distance_to(self, other):
        return self.location.distance_to(other.location)

    def bearing(self, other):
        return self.location.bearing(other.location)

    def copy(self, name='Dogleg'):
        loc = self.location
        return Waypoint(id=name, location=Point2D(loc.lng, loc.lat))


@dataclass
class Aircraft:
    id: str
    aircraftType: AircraftType
    airline: str = None


@dataclass
class Routing:
    id: str
    waypointList: List[Waypoint]
    start: int = 0
    other: List[int] = None

    def get_points(self, section=None):
        wpt_list = self.waypointList
        if section is not None:
            points = [wpt_list[idx] for idx in section]
        else:
            points = wpt_list
        return points

    def get_coordinates(self, section=None):
        wpt_list = self.waypointList
        if section is not None:
            coordinates = [wpt_list[idx].location.toArray() for idx in section]
        else:
            coordinates = [wpt.location.toArray() for wpt in wpt_list]
        return coordinates


@dataclass
class FlightPlan:
    id: str
    routing: Routing
    startTime: int
    aircraft: Aircraft
    min_alt: float
    max_alt: float

    def to_dict(self):
        return dict(id=self.id, routing=self.routing.id, startTime=self.startTime,
                    aircraft=self.aircraft.id, acType=self.aircraft.aircraftType.id,
                    min_alt=self.min_alt, max_alt=self.max_alt, other=self.routing.other)


@dataclass
class DataSet:
    waypoints: Dict[str, Waypoint]
    routings: Dict[str, Routing]
    flightPlans: Dict[str, FlightPlan]
    aircrafts: Dict[str, Aircraft]


@dataclass
class Conflict:
    id: str
    time: int
    hDist: float
    vDist: float
    fpl0: FlightPlan
    fpl1: FlightPlan
    pos0: tuple
    pos1: tuple

    def to_string(self):
        return "{} {} {} {}".format(self.id, self.time, round(self.hDist, 1), round(self.vDist, 1))

    def to_dict(self):
        # return dict(id=self.id, time=self.time, hDist=self.hDist, vDist=self.vDist, pos0=self.pos0, pos1=self.pos1)
        return dict(id=self.id, time=self.time, hDist=self.hDist, vDist=self.vDist,
                    pos0=self.pos0, pos1=self.pos1, start0=self.fpl0.startTime, start1=self.fpl1.startTime)

    def branch(self, shift_list, conflicts_tmp, ac_ac_dict, shift=360):
        [a0, a1] = self.id.split('-')
        conflicts_tmp.append([a0, a1])

        if a0 in ac_ac_dict.keys():
            ac_ac_dict[a0].append(a1)
        else:
            ac_ac_dict[a0] = [a1]

        if a1 in ac_ac_dict.keys():
            ac_ac_dict[a1].append(a0)
        else:
            ac_ac_dict[a1] = [a0]

        if self.time - self.fpl0.startTime < shift:
            shift_list.append(a0)
        if self.time - self.fpl1.startTime < shift:
            shift_list.append(a1)

    def printf(self):
        fpl0, fpl1 = self.fpl0, self.fpl1
        print('-------------------------------------')
        print('|  Conflict ID: ', self.id)
        print('|Conflict Time: ', self.time)
        print('|   H Distance: ', self.hDist)
        print('|   V Distance: ', self.vDist)
        print('|     a0 state: ', self.pos0)
        print('|      a0 info: ', fpl0.startTime, fpl0.min_alt, fpl0.max_alt, fpl0.routing.id)
        print('|     a1 state: ', self.pos1)
        print('|      a1 info: ', fpl1.startTime, fpl1.min_alt, fpl1.max_alt, fpl1.routing.id)
        print('-------------------------------------')


@dataclass
class ConflictScenarioInfo:
    id: str
    time: int
    conflict_ac: List[str]
    other: List[object]
    start: int
    end: int
    fpl_list: List[FlightPlan]

    def to_dict(self):
        [_, _, h_dist, v_dist] = self.other
        return dict(id=self.id, time=self.time, c_ac=self.conflict_ac,
                    fpl=len(self.fpl_list), h_dist=round(h_dist, 1), v_dist=round(v_dist, 1))

    def to_string(self):
        print(self.id, self.conflict_ac, self.time, self.start, self.end, len(self.fpl_list))

