import pymongo
from tqdm import tqdm

from flightSim.model import *


def load_waypoint(database):
    ret = {}
    for e in database['Waypoint'].find():
        key = e['id']
        ret[key] = Waypoint(id=key,
                            location=Point2D(e['point']['lng'], e['point']['lat']))

    for e in database["Airport"].find():
        key = e['id']
        ret[key] = Waypoint(id=key,
                            location=Point2D(e['location']['lng'], e['location']['lat']))
    return ret


def load_routing(database, wpt_dict):
    ret = {}
    for e in database['Routing'].find():
        key = e['id']
        ret[key] = Routing(id=key,
                           waypointList=[wpt_dict[wpt_id] for wpt_id in e['waypointList']])
    return ret


def load_aircraft(database):
    ret = {}
    for e in database['AircraftRandom'].find():
        key = e['id']
        ret[key] = Aircraft(id=key,
                            aircraftType=aircraftTypes[e['aircraftType']])
    return ret


def load_flight_plan(database, aircraft, routes):
    ret = {}
    for e in database['FlightPlan'].find():
        key = e['id']
        ret[key] = FlightPlan(id=key,
                              min_alt=0,
                              routing=routes[e['routing']],
                              startTime=e['startTime'],
                              aircraft=aircraft[e['aircraft']],
                              max_alt=e['flightLevel'])
    return ret


def load_data_set():
    connection = pymongo.MongoClient('localhost')
    database = connection['admin']

    wpt_dict = load_waypoint(database)
    ac_dict = load_aircraft(database)
    route_dict = load_routing(database, wpt_dict)
    fpl_dict = load_flight_plan(database, ac_dict, route_dict)

    connection.close()
    return DataSet(wpt_dict, route_dict, fpl_dict, ac_dict)


def load_and_split_data(col='scenarios_big_flow_new', size=None, ratio=0.8, density=1):
    if size is None:
        size = int(1e6)

    route_dict = load_data_set().routings
    collection = pymongo.MongoClient('localhost')['admin'][col]

    scenarios, count = [], 0
    for info in tqdm(collection.find(), desc='Loading from database'):
        del info['_id']

        c_times = [c['time'] for c in info['conflict_list']]

        fpl_list, candi = [], {}
        for i, fpl in enumerate(info['fpl_list']):
            if i % density != 0:
                continue

            # start time
            start = fpl['startTime']

            # routing
            routing, section = route_dict[fpl['routing']], fpl['other']
            routing = Routing(id=routing.id, waypointList=routing.get_points(section=section), other=section)

            # aircraft
            ac = Aircraft(id=fpl['aircraft'], aircraftType=aircraftTypes[fpl['acType']])

            # flight plan
            fpl = FlightPlan(id=fpl['id'], routing=routing, aircraft=ac, startTime=start,
                             min_alt=fpl['min_alt'], max_alt=fpl['max_alt'])

            fpl_list.append(fpl)

            if start in candi.keys():
                candi[start].append(fpl.id)
            else:
                candi[start] = [fpl.id]

        scenarios.append(dict(id=str(count + 1), clock=min(c_times), fpl_list=fpl_list, candi=candi))
        count += 1

        if count >= size:
            break

    split_size = int(count * ratio)
    return scenarios[:split_size], scenarios[split_size:]
