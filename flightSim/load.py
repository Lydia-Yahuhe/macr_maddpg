import numpy as np

import pymongo
from tqdm import tqdm

from flightSim.model import Waypoint, Routing, Aircraft, aircraftTypes, FlightPlan, DataSet, Point2D


def load_waypoint(db):
    wpts = {}
    for e in db['Waypoint'].find():
        wpts[e['id']] = Waypoint(id=e['id'], location=Point2D(e['point']['lng'], e['point']['lat']))

    cursor = db["Airport"].find()
    for pt in cursor:
        wpt = Waypoint(id=pt['id'], location=Point2D(pt['location']['lng'], pt['location']['lat']))
        wpts[wpt.id] = wpt

    return wpts


def load_routing(db, wpts):
    ret = {}
    cursor = db['Routing'].find()
    # for e in cursor:
    #     wptList = [wpts[e["departureAirport"]]]
    #     for wptId in e['waypointList']:
    #         wptList.append(wpts[wptId])
    #     wptList.append(wpts[e["arrivalAirport"]])
    #     r = Routing(e['id'], wptList)
    #     ret[r.id] = r

    for e in cursor:
        wptList = []
        for wptId in e['waypointList']:
            wptList.append(wpts[wptId])
        r = Routing(id=e['id'], waypointList=wptList)
        ret[r.id] = r

    return ret


def load_aircraft(db):
    ret = {}
    cursor = db['AircraftRandom'].find()
    for e in cursor:
        info = Aircraft(id=e['id'], aircraftType=aircraftTypes[e['aircraftType']])
        ret[info.id] = info

    return ret


def load_flight_plan(db, aircraft, routes):
    ret = {}
    cursor = db['FlightPlan'].find()
    for e in cursor:
        a = aircraft[e['aircraft']]

        fpl = FlightPlan(
            id=e['id'],
            min_alt=0,
            routing=routes[e['routing']],
            startTime=e['startTime'],
            aircraft=a,
            max_alt=e['flightLevel']
        )

        ret[fpl.id] = fpl

    return ret


def load_data_set():
    connection = pymongo.MongoClient('localhost')
    db = connection['admin']

    wpts = load_waypoint(db)
    aircrafts = load_aircraft(db)
    routes = load_routing(db, wpts)
    fpls = load_flight_plan(db, aircrafts, routes)

    connection.close()
    return DataSet(wpts, routes, fpls, aircrafts)


routings = load_data_set().routings


def load_and_split_data(col='scenarios_big_flow_new', size=None, ratio=0.8):
    data_set = load_data_set()
    route_dict = data_set.routings

    database = pymongo.MongoClient('localhost')['admin']
    collection = database[col]

    scenarios, count = [], 0
    for info in tqdm(collection.find(), desc='Loading from database'):
        del info['_id']
        min_c_time, conflict_acs = 1e6, []
        for c_info in info['conflict_list']:
            min_c_time = min(min_c_time, c_info['time'])
            conflict_acs += c_info['id'].split('-')
        conflict_acs = list(set(conflict_acs))

        fpl_list, candi = [], {}
        for i, fpl in enumerate(info['fpl_list']):
            # if i % 3 == 0 and fpl['id'] not in conflict_acs:
            #     continue

            # routing
            routing, section = route_dict[fpl['routing']], fpl['other']
            routing = Routing(id=routing.id, waypointList=routing.get_points(section=section), other=section)

            # aircraft
            ac = Aircraft(id=fpl['aircraft'], aircraftType=aircraftTypes[fpl['acType']])

            # flight plan
            fpl = FlightPlan(id=fpl['id'], routing=routing, aircraft=ac, startTime=fpl['startTime'],
                             min_alt=fpl['min_alt'], max_alt=fpl['max_alt'])
            fpl_list.append(fpl)

            start = fpl.startTime
            if start in candi.keys():
                candi[start].append(fpl.id)
            else:
                candi[start] = [fpl.id]

        scenarios.append(dict(id=str(count+1), clock=min_c_time, fpl_list=fpl_list, candi=candi))
        count += 1

        if size is not None and count >= size:
            break

    split_size = int(size*ratio)
    return scenarios[:split_size], scenarios[split_size:]
