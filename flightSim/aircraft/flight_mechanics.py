from __future__ import annotations

from flightSim.model import compute_performance


def reset_status_with_fpl(pdata, fpl):
    pdata.alt = fpl.min_alt
    compute_performance(pdata.acType, fpl.min_alt, pdata.performance)
    performance = pdata.performance
    pdata.hSpd = performance.normCruiseTAS
    pdata.vSpd = 0

    [wpt0, wpt1] = fpl.routing.waypointList[:2]
    pdata.location.set(wpt0.location)
    pdata.heading = wpt0.bearing(wpt1)
    pdata.phase = 'EnRoute'


def update_status(phsyData, guidance):
    move_horizontal(phsyData, guidance)
    move_vertical(phsyData, guidance)
    update_performance(phsyData)


def move_horizontal(pdata, guidance):
    preHSpd = pdata.hSpd
    if pdata.hSpd > guidance.targetHSpd:
        dec = pdata.acType.normDeceleration
        pdata.hSpd = max(preHSpd - dec * 1, guidance.targetHSpd)
    elif pdata.hSpd < guidance.targetHSpd:
        acc = pdata.acType.normAcceleration
        pdata.hSpd = min(preHSpd + acc * 1, guidance.targetHSpd)
        
    performance = pdata.performance
    diff = (guidance.targetCourse - pdata.heading) % 360
    diff = diff-360 if diff > 180 else diff
    if abs(diff) > 90:
        turn = performance.maxTurnRate * 1
    else:
        turn = performance.normTurnRate * 1
    diff = min(max(-turn, diff), turn)
    pdata.heading = (pdata.heading + diff) % 360

    pdata.location.move(pdata.heading, (preHSpd + pdata.hSpd) * 1 / 2)


def move_vertical(pdata, guidance):
    diff = guidance.targetAlt - pdata.alt

    if diff < 0:
        v_spd = max(-pdata.performance.normDescentRate * 1, diff)
    elif diff > 0:
        v_spd = min(pdata.performance.normClimbRate * 1, diff)
    else:
        v_spd = 0

    pdata.alt += v_spd
    pdata.vSpd = v_spd


def update_performance(pdata):
    if pdata.performance.altitude != pdata.alt:
        compute_performance(pdata.acType, pdata.alt, pdata.performance)
