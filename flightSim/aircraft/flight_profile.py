from __future__ import annotations

import math

from flightSim.utils import KM2M


def reset_profile_with_fpl(profile, fpl):
    profile.route = fpl.routing
    profile.make_leg_from_waypoint()

    profile.curLegIdx = 0
    profile.update_cur_next_leg()


def update_profile(profile, status):
    if target_passed(profile, status):
        profile.curLegIdx += 1
        if not profile.update_cur_next_leg():
            status.phase = 'Finished'
            return

    target = profile.target
    if not target:
        profile.distToTarget = 0
        profile.courseToTarget = 0
    else:
        profile.distToTarget = status.location.distance_to(target.location)
        profile.courseToTarget = status.location.bearing(target.location)


def target_passed(profile, phsyData):
    dist = profile.distToTarget
    if dist >= 20 * KM2M:
        return False

    h_spd = phsyData.hSpd
    if dist < h_spd * 1:
        return True

    if profile.nextLeg is not None:
        turnAngle = (profile.nextLeg.course - profile.curLeg.course) % 360
        if dist <= calc_turn_prediction(h_spd, turnAngle, phsyData.performance.normTurnRate):
            return True
    diff = (phsyData.heading - profile.courseToTarget) % 360
    return 270 > diff >= 90


def calc_turn_prediction(spd, turnAngle, turnRate):
    if turnAngle > 180:
        turnAngle = turnAngle - 360
    turnAngle = abs(turnAngle)
    turnRadius = spd / math.radians(turnRate)
    return turnRadius * math.tan(math.radians(turnAngle / 2))
