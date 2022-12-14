from __future__ import annotations

from flightSim.utils import calc_level


def reset_guidance_with_fpl(guidance, fpl):
    guidance.targetAlt = fpl.max_alt
    guidance.targetHSpd = 0
    guidance.targetCourse = 0


def update_guidance(now, guidance, status, control, profile):
    v_spd = status.vSpd
    performance = status.performance

    if v_spd == 0.0:
        target_spd = performance.normCruiseTAS
    elif v_spd > 0.0:
        target_spd = performance.normClimbTAS
    else:
        target_spd = performance.normDescentTAS

    spd_cmd = control.spdCmd
    if spd_cmd is not None and now >= spd_cmd.assignTime:
        guidance.targetHSpd = performance.min_max_spd(target_spd+spd_cmd.delta, v_spd=v_spd)
    else:
        guidance.targetHSpd = target_spd

    alt_cmd = control.altCmd
    if alt_cmd is not None and now == alt_cmd.assignTime:
        delta = alt_cmd.delta
        target_alt = calc_level(status.alt, status.vSpd, delta)
        if v_spd * delta < 0 or target_alt > 12000 or target_alt < 6000:
            alt_cmd.ok = False
        else:
            guidance.targetAlt = target_alt
        control.transition(mode='Alt')

    hdg_cmd = control.hdgCmd
    if hdg_cmd is None:
        guidance.targetCourse = profile.courseToTarget
        return

    delta, assign_time = hdg_cmd.delta, hdg_cmd.assignTime
    if delta == 0:
        control.transition(mode='Hdg')
        return

    elif now - assign_time == 0:    # 以delta角度出航
        guidance.targetCourse = (delta+status.heading) % 360
    elif now - assign_time == 120:   # 转向后持续60秒飞行，之后以30°角切回航路
        prefix = abs(delta) / delta
        guidance.targetCourse = (-prefix*(abs(delta)+30)+status.heading) % 360
    elif now - assign_time == 240:  # 结束偏置（dogleg机动）
        control.transition(mode='Hdg')
