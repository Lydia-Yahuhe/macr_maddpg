from flightSim.aircraft import AltCmd, HdgCmd, SpdCmd

CmdCount = 6


def parse_and_assign_cmd(now: int, idx: list, target):
    # alt cmd
    time_idx = int((idx[0]+1)*100)
    alt_idx = int(idx[1]*3)
    alt_cmd = AltCmd(delta=alt_idx * 300.0, assignTime=now+time_idx)
    target.assign_cmd(alt_cmd)
    print('{:>3d}({:>+4.2f}) {:>+3d}({:>+4.2f})'.format(time_idx, idx[0], alt_idx, idx[1]), end='\t')
    # print('{:>3d} {:>+3d}'.format(time_idx, alt_idx), end='\t')

    # hdg cmd
    time_idx = int((idx[2]+1)*100)
    hdg_idx = idx[3]*4
    hdg_cmd = HdgCmd(delta=hdg_idx * 15.0, assignTime=now+time_idx)
    target.assign_cmd(hdg_cmd)
    print('{:>3d}({:>+4.2f}) {:>+3.1f}({:>+4.2f})'.format(time_idx, idx[2], hdg_idx, idx[3]), end='\t')
    # print('{:>3d} {:>+3.1f}'.format(time_idx, hdg_idx), end='\t')

    # spd cmd
    time_idx = int((idx[4]+1)*100)
    spd_idx = idx[5]*3
    spd_cmd = SpdCmd(delta=spd_idx * 10, assignTime=now+time_idx)
    target.assign_cmd(spd_cmd)
    print('{:>3d}({:>+4.2f}) {:>+3.1f}({:>+4.2f})'.format(time_idx, idx[4], spd_idx, idx[5]), end='\t')
    # print('{:>3d} {:>+3.1f}'.format(time_idx, spd_idx), end='\t')

    return [alt_cmd, hdg_cmd, spd_cmd]


def rew_for_cmd(conflict_acs, cmd_info):
    rewards = []
    for ac in conflict_acs:
        [alt_cmd, hdg_cmd, spd_cmd] = cmd_info[ac]
        rew_alt = 0.3 - abs(alt_cmd.delta) / 3000.0
        rew_hdg = 0.4 - abs(hdg_cmd.delta) / 150.0
        rew_spd = 0.3 - abs(spd_cmd.delta) / 100.0
        rewards.append(rew_alt+rew_spd+rew_hdg)
    return rewards
