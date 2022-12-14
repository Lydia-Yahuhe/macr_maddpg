from dataclasses import dataclass


class ATCCmd:
    pass


@dataclass
class AltCmd(ATCCmd):
    delta: float
    currentAlt: float = 0.0
    targetAlt: float = 0.0
    assignTime: int = 0
    ok: bool = True
    cmdType = "Altitude"

    def to_dict(self):
        return {'ALT': '{},{}'.format(round(self.delta, 2), self.assignTime)}

    def to_string(self):
        return '{},{},{}'.format(self.cmdType, self.assignTime, self.delta)

    def __str__(self):
        return 'ALTCMD: <TIME:%d, DELTA:%0.2f, TARGET:%0.2f>' % (self.assignTime, self.delta, self.targetAlt)

    def __repr__(self):
        return 'ALTCMD: <TIME:%d, DELTA:%0.2f, TARGET:%0.2f>' % (self.assignTime, self.delta, self.targetAlt)


@dataclass
class SpdCmd(ATCCmd):
    delta: float
    assignTime: int = 0
    ok: bool = True
    currentSpd: float = 0.0
    targetSpd: float = 0.0
    cmdType = "Speed"

    def to_dict(self):
        return {'SPD': '{},{}'.format(round(self.delta, 2), self.assignTime)}

    def to_string(self):
        return '{},{},{}'.format(self.cmdType, self.assignTime, self.delta)

    def __str__(self):
        return 'SPDCMD: <TIME:%d, DELTA:%0.2f, TARGET:%0.2f>' % (self.assignTime, self.delta, self.targetSpd)

    def __repr__(self):
        return 'SPDCMD: <TIME:%d, DELTA:%0.2f, TARGET:%0.2f>' % (self.assignTime, self.delta, self.targetSpd)


@dataclass
class HdgCmd(ATCCmd):
    delta: float
    assignTime: int = 0
    ok: bool = True
    currentHdg: float = 0.0
    targetHdg: float = 0.0
    cmdType = "Heading"

    def to_dict(self):
        return {'HDG': '{},{}'.format(round(self.delta, 2), self.assignTime)}

    def to_string(self):
        return '{},{},{}'.format(self.cmdType, self.assignTime, self.delta)

    def __str__(self):
        return 'OFFSET: <TIME:%d, DELTA:%0.2f, TARGET:%d>' % (self.assignTime, self.delta, self.targetHdg)

    def __repr__(self):
        return 'OFFSET: <TIME:%d, DELTA:%0.2f, TARGET:%d>' % (self.assignTime, self.delta, self.targetHdg)
