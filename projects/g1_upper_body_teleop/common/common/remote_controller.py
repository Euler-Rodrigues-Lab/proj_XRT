from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
else:
    WirelessController_ = Any

class KeyMap:
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F3 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0
        self.button = [0] * 16

    def set(self, data: WirelessController_):
        # wireless_remote
        keys = data.keys
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = data.lx
        self.rx = data.rx
        self.ry = data.ry
        self.ly = data.ly
