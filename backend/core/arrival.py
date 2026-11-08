"""Arrival confirmation for the preset 242 exit (DCC plan meters)."""
import math


class ExitArrival:
    def __init__(self):
        self.matches = 0
        self.arrived = False

    def update(self, localized, room_id, xy, role):
        previous = self.arrived
        if role != "occupant":
            self.matches = 0
            self.arrived = False
        elif not localized:
            self.matches = 0
        else:
            # 242 lobby keyframes are centered at (84, 61.5); exterior
            # door F2-D56 is at (84.29, 63.29). Confirm both area and proximity.
            at_exit = room_id == "F2-2ST6" and xy is not None and math.hypot(
                xy[0] - 84.29, xy[1] - 63.29
            ) <= 3.0
            self.matches = self.matches + 1 if at_exit else 0
            if not at_exit:
                self.arrived = False
            elif self.matches >= 2:
                self.arrived = True
        return self.arrived, self.arrived and not previous
