import unittest

from backend.core.arrival import ExitArrival


class ExitArrivalTests(unittest.TestCase):
    def test_confirm_arrival_and_announce_once(self):
        arrival = ExitArrival()
        self.assertEqual(arrival.update(True, 'F2-2ST6', [84, 61.5], 'occupant'), (False, False))
        self.assertEqual(arrival.update(True, 'F2-2ST6', [84, 61.5], 'occupant'), (True, True))
        self.assertEqual(arrival.update(True, 'F2-2ST6', [84, 61.5], 'occupant'), (True, False))
        self.assertEqual(arrival.update(False, None, None, 'occupant'), (True, False))
        self.assertEqual(arrival.update(True, 'F2-219', [79, 55], 'occupant'), (False, False))

    def test_reject_wrong_room_distant_match_and_tracking_gap(self):
        arrival = ExitArrival()
        for room, xy in [('F2-232', [40.1, 60]), ('F2-219', [84, 61.5]), ('F2-2ST6', [79, 55])]:
            for _ in range(4):
                self.assertEqual(arrival.update(True, room, xy, 'occupant'), (False, False))
        arrival.update(True, 'F2-2ST6', [84, 61.5], 'occupant')
        arrival.update(False, None, None, 'occupant')
        self.assertEqual(arrival.update(True, 'F2-2ST6', [84, 61.5], 'occupant'), (False, False))
        self.assertEqual(arrival.update(True, 'F2-2ST6', [84, 61.5], 'firefighter'), (False, False))
