from typing import Any
from Tracker import Tracker


class Consumer:
    def __init__(self):
        try:
            self._tracker_handler = Tracker()
        except Exception as e:
            print(e)


    def __call__(self, frame):
        return self._tracker_handler(frame)


