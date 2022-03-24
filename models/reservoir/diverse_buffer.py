from .prs import PRS
import numpy as np


class DiverseBuffer(PRS):
    def state(self, key):
        """override state function
        :param key: data names
        :return: states of reservoir
        """
        if key == "corrupts":
            ncorrupt = sum(self.rsvr[key]) - sum(self.rsvr[key][self.rsvr_available_idx])  # exclude examples in pool
            return "#normal data: {}, \t #corrupted data: {}".format(len(self) - ncorrupt, ncorrupt)

        return ""

