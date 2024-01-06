# Copyright 2022-2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import collections
from typing import Optional


class OutlierDetector:
    """OutlierDetector.

    This class implements an algorithm to detect outliers in sequential
    numeric data (e.g. for gradient/moment norms in optimizers). It relies on a
    delayed moving average which is the moving average of observations from time
    step T-2*`delay_interval` to T-`delay_interval`. The motivation is that
    outliers typically occur in clusters that can potentially stretch for many
    observations, hence it's best to use a delayed moving average to detect
    outliers.

    It defines an outlier as any data point that is `threshold` times larger than the delayed moving average.

    The class assumes data is inserted sequentially at evenly spaced intervals of unit length 1.
    """

    def __init__(self, threshold: float = 7.5, delay_interval: int = 500):

        self.intermediate_data_queue = collections.deque(maxlen=delay_interval)
        self.delayed_moving_average = collections.deque(maxlen=delay_interval)
        self.threshold = threshold

    def insert_observation(self, obs: float) -> bool:
        """Insert observation.

        Inserts obs into the data buffer and returns true if it is an "outlier", defined `threshold` times larger than
        the windowed moving average from [T-2*`delay_interval` : T-`delay_interval`].

        This algorithm first moves recent data into an intermediate circular buffer, and then moves data in to the delayed moving average buffer
        once it is old enough to be evicted from the intermediate data. This is to ensure that we take a delayed moving average that doesn't include recent data.

        Args:
            obs (float): Numeric observation for the current timestep.

        Returns:
            bool: Indicator of whether the most recent observation was an outlier.
        """
        assert self.intermediate_data_queue.maxlen is not None, 'expected maxlen defined'

        if len(self.intermediate_data_queue
              ) >= self.intermediate_data_queue.maxlen:
            # move data from intermediate queue to slow moving average queue
            intermediate_obs = self.intermediate_data_queue.popleft()
            self.delayed_moving_average.append(intermediate_obs)

        self.intermediate_data_queue.append(obs)
        delayed_mva = self.get_delayed_mva()
        return delayed_mva is not None and obs > self.threshold * delayed_mva

    def get_delayed_mva(self) -> Optional[float]:
        if len(self.delayed_moving_average) > 0:
            return sum(self.delayed_moving_average) / len(
                self.delayed_moving_average)
        else:
            return None
