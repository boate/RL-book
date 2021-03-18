import itertools
import numpy as np
import sys

from dataclasses import dataclass
from typing import Mapping, Tuple, Dict, Callable
from rl.distribution import Categorical, Constant
from rl.markov_process import MarkovProcess, Transition
from rl.chapter9.order_book import DollarsAndShares, OrderBook

sys.path.append('../')


class OrderBookProcess(MarkovProcess[OrderBook]):

    def __init__(
        self,
        orderbook,
    ):
        self.curr_ob = orderbook

        super().__init__(self.get_transition_map())

    def transition(self, state: OrderBook) -> Optional[Distribution[OrderBook]]:
        pass
