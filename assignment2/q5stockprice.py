# Extend one of the Stock Price examples in Chapter 1 to be a Markov Reward Process by defining
# a Reward Rt that is a function f of the Stock price Xt at each time t. Program it as a class
# that implements the interface of the @abstractclass MarkovRewardProcess and allow flexibility in
# specifying your own function f. This is an infinite-states, non-terminating MRP. Compute it’s
# Value Function for any discount factor 0 ≤ γ < 1.

from dataclasses import dataclass
from typing import Tuple, Callable
import numpy as np
import itertools
from rl.distribution import Categorical, Constant
from rl.markov_process import MarkovRewardProcess
from rl.gen_utils.common_funcs import get_logistic_func


@dataclass(frozen=True)
class StateMP1:
    price: int


@dataclass
class StockPriceMP1(MarkovRewardProcess[StateMP1]):

    level_param: int  # level to which price mean-reverts
    reward_func: Callable
    alpha1: float = 0.25  # strength of mean-reversion (non-negative value)

    def up_prob(self, state: StateMP1) -> float:
        return get_logistic_func(self.alpha1)(self.level_param - state.price)

    def transition_reward(self, state: StateMP1) -> \
            Categorical[Tuple[StateMP1, float]]:

        up_p = self.up_prob(state)
        up_reward = self.reward_func(state.price + 1)
        down_reward = self.reward_func(state.price - 1)
        return Categorical({
            (StateMP1(state.price + 1), up_reward): up_p,
            (StateMP1(state.price - 1), down_reward): 1 - up_p
        })


def process1_price_traces(
    start_price: int,
    level_param: int,
    alpha1: float,
    reward_func: Callable,
    time_steps: int,
    num_traces: int
) -> np.ndarray:
    mp = StockPriceMP1(level_param=level_param,
                       alpha1=alpha1,
                       reward_func=reward_func)
    start_state_distribution = Constant(StateMP1(price=start_price))
    return np.vstack([
        np.fromiter((s.price for s in itertools.islice(
            mp.simulate(start_state_distribution),
            time_steps + 1
        )), float) for _ in range(num_traces)])


# simple reward example equal to price itself
def price_reward(price):
    return price


if __name__ == '__main__':
    start_price: int = 100
    level_param: int = 100
    alpha1: float = 0.25
    alpha2: float = 0.75
    alpha3: float = 1.0
    time_steps: int = 100
    num_traces: int = 1000
    reward_func = price_reward

    process1_traces: np.ndarray = process1_price_traces(
        start_price=start_price,
        level_param=level_param,
        alpha1=alpha1,
        reward_func=reward_func,
        time_steps=time_steps,
        num_traces=num_traces
    )

    trace1 = process1_traces[0]

    print(trace1)

    # TODO value function for this process
