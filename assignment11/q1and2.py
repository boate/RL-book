from typing import Iterable, Iterator, TypeVar, Dict, Callable

import rl.markov_process as mp
from rl.returns import returns

S = TypeVar('S')

def weight_func(counter: Dict[S, float], S) -> float:
    return 1.0/counter[S]

def mc_tabular(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        value_func : Dict[S, float],
        γ: float,
        weight_func : Callable,
        tolerance: float = 1e-6
) -> Iterator[Dict[S, float]]:

    episodes: Iterator[Iterator[mp.ReturnStep[S]]] = \
        (returns(trace, γ, tolerance) for trace in traces)
    counter: Dict[S, float] = {}

    for episode in episodes:
        for step in episode:

            state = step.state
            if state in counter:
                counter[state] += 1
            else:
                counter[state] = 1
            weight = weight_func(counter, state)
            value_func[state] = (1-weight)*value_func[state] + weight*step.return_

        yield value_func

def td_tabular(
        transitions: Iterable[mp.TransitionStep[S]],
        value_func: Dict[S,float],
        gamma: float,
        weight_func: Callable,
        alpha: float = 1,
) -> Iterator[Dict[S,float]]:

    counter: Dict[S, float] = {}
    for step in transitions:
        state = step.state
        reward = step.reward
        nxtstate = step.next_state

        if state in counter:
            counter[state] += 1
        else:
            counter[state] = 1
        weight = weight_func(counter, state)
        value_func[state] += alpha * weight * (reward + gamma*value_func[nxtstate] - value_func[state])

        yield value_func