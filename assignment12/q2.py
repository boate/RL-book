""" Implementing TD(lambda) Prediction algorithm"""

from typing import Iterable, Iterator, TypeVar, Dict, Callable, List
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
from rl.distribution import Choose
from rl.function_approx import Tabular
from rl.td import td_prediction
from pprint import pprint
from assignment11.q1and2 import td_tabular
from rl.function_approx import FunctionApprox
import rl.markov_process as mp
import rl.iterate as iterate

S = TypeVar('S')

def get_last_vf(iterator, num_traces):
    last = next(iterator)
    for i in range(num_traces):
        last = next(iterator)
    return last

def weight_func(counter: Dict[S, float], S) -> float:
    # fixed learning rate
    return 0.05

def td_lambda_tabular(
        transitions: Iterable[mp.TransitionStep[S]],
        value_func: Dict[S,float],
        gamma: float,
        weight_func: Callable,
        lamb: float
) -> Iterator[Dict[S,float]]:
    eligibility: Dict[S, float] = {state: 0 for state in value_func.keys()}
    counter: Dict[S, float] = {}
    for step in transitions:
        state = step.state
        reward = step.reward
        for st in value_func.keys():
            eligibility[st] = gamma*lamb*eligibility[st] + 1.0*(st == state)
        nxtstate = step.next_state
        if state in counter:
            counter[state] += 1
        else:
            counter[state] = 1
        weight = weight_func(counter, state)
        value_func[state] += weight * eligibility[state] * (reward + gamma*value_func[nxtstate] - value_func[state])

        yield value_func

def td_lambda_funcapprox(
        transitions: Iterable[mp.TransitionStep[S]],
        approx_0: FunctionApprox[S],
        gamma: float,
        lamb: float,
        states: List[S],
) -> Iterator[FunctionApprox[S]]:
    """Note: without weight func"""
    eligibility: Dict[S, float] = {state: 0 for state in states}
    def step(
            v: FunctionApprox[S],
            transition: mp.TransitionStep[S]
    ) -> FunctionApprox[S]:
        for state in states:
            eligibility[state] = gamma * lamb * eligibility[state] + 1.0 * (transition.state == state)
        return v.update([(
            transition.state,
            eligibility[transition.state]*(transition.reward + gamma * v(transition.next_state))
        )])

    return iterate.accumulate(transitions, step, initial=approx_0)


if __name__ == '__main__':
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.9

    si_mrp = SimpleInventoryMRPFinite(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda,
        holding_cost=user_holding_cost,
        stockout_cost=user_stockout_cost
    )

    print("Value Function - Solution")
    print("--------------")
    si_mrp.display_value_function(gamma=user_gamma)
    print()

    print("Value Function - TD FuncApprox")
    print("--------------")
    transitions = si_mrp.simulate_reward(Choose(set(si_mrp.non_terminal_states)))
    td_approx = td_prediction(
        transitions=transitions,
        approx_0=Tabular(),
        γ=user_gamma,
    )
    result = get_last_vf(td_approx, 100000)
    pprint({state: result.evaluate([state])[0] for state in si_mrp.non_terminal_states})

    print("Value Function - TD Lambda Function Approx")
    print("--------------")
    transitions = si_mrp.simulate_reward(Choose(set(si_mrp.non_terminal_states)))
    td_approxlamb = td_lambda_funcapprox(
        transitions=transitions,
        approx_0=Tabular(),
        gamma=user_gamma,
        lamb=0.5,
        states = si_mrp.non_terminal_states
    )
    result = get_last_vf(td_approxlamb, 100000)
    pprint({state: result.evaluate([state])[0] for state in si_mrp.non_terminal_states})

    print("Value Function - TD Tabular")
    print("--------------")
    transitions = si_mrp.simulate_reward(Choose(set(si_mrp.non_terminal_states)))
    td_tab = td_tabular(
        transitions=transitions,
        value_func={state: 1 for state in si_mrp.non_terminal_states},
        gamma=user_gamma,
        weight_func=weight_func
    )
    result = get_last_vf(td_tab, 40000)
    pprint(result)

    print("Value Function - TD Lambda")
    print("--------------")
    transitions = si_mrp.simulate_reward(Choose(set(si_mrp.non_terminal_states)))
    td_tablamb = td_lambda_tabular(
        transitions=transitions,
        value_func={state: 1 for state in si_mrp.non_terminal_states},
        gamma=user_gamma,
        weight_func=weight_func,
        lamb=0.5
    )
    result = get_last_vf(td_tablamb, 40000)
    pprint(result)

