
from typing import Iterable, Iterator, TypeVar, Dict, Callable, Sequence
from rl.markov_decision_process import FinitePolicy, StateActionMapping
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap, InventoryState
from rl.dynamic_programming import value_iteration_result
import random
from pprint import pprint

S = TypeVar('S')
A = TypeVar('A')

def get_last_vf(iterator, num_traces):
    last = next(iterator)
    for i in range(num_traces):
        last = next(iterator)
    return last

def epsilon_greedy(epsilon, q, state):
    if random.random() < epsilon:
        return random.choice(list(q[state].keys()))
    else:
        return max(q[state], key=q[state].get)

def TabularSarsa(non_terminal_states: Sequence[S],
        q: Dict[S, Dict[A, float]],
        transition_map:  StateActionMapping[S, int],
        gamma: float,
        start_state: S,
        num_episodes = 1000,
        ) -> Iterator[Dict[S, Dict[A, float]]]:

    k = 1.0
    for episode in range(num_episodes):
        counter: Dict[S, float] = {}
        # start at a random state?
        state = start_state
        epsilon = 1.0 / k
        action = epsilon_greedy(epsilon, q, state)
        # until state is terminal
        iter = 0
        while state in non_terminal_states and iter < 1000:
            k += 1
            iter += 1
            epsilon = 1.0/k
            nxtstate, reward = transition_map[state][action].sample()
            nxtaction = epsilon_greedy(epsilon, q, nxtstate)

            # weighting
            if state in counter:
                counter[state] += 1.0
            else:
                counter[state] = 1.0
            weight = 1.0/counter[state]
            q[state][action] += weight*(reward + gamma*q[nxtstate][nxtaction] - q[state][action])
            state = nxtstate
            action = nxtaction

        yield q

if __name__ == '__main__':
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.9

    si_mdp = SimpleInventoryMDPCap(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda,
        holding_cost=user_holding_cost,
        stockout_cost=user_stockout_cost
    )


    print("MDP Value Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_vi, opt_policy_vi = value_iteration_result(si_mdp, gamma=user_gamma)
    pprint(opt_vf_vi)
    print(opt_policy_vi)
    print()

    print("Tabular SARSA")
    print("--------------")
    actions = {s: list(si_mdp.actions(s)) for s in si_mdp.non_terminal_states}
    tabsarsa = TabularSarsa(si_mdp.non_terminal_states,
        {s: {a: 0 for a in actions[s]} for s in si_mdp.non_terminal_states},
        si_mdp.get_action_transition_reward_map(),
        user_gamma,
        InventoryState(0,0),
        num_episodes = 2000)
    result = get_last_vf(tabsarsa, 1999)
    print("Optimal Policy")
    pprint({s: max(result[s]) for s in si_mdp.non_terminal_states})
    print("Optimal Value Function")
    pprint({s: max(result[s].values()) for s in si_mdp.non_terminal_states})
