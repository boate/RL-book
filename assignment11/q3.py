from q1and2 import mc_tabular, weight_func, td_tabular
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
from rl.distribution import Choose
from rl.function_approx import Tabular
from rl.monte_carlo import mc_prediction
from rl.td import td_prediction
from pprint import pprint

def get_last_vf(iterator, num_traces):
    last = next(iterator)
    for i in range(num_traces):
        last = next(iterator)
    return last

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

    print("Value Function - MC FuncApprox")
    print("--------------")
    traces = si_mrp.reward_traces(Choose(set(si_mrp.non_terminal_states)))
    mc_approx = mc_prediction(
        traces=traces,
        approx_0=Tabular(),
        γ=user_gamma
    )
    result = get_last_vf(mc_approx, 10000)
    pprint({state: result.evaluate([state])[0] for state in si_mrp.non_terminal_states})

    print("Value Function - MC Tabular")
    print("--------------")
    traces = si_mrp.reward_traces(Choose(set(si_mrp.non_terminal_states)))
    mc_tab = mc_tabular(
        traces=traces,
        value_func = {state: 1 for state in si_mrp.non_terminal_states},
        γ=user_gamma,
        weight_func=weight_func,
    )
    result = get_last_vf(mc_tab, 10000)
    pprint(result)

    print("Value Function - TD FuncApprox")
    print("--------------")
    transitions = si_mrp.simulate_reward(Choose(set(si_mrp.non_terminal_states)))
    td_approx = td_prediction(
        transitions=transitions,
        approx_0=Tabular(),
        γ=user_gamma,
    )
    result = get_last_vf(td_approx, 40000)
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

