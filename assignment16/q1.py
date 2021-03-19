from rl.chapter7.asset_alloc_discrete import AssetAllocDiscrete
from typing import Sequence, Callable, Tuple, TypeVar, Optional
from rl.distribution import Gaussian, Categorical
from rl.markov_decision_process import Policy
from rl.function_approx import DNNSpec
import numpy as np

S = TypeVar('S')
A = TypeVar('A')

# softmax policy following problem 3 of assignment 16
class SoftMax(Policy[S,A]):
    def __init__(self, theta: Sequence, features, actions):
        self.theta = theta
        self.features: Sequence[Callable[[S, A], float]] = features
        self.actions = actions

    def act(self, state) -> Optional[Categorical[A]]:
        d = {}
        normalizer = 0.0
        for action in self.actions:
            normalizer += np.exp(np.dot([func(state, action) for func in self.features], self.theta))

        for action in self.actions:
            result = np.exp(np.dot([func(state, action) for func in self.features], self.theta))/normalizer
            d[action] = result

        return Categorical(d)

def reinforce(features, episodes, returns, actions, simulator, gamma, alpha):

    # initialize policy
    theta = np.zeros(len(features))

    for i in range(episodes):
        # set up sim and get returns
        current_sim = simulator(init_wealth_distr, SoftMax(theta, features, actions))
        episode = list(returns(current_sim, gamma, 1e-6))

        for timestep in range(len(episode)):
            state = episode[timestep].state
            action = episode[timestep].action
            phi = [func(state, action) for func in features]
            #  calculate gradient here
            # how to derive softmax??

            gradient = 0
            theta += alpha * gamma**t * gradient * episode[t].return_

    return theta

if __name__ == '__main__':

    from pprint import pprint

    steps: int = 4
    μ: float = 0.13
    σ: float = 0.2
    r: float = 0.07
    a: float = 1.0
    init_wealth: float = 1.0
    init_wealth_var: float = 0.1

    excess: float = μ - r
    var: float = σ * σ
    base_alloc: float = excess / (a * var)

    risky_ret: Sequence[Gaussian] = [Gaussian(μ=μ, σ=σ) for _ in range(steps)]
    riskless_ret: Sequence[float] = [r for _ in range(steps)]
    utility_function: Callable[[float], float] = lambda x: - np.exp(-a * x) / a
    alloc_choices: Sequence[float] = np.linspace(
        2 / 3 * base_alloc,
        4 / 3 * base_alloc,
        11
    )
    feature_funcs: Sequence[Callable[[Tuple[float, float]], float]] = \
        [
            lambda _: 1.,
            lambda w_x: w_x[0],
            lambda w_x: w_x[1],
            lambda w_x: w_x[1] * w_x[1]
        ]
    dnn: DNNSpec = DNNSpec(
        neurons=[],
        bias=False,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: - np.sign(a) * np.exp(-x),
        output_activation_deriv=lambda y: -y
    )
    init_wealth_distr: Gaussian = Gaussian(μ=init_wealth, σ=init_wealth_var)

    aad: AssetAllocDiscrete = AssetAllocDiscrete(
        risky_return_distributions=risky_ret,
        riskless_returns=riskless_ret,
        utility_func=utility_function,
        risky_alloc_choices=alloc_choices,
        feature_functions=feature_funcs,
        dnn_spec=dnn,
        initial_wealth_distribution=init_wealth_distr
    )

    print("Analytical Solution")
    print("-------------------")
    print()

    for t in range(steps):
        print(f"Time {t:d}")
        print()
        left: int = steps - t
        growth: float = (1 + r) ** (left - 1)
        alloc: float = base_alloc / growth
        val: float = - np.exp(- excess * excess * left / (2 * var)
                              - a * growth * (1 + r) * init_wealth) / a
        bias_wt: float = excess * excess * (left - 1) / (2 * var) + \
            np.log(np.abs(a))
        w_t_wt: float = a * growth * (1 + r)
        x_t_wt: float = a * excess * growth
        x_t2_wt: float = - var * (a * growth) ** 2 / 2

        print(f"Opt Risky Allocation = {alloc:.3f}, Opt Val = {val:.3f}")
        print(f"Bias Weight = {bias_wt:.3f}")
        print(f"W_t Weight = {w_t_wt:.3f}")
        print(f"x_t Weight = {x_t_wt:.3f}")
        print(f"x_t^2 Weight = {x_t2_wt:.3f}")
        print()
