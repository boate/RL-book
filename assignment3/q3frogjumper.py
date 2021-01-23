import sys
sys.path.append('../')

from dataclasses import dataclass
from typing import Mapping, Tuple, Dict, Callable
from rl.distribution import Categorical, Constant
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_decision_process import FinitePolicy, StateActionMapping
from rl.dynamic_programming import evaluate_mrp_result
from rl.dynamic_programming import policy_iteration_result
from rl.dynamic_programming import value_iteration_result

import itertools
import numpy as np


@dataclass(frozen=True)
class LilyPadState:
    # check that at most one of the above is true
    state_num: int

PadJumpMapping = StateActionMapping[LilyPadState, int]

class FrogJumper(FiniteMarkovDecisionProcess[LilyPadState, int]):

    def __init__(
        self,
        highest_num_state: int,
    ):
        self.highest_num_state = highest_num_state
        super().__init__(self.get_action_transition_reward_map())
    
    def rewardf(
        self,
        curr_state: int,
        highest_num_state: int
    ):
        if curr_state == highest_num_state:
            # positive reward for escaping
            return 1.0
        elif curr_state == 0:
            # negative reward for getting eaten
            return -1.0
        else:
            # no reward for an intermediate state
            return 0

    def get_action_transition_reward_map(self) -> PadJumpMapping:
        d: Dict[LilyPadState, Dict[int, Categorical[Tuple[LilyPadState,
                                                          float]]]] = {}
        # every state except the terminal
        for padnum in range(1, self.highest_num_state):
            state: LilyPadState = LilyPadState(padnum)

            #actions to probability distributions
            d1: Dict[int, Categorical[Tuple[LilyPadState, float]]] = {}
 
            # Croak A - denoted as the 0th action
            state_reward_probs_dict: Dict[Tuple[LilyPadState, float], float] =\
                        {(LilyPadState(padnum - 1), self.rewardf(padnum - 1, self.highest_num_state)) :\
                          float(padnum)/float(self.highest_num_state), 
                         (LilyPadState(padnum + 1), self.rewardf(padnum + 1, self.highest_num_state)) :\
                          float(self.highest_num_state - padnum)/float(self.highest_num_state)}
            d1[0] = Categorical(state_reward_probs_dict)

            # Croak B - denoted as the 1st action
            state_reward_probs_dict: Dict[Tuple[LilyPadState, float], float] =\
                        {}
            
            for j in range(0, self.highest_num_state + 1):
                # any other pad except itself
                if j != padnum:
                    state_reward_probs_dict[(LilyPadState(j), self.rewardf(j, self.highest_num_state))] = \
                          1.0/float(self.highest_num_state)
            d1[1] = Categorical(state_reward_probs_dict)

            d[state] = d1
        
        #terminal states
        d[LilyPadState(self.highest_num_state)] = None
        d[LilyPadState(0)] = None

        return d
    
if __name__ == '__main__':
    from pprint import pprint

    user_gamma = 0.9
    toppad = 9
    si_mdp: FiniteMarkovDecisionProcess[LilyPadState, int] =\
        FrogJumper(
            highest_num_state = toppad
        )

    print("MDP Transition Map")
    print("------------------")
    print(si_mdp)

    policies = list(itertools.product([0, 1], repeat=toppad - 1)) 
    print(policies)

    # iterate over each deterministic policy
    for policy in policies:
        print("NEW POLICY")
        print("----------------------------")
        fdp: FinitePolicy[LilyPadState, int] = FinitePolicy(
            {LilyPadState(padnum):
            Constant(policy[padnum - 1]) for padnum in range(1, toppad)}
        )

        print("Policy Map")
        print("----------")
        print(fdp)

        implied_mrp: FiniteMarkovRewardProcess[LilyPadState] =\
            si_mdp.apply_finite_policy(fdp)

        #print(implied_mrp.get_value_function_vec(gamma=user_gamma))
        print("Implied MRP Policy Evaluation Value Function")
        print("--------------")
        pprint(evaluate_mrp_result(implied_mrp, gamma=user_gamma))
        print()


    print("MDP Value Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_vi, opt_policy_vi = value_iteration_result(si_mdp, gamma=user_gamma)
    pprint(opt_vf_vi)
    print(opt_policy_vi)
    print()