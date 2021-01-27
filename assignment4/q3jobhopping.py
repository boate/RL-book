# custom job hopper environment
import random
import numpy as np
from typing import Tuple, Dict
from pprint import pprint

class JobHopper():

    def __init__(
        self,
        rewards: list,
        alpha: float,
        gamma: float
    ):
        self.rewards = rewards
        self.alpha = alpha
        self.gamma = gamma
        self.nstates = len(rewards)
        self.actions = [0, 1]

    def transition_prob(
            self,
            state: int,
            action: int,
            nxstate: int) -> float:

        #accept
        if action == 1:
            if state == 0:
                if nxstate != 0:
                    return 1.0/(self.nstates - 1)
                else:
                    return 0
            else:
                if nxstate != 0:
                    if nxstate != state:
                        return self.alpha/(self.nstates - 1)
                    else:
                        return self.alpha/(self.nstates - 1) + (1-alpha)
                else:
                    return 0 
        #reject
        else:
            if state == 0:
                if nxstate != 0:
                    return 0
                else:
                    return 1
            else:
                if nxstate != 0:
                    if nxstate == state:
                        return 1-self.alpha
                    else:
                        return 0
                else:
                    return self.alpha 

    def reward(self, state: int) -> float:
        return self.rewards[state]
    
    def value_iteration(self, threshold: float) -> Tuple[list, list]:

        # initialize arbitrarily
        valuefn: list = [0.0]*self.nstates
        policy: list = [0.0]*self.nstates

        while True:
            delta: float = 0.0
            #value iteration
            for state in range(self.nstates):
                oldvalue: float = valuefn[state]
                qs: list = [0.0, 0.0]
                for action in range(2):
                    for nxstate in range(self.nstates):
                        qs[action] += self.transition_prob(state, action, nxstate)*(self.reward(nxstate)
                                    + self.gamma*valuefn[nxstate])
                        
                if qs[1] >= qs[0]:
                    valuefn[state] = qs[1]
                    policy[state] = 1
                else:
                    valuefn[state] = qs[0]
                    policy[state] = 0
                delta = max(delta, abs(oldvalue - valuefn[state]))

            if delta < threshold:
                break


        return (valuefn, policy)


if __name__ == '__main__':
    rewards = [3.9, 2.0, 3.0, 4.0, 5.0, 6.0]
    alpha = 0.1
    gamma = 0.9

    h = JobHopper(rewards, alpha, gamma)
    valuefn, policy = h.value_iteration(0.01)

    print("----------------")
    print("Results for Reward Vector: ")
    pprint(rewards)
    print("Alpha, Gamma = ", alpha,  ", ", gamma)
    print("Optimal Value Function")
    pprint(dict(zip(["State {}:".format(x) for x in range(len(rewards))], 
                     valuefn)))
    print("Optimal Policy")
    pprint(dict(zip(["State {}:".format(x) for x in range(len(rewards))],
                    ["Accept" if x == 1 else "Reject" for x in policy])))
    print("----------------")

