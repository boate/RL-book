import numpy as np
from rl.markov_process import TransitionStep
from typing import Sequence, Callable, TypeVar, Sequence, Dict
import numpy.linalg as la
import copy

S = TypeVar('S')
A = TypeVar('A')

class LSTD:
    def __init__(self, featurevec, gamma, transition_map, num_episodes):
        self.featurevec: Sequence[Callable[S, float]] = featurevec
        self.gamma: float = gamma
        self.dim: int = len(featurevec)
        self.A: np.ndarray = np.zeros((self.dim, self.dim))
        self.b: np.ndarray = np.zeros(self.dim)

    def get_weights(self):
        return(la.dot(la.inv(self.A), self.b))

    # update using transition step, which is a triplet of state, reward,
    # next state
    def update(self, data: Sequence[TransitionStep]):
        for item in data:
            state = item.state
            nxtstate = item.next_state
            reward = item.reward
            s1features = np.array([self.featurevec[i](state) for i in range(self.dim)])
            s2features = np.array([self.featurevec[i](nxtstate) for i in range(self.dim)])
            # update according to LSTD
            self.A += la.outer(s1features, s1features - self.gamma*s2features)
            self.b += reward*s1features

        print("New Weights")
        print(la.dot(la.inv(self.A), self.b))

class LSPI:
    def __init__(self, featurevec, gamma, states):
        self.featurevec: Sequence[Callable[(S,A), float]] = featurevec
        self.gamma: float = gamma
        self.dim: int = len(featurevec)
        self.A: np.ndarray = np.zeros((self.dim, self.dim))
        self.b: np.ndarray = np.zeros(self.dim)
        self.policy: Dict[S, A]
        self.states = states

    def get_weights(self):
        return(la.dot(la.inv(self.A), self.b))

    # update using transition step, which is a triplet of state, reward,
    # next state
    def update(self, data: Sequence[TransitionStep]):
        for item in data:
            # s, a, r, s' tuple
            state = item.state
            action = self.policy[state]
            nxtstate = item.next_state
            nxtaction = self.policy[nxtstate]
            reward = item.reward
            s1features = np.array([self.featurevec[i](state, action) for i in range(self.dim)])
            s2features = np.array([self.featurevec[i](nxtstate, nxtaction) for i in range(self.dim)])
            # update according to LSPI
            self.A += la.outer(s1features, s1features - self.gamma*s2features)
            self.b += reward*s1features

        print("New Weights")
        print(self.get_weights())


    def solve(self, data, iterations):
        while True:
            oldpolicy = copy.deepcopy(self.policy)
            #update the policy by maximizing over q values implied by weights
            weights = self.get_weights()
            for state in self.states:
                q = -999999999
                for action in self.actions[state]:
                    features = np.array([self.featurevec[i](state, action) for i in range(self.dim)])
                    newq = la.inner(features, weights)
                    if newq > q:
                        q = newq
                        self.policy[state] = action
            if self.policy == oldpolicy:
                return self.policy

            self.update(data)



