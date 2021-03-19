from typing import Sequence, Tuple, Mapping, Dict
import numpy as np
import random
S = str
DataType = Sequence[Sequence[Tuple[S, float]]]
ProbFunc = Mapping[S, Mapping[S, float]]
RewardFunc = Mapping[S, float]
ValueFunc = Mapping[S, float]

""" COMMENTS:
    LSTD gives the same value function as MRP since our features are just the identity
    for each indicator for being in a certain state. TD also does well because it updates
    after seeing each sample incrementally so it can reuse samples and is more efficient.
    MC is pretty off either because I implemented it incorrectly or just because of lack of
    efficiency with samples.
"""

def get_state_return_samples(
    data: DataType
) -> Sequence[Tuple[S, float]]:
    """
    prepare sequence of (state, return) pairs.
    Note: (state, return) pairs is not same as (state, reward) pairs.
    """
    return [(s, sum(r for (_, r) in l[i:]))
            for l in data for i, (s, _) in enumerate(l)]


def get_mc_value_function(
    state_return_samples: Sequence[Tuple[S, float]]
) -> ValueFunc:
    """
    Implement tabular MC Value Function compatible with the interface defined above.
    """
    value_func = {s[0]: 0 for s in state_return_samples}
    counts = {s[0]: 0 for s in state_return_samples}
    for sample in state_return_samples:
        counts[sample[0]] += 1
        weight = 1/counts[sample[0]]
        value_func[sample[0]] = (1-weight)*value_func[sample[0]] + weight*sample[1]
    return value_func


def get_state_reward_next_state_samples(
    data: DataType
) -> Sequence[Tuple[S, float, S]]:
    """
    prepare sequence of (state, reward, next_state) triples.
    """
    return [(s, r, l[i+1][0] if i < len(l) - 1 else 'T')
            for l in data for i, (s, r) in enumerate(l)]


def get_probability_and_reward_functions(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> Tuple[ProbFunc, RewardFunc]:
    """
    Implement code that produces the probability transitions and the
    reward function compatible with the interface defined above.
    """
    # from samples just use MLE estimate of rewards??
    totalcounts = {}
    individualcounts = {}
    for sample in srs_samples:
        if sample[0] not in totalcounts:
            totalcounts[sample[0]] = 1
        else:
            totalcounts[sample[0]] += 1

        if sample[0] not in individualcounts:
            individualcounts[sample[0]] = {sample[2]: 1}
        # add to indvidual count
        else:
            if sample[2] not in individualcounts[sample[0]]:
                individualcounts[sample[0]][sample[2]] = 1
            else:
                individualcounts[sample[0]][sample[2]] += 1

    # normalize to probabilities
    for state in totalcounts:
        for nxtstate in individualcounts[state]:
            individualcounts[state][nxtstate] /= totalcounts[state]

    # get rewards
    rewards = {}
    for sample in srs_samples:
        if sample[0] not in rewards:
            rewards[sample[0]] = [sample[1]]
        else:
            rewards[sample[0]].append(sample[1])
    rewardfunc = {s: np.mean(rewards[s]) for s in list(rewards.keys())}

    return (individualcounts, rewardfunc)

def get_mrp_value_function(
    prob_func: ProbFunc,
    reward_func: RewardFunc
) -> ValueFunc:
    """
    Implement code that calculates the MRP Value Function from the probability
    transitions and reward function, compatible with the interface defined above.
    Hint: Use the MRP Bellman Equation and simple linear algebra
    """
    states = list(reward_func.keys())
    rewards = np.array([reward_func[s] for s in states])
    prob = []
    for state in states:
        curr = []
        for nxtstate in states:
            if nxtstate in prob_func[state]:
                curr.append(prob_func[state][nxtstate])
            else:
                curr.append(0)
        prob.append(curr)

    # same as problem 2 of midterm
    value_func = np.linalg.inv(np.eye(len(states)) - prob).dot(rewards)
    return {states[i]: value_func[i] for i in range(len(states))}

def get_td_value_function(
    srs_samples: Sequence[Tuple[S, float, S]],
    num_updates: int = 300000,
    learning_rate: float = 0.3,
    learning_rate_decay: int = 30
) -> ValueFunc:
    """
    Implement tabular TD(0) (with experience replay) Value Function compatible
    with the interface defined above. Let the step size (alpha) be:
    learning_rate * (updates / learning_rate_decay + 1) ** -0.5
    so that Robbins-Monro condition is satisfied for the sequence of step sizes.
    """
    probfunc, rewardfunc = get_probability_and_reward_functions(srs_samples)
    value_func = {s: 0 for s in list(probfunc.keys())}
    for update in range(num_updates):
        alpha = learning_rate * (update / learning_rate_decay + 1) ** -0.5
        state, reward, nxtstate = random.sample(srs_samples, 1)[0]
        if nxtstate != 'T':
            value_func[state] += alpha * (reward + value_func[nxtstate] - value_func[state])
        else:
            value_func[state] += alpha * (reward - value_func[state])

    return value_func


def get_lstd_value_function(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> ValueFunc:
    """
    Implement LSTD Value Function compatible with the interface defined above.
    Hint: Tabular is a special case of linear function approx where each feature
    is an indicator variables for a corresponding state and each parameter is
    the value function for the corresponding state.
    """
    # get unique set of states
    states = list(set(sample[0] for sample in srs_samples))
    dim = len(states)

    # preliminaries
    feat = np.eye(dim)
    A = np.zeros((dim, dim))
    b = np.zeros(dim)

    for sample in srs_samples:
        state = sample[0]
        statenum = states.index(state)
        ret = sample[1]
        nxtstate = sample[2]
        # check if nonterminal
        if nxtstate != 'T':
            nxtstatenum = states.index(nxtstate)
            p = feat[nxtstatenum]
        else:
            p = np.zeros(dim)

        # updating step
        A += np.outer(feat[statenum], feat[statenum] - p)
        b += feat[states.index(state)] * ret

    # create value func
    weights = np.linalg.inv(A).dot(b)
    value_func = {}
    for i in range(dim):
        value_func[states[i]] = weights[i]

    return value_func

if __name__ == '__main__':
    given_data: DataType = [
        [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
        [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
        [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
        [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
        [('B', 8.), ('B', 2.)]
    ]

    sr_samps = get_state_return_samples(given_data)

    print("------------- MONTE CARLO VALUE FUNCTION --------------")
    print(get_mc_value_function(sr_samps))

    srs_samps = get_state_reward_next_state_samples(given_data)
    print(get_probability_and_reward_functions(srs_samps))
    pfunc, rfunc = get_probability_and_reward_functions(srs_samps)
    print("-------------- MRP VALUE FUNCTION ----------")
    print(get_mrp_value_function(pfunc, rfunc))

    print("------------- TD VALUE FUNCTION --------------")
    print(get_td_value_function(srs_samps))

    print("------------- LSTD VALUE FUNCTION --------------")
    print(get_lstd_value_function(srs_samps))
