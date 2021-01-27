import sys
sys.path.append('../')

from pprint import pprint
import assignment3.q3frogjumper as fj

if __name__ == '__main__':
    

    user_gamma = 0.9
    toppad = 9
    si_mdp: fj.FiniteMarkovDecisionProcess[fj.LilyPadState, int] =\
        fj.FrogJumper(
            highest_num_state = toppad
        )

    print("MDP Transition Map")
    print("------------------")
    print(si_mdp)

    print("MDP Policy Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_pi, opt_policy_pi = fj.policy_iteration_result(
        si_mdp,
        gamma=user_gamma
    )
    pprint(opt_vf_pi)
    print(opt_policy_pi)
    print()

    print("MDP Value Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_vi, opt_policy_vi = fj.value_iteration_result(si_mdp, gamma=user_gamma)
    pprint(opt_vf_vi)
    print(opt_policy_vi)
    print()