### MDP Value Iteration and Policy Iteration

import numpy as np
from riverswim import RiverSwim

np.set_printoptions(precision=3)

def bellman_backup(state, action, R, T, gamma, V):
    """
    Perform a single Bellman backup.

    Parameters
    ----------
    state: int
    action: int
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    V: np.array (num_states)

    Returns
    -------
    backup_val: float
    """
    backup_val = 0.
    ############################
    # YOUR IMPLEMENTATION HERE #
    backup_val = R[state, action] + gamma * np.array(T[state, action, :]).dot(V)
    ############################

    return backup_val

def policy_evaluation(policy, R, T, gamma, tol=1e-3):
    """
    Compute the value function induced by a given policy for the input MDP
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    value_function: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)

    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        new_value_function = np.zeros(num_states)

        for state in range(num_states):
            action = policy[state]
            new_value_function[state] = bellman_backup(
                state, action, R, T, gamma, value_function
            )

        delta = np.max(np.abs(new_value_function - value_function))
        value_function = new_value_function

        if delta < tol:
            break
    ############################
    return value_function


def policy_improvement(policy, R, T, V_policy, gamma):
    """
    Given the value function induced by a given policy, perform policy improvement
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    V_policy: np.array (num_states)
    gamma: float

    Returns
    -------
    new_policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    new_policy = np.zeros(num_states, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #
    for state, action in enumerate(policy):
        action_values = [
            bellman_backup(state, a, R, T, gamma, V_policy)
            for a in range(num_actions)
        ]

        best_action = np.argmax(action_values)

        if action_values[best_action] > action_values[action]:
            new_policy[state] = best_action
        else:
            new_policy[state] = action
    ############################
    return new_policy


def policy_iteration(R, T, gamma, tol=1e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    V_policy: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    V_policy = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        V_policy = policy_evaluation(policy, R, T, gamma, tol=tol)
        new_policy = policy_improvement(policy, R, T, V_policy, gamma)

        if np.array_equal(new_policy, policy):
            break

        policy = new_policy
    ############################
    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """Runs value iteration.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    value_function: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        action_values = np.zeros((num_states, num_actions))

        for state in range(num_states):
            for action in range(num_actions):
                action_values[state, action] = bellman_backup(
                    state, action, R, T, gamma, value_function
                )

        new_value_function = action_values.max(axis=1)
        delta = np.max(np.abs(new_value_function - value_function))
        value_function = new_value_function

        if delta < tol:
            break

    policy = policy_improvement(policy, R, T, value_function, gamma)
    ############################
    return value_function, policy


# Edit below to run policy and value iteration on different configurations
# You may change the parameters in the functions below
if __name__ == "__main__":
    SEED = 1234

    RIVER_CURRENT = 'WEAK'
    assert RIVER_CURRENT in ['WEAK', 'MEDIUM', 'STRONG']
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()
    discount_factor = 0.99

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, policy_pi = policy_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_pi)
    print([['L', 'R'][a] for a in policy_pi])

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_vi)
    print([['L', 'R'][a] for a in policy_vi])


# Q4(d): discount-factor sweep and initial-state values.
RUN_Q4D = True

def plot_q4d():
    from pathlib import Path
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    gammas = np.arange(100) / 100.0
    figure = Figure(figsize=(11, 6), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()

    for current, color in zip(('WEAK', 'MEDIUM', 'STRONG'),
                              ('#1877b8', '#e17c19', '#248f53')):
        R, T = RiverSwim(current).get_model()
        initial_values = []
        left_gammas_pi, left_gammas_vi = [], []
        for gamma in gammas:
            V_pi, policy_pi = policy_iteration(R, T, gamma, tol=1e-10)
            V_vi, policy_vi = value_iteration(R, T, gamma, tol=1e-10)
            assert np.allclose(V_pi, V_vi, rtol=0, atol=1e-8)
            initial_values.append(V_pi[0])
            if policy_pi[0] == 0:
                left_gammas_pi.append(gamma)
            if policy_vi[0] == 0:
                left_gammas_vi.append(gamma)

        assert left_gammas_pi == left_gammas_vi
        print(f'{current}: max gamma choosing LEFT at s1 = '
              f'{max(left_gammas_pi):.2f}; '
              f'V*(s1) at gamma=0.99 = {initial_values[-1]:.6f}')
        axis.plot(gammas, initial_values, color=color, label=current)

    axis.set(title='RiverSwim: optimal initial-state value vs. discount factor',
             xlabel=r'Discount factor $\gamma$',
             ylabel=r'Optimal initial-state value $V^*(s_1)$ (log scale)',
             xlim=(0, 0.99), yscale='log')
    axis.grid(alpha=0.25, which='both')
    axis.legend(title='River current')
    code_dir = Path(__file__).resolve().parent
    latex_dir = code_dir.parent / 'latex'
    image_dir = latex_dir if latex_dir.is_dir() else code_dir
    image_path = image_dir / 'riverswim_initial_state_value.png'
    figure.savefig(image_path, dpi=180)
    print(f'Figure saved: {image_path}')
    return image_path


if __name__ == '__main__' and RUN_Q4D:
    plot_q4d()
