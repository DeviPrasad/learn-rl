"""
Sutton & Barto — Example 3.5: Gridworld
Iterative Policy Evaluation.

Rewards:
  Leave A   -> +10, goto A'
  Leave B   ->  +5, goto B'
  Off-grid  ->  -1, stay in the same cell
  otherwise ->   0
"""

import numpy as np

# ── Grid parameters ────────────────────────────────────────────────────────────
ROWS, COLS = 5, 5
GAMMA = 0.9  # discount factor
THETA = 0.001  # convergence threshold


# action space
ACTIONS = {  # direction vectors for N, S, E, W
    "N": (-1, 0),
    "W": (0, -1),
    "S": (1, 0),
    "E": (0, 1),
}

A = (0, 1)
A_PRIME = (4, 1)
B = (0, 3)
B_PRIME = (2, 3)

VALUES = []


def show_values(iteration, values):
    if iteration is not None:
        print()
        print(f"Iteration {iteration}")
    print("=" * 67)
    print(f"  {'Col0':>13} {'Col1':>10} {'Col2':>12} {'Col3':>10} {'Col4':>12}")
    print("-" * 67)
    for r in range(ROWS):
        row_str = f"R{r} "
        for c in range(COLS):
            label = ""
            if (r, c) == A:
                label = "A"
            elif (r, c) == A_PRIME:
                label = "A'"
            elif (r, c) == B:
                label = "B"
            elif (r, c) == B_PRIME:
                label = "B'"
            cell = f"{values[r,c]:5.2f}"
            if label:
                cell = f"{cell}[{label}]"
            row_str += f"{cell:>12}"
        print(row_str)
    print("=" * 67)


# transition model - gives next state and reward for a given state and action
def step(r, c, action):
    # special transitions for A and B
    if (r, c) == A:
        return (*A_PRIME, 10.0)
    if (r, c) == B:
        return (*B_PRIME, 5.0)

    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc

    if 0 <= nr < ROWS and 0 <= nc < COLS:
        return (nr, nc, 0.0)

    # Off-grid: stay in the same cell, reward -1
    return (r, c, -1.0)


# iterative policy evaluation
def policy_evaluation():
    V = np.zeros((ROWS, COLS))  # initial values

    _iteration = 0
    while True:
        delta = 0
        V_new = np.zeros_like(V)

        _iteration += 1
        if _iteration <= 3:
            print(f"Iteration {_iteration}")
        for r in range(ROWS):
            for c in range(COLS):
                v = 0.0
                for action in ACTIONS:
                    # random policy: an action's probability = 0.25
                    nr, nc, reward = step(r, c, action)
                    v += 0.25 * (reward + GAMMA * V[nr, nc])
                    # print details for the first state in the second iteration
                    if _iteration <= 3 and r == 1 and c == 1:
                        print(
                            f"\tState ({r},{c}), Action '{action}': "
                            f"Next State ({nr},{nc}), Reward {reward:.1f}, CurVal {V[nr, nc]:.4f}, "
                            f"Update = {v:.4f}"
                        )
                #
                V_new[r, c] = v
                delta = max(delta, abs(v - V[r, c]))
        #
        V = V_new

        VALUES.append(V.copy())

        if delta < THETA:
            break
    #
    return V


if __name__ == "__main__":
    V = policy_evaluation()

    for n, V in enumerate(VALUES):
        show_values(n + 1, V)
