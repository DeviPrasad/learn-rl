import numpy as np

ROWS, COLS = 5, 5  # grid dimensions
GAMMA = 0.9  # discount factor
THETA = 0.001  # convergence threshold

SPECIAL_STATES = {
    (0, 1): ((4, 1), 10),  # A -> A', reward +10
    (0, 3): ((2, 3), 5),  # B -> B', reward +5
}

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ARROWS = ["↑", "↓", "←", "→"]


def transition(r: int, c: int, a: int) -> tuple[int, int, float]:
    if (r, c) in SPECIAL_STATES:
        (nr, nc), rew = SPECIAL_STATES[(r, c)]
        return (nr, nc, rew)
    dr, dc = ACTIONS[a]
    nr, nc = r + dr, c + dc
    if 0 <= nr < ROWS and 0 <= nc < COLS:
        return (nr, nc, 0.0)
    return (r, c, -1.0)  # off-grid -- stay, reward = -1


def value_iteration(gamma: float = GAMMA, theta: float = THETA):
    V = np.zeros((ROWS, COLS))

    for s in range(1, 100_001):
        delta = 0.0
        V_new = np.empty_like(V)

        for r in range(ROWS):
            for c in range(COLS):
                # Q(s, a) for each action; take the max (Bellman optimality)
                q_values = []
                for a in range(len(ACTIONS)):
                    nr, nc, rew = transition(r, c, a)
                    q_values.append(rew + gamma * V[nr, nc])

                V_new[r, c] = max(q_values)
                delta = max(delta, abs(V_new[r, c] - V[r, c]))

        V = V_new
        if delta < theta:
            break

    return V


# Extract greedy policy from V
# returns list-of-lists of optimal action indices.
def greedy_policy(V: np.ndarray, gamma: float = GAMMA):
    pi = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            q_values = []
            for a in range(len(ACTIONS)):
                nr, nc, rew = transition(r, c, a)
                q_values.append(rew + gamma * V[nr, nc])
            best_q = max(q_values)
            # collect all actions that tie for the best
            row.append([a for a, q in enumerate(q_values) if abs(q - best_q) < 1e-9])
        pi.append(row)
    return pi


def print_values(V: np.ndarray):
    print("v*(s)  — optimal state values")
    print("─" * (8 * COLS + 1))
    for r in range(ROWS):
        row_str = "|"
        for c in range(COLS):
            row_str += f" {V[r, c]:+6.2f} |"
        print(row_str)
    print("─" * (8 * COLS + 1))
    print()


def print_policy(pi: list):
    print("greedy optimal policy")
    print("─" * (8 * COLS + 1))
    for r in range(ROWS):
        row_str = "|"
        for c in range(COLS):
            arrows = "".join(ARROWS[a] for a in pi[r][c])
            row_str += f" {arrows:^6} |"
        print(row_str)
    print("─" * (8 * COLS + 1))


if __name__ == "__main__":
    V = value_iteration(gamma=GAMMA, theta=THETA)
    pi = greedy_policy(V, gamma=GAMMA)
    print_values(V)
    print_policy(pi)
