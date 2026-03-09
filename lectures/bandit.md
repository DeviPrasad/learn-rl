"""
10-Armed Bandit Testbed.
"Reinforcement Learning: An Introduction". (2nd Edition).
Chapter 2: Multi-armed Bandits.
An attempt to reproduces the scenario described in Figure 2.2 from the book.
"""

import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, k=10, mean=0.0, std=1.0):
        self.k = k
        # True action values q*(a) ~ N(mean, std^2)
        self.q_true = np.random.normal(mean, std, k)
        # Track the optimal action
        self.optimal_action = np.argmax(self.q_true)

    def pull(self, action):
        # Reward is drawn from N(q*(action), 1)
        reward = np.random.normal(self.q_true[action], 1.0)
        return reward

    def is_optimal(self, action):
        return action == self.optimal_action


class Agent:
    """
    An agent that learns action values using sample averaging or constant step size.

    Parameters:
    -----------
    k : int
        Number of actions
    epsilon : float
        Probability of exploration (epsilon-greedy)
    initial_value : float
        Initial Q-value estimates
    alpha : float or None
        Step size parameter. If None, uses sample averaging (1/n)
    """

    def __init__(self, k=10, epsilon=0.0, initial_value=0.0, alpha=None):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha  # If None, use sample averaging

        # Q(a): estimated action values
        self.Q = np.full(k, initial_value, dtype=float)

        # N(a): number of times each action has been selected
        self.N = np.zeros(k, dtype=int)

    def select_action(self):
        """
        Select an action using epsilon-greedy policy.

        Returns:
        --------
        action : int
            The selected action
        """
        if np.random.random() < self.epsilon:
            # Explore: choose randomly
            return np.random.randint(self.k)
        else:
            # Exploit: choose greedily (break ties randomly)
            max_value = np.max(self.Q)
            best_actions = np.where(self.Q == max_value)[0]
            return np.random.choice(best_actions)

    def update(self, action, reward):
        self.N[action] += 1

        if self.alpha is None:
            # Sample averaging: alpha = 1/n
            step_size = 1.0 / self.N[action]
        else:
            # Constant step size
            step_size = self.alpha

        # Incremental update rule: Q(a) <- Q(a) + alpha * [R - Q(a)]
        self.Q[action] += step_size * (reward - self.Q[action])

    def reset(self, initial_value=0.0):
        self.Q = np.full(self.k, initial_value, dtype=float)
        self.N = np.zeros(self.k, dtype=int)


def run_experiment(agent, bandit, steps=1000):
    rewards = np.zeros(steps)
    optimal_actions = np.zeros(steps)

    for t in range(steps):
        # Select action
        action = agent.select_action()

        # Get reward
        reward = bandit.pull(action)

        # Update agent
        agent.update(action, reward)

        # Record results
        rewards[t] = reward
        optimal_actions[t] = 1 if bandit.is_optimal(action) else 0

    return rewards, optimal_actions


def run_testbed(n_bandits=2000, steps=1000, epsilon_values=[0.0, 0.01, 0.1]):
    """
    Run the complete testbed experiment as described in Sutton & Barto Figure 2.2.

    Parameters:
    -----------
    n_bandits : int
        Number of bandit problems to average over
    steps : int
        Number of time steps per run
    epsilon_values : list
        List of epsilon values to test

    Returns:
    --------
    results : dict
        Dictionary containing average rewards and optimal action percentages
    """
    results = {}

    for epsilon in epsilon_values:
        print(f"Running epsilon = {epsilon}...")

        all_rewards = np.zeros((n_bandits, steps))
        all_optimal = np.zeros((n_bandits, steps))

        for i in range(n_bandits):
            # Create a new bandit problem
            bandit = Bandit(k=10, mean=0.0, std=1.0)

            # Create agent with this epsilon value
            agent = Agent(k=10, epsilon=epsilon, initial_value=0.0, alpha=None)

            # Run experiment
            rewards, optimal_actions = run_experiment(agent, bandit, steps)

            all_rewards[i] = rewards
            all_optimal[i] = optimal_actions

        # Average across all bandit problems
        results[epsilon] = {
            'avg_reward': np.mean(all_rewards, axis=0),
            'pct_optimal': np.mean(all_optimal, axis=0) * 100
        }

    return results


def plot_results(results, steps=1000):
    """
    Plot the results similar to Figure 2.2 in Sutton & Barto.

    Parameters:
    -----------
    results : dict
        Results from run_testbed
    steps : int
        Number of steps
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    for epsilon, data in results.items():
        label = f'ε = {epsilon}' if epsilon > 0 else 'greedy (ε = 0)'

        # Plot average reward
        ax1.plot(data['avg_reward'], label=label)

        # Plot % optimal action
        ax2.plot(data['pct_optimal'], label=label)

    # Format average reward plot
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.set_xlabel('Steps', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('10-Armed Testbed Results (Average over 2000 runs)', fontsize=14, fontweight='bold')

    # Format optimal action plot
    ax2.set_ylabel('% Optimal Action', fontsize=12)
    ax2.set_xlabel('Steps', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    print("=" * 60)
    print("10-Armed Bandit Testbed")
    print("Sutton & Barto, Chapter 2")
    print("=" * 60)
    print()

    # Run the testbed with different epsilon values
    results = run_testbed(
        n_bandits=2000,  # Number of different bandit problems
        steps=1000,       # Number of steps per problem
        epsilon_values=[0.0, 0.01, 0.1]  # Exploration rates to test
    )

    # Plot results
    plot_results(results)

    # Print final statistics
    print("\n" + "=" * 60)
    print("Final Results (at step 1000):")
    print("=" * 60)
    for epsilon, data in results.items():
        print(f"\nε = {epsilon}:")
        print(f"  Average Reward: {data['avg_reward'][-1]:.3f}")
        print(f"  % Optimal Action: {data['pct_optimal'][-1]:.1f}%")