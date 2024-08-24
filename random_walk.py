import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

# Random Walk Environment
class Env:
    def __init__(self, n):
        self.nStates = n
        self.state = n // 2 + 1

    def reset(self):
        self.state = self.nStates // 2 + 1
        return self.state

    def step(self):
        new_state = np.random.choice([self.state - 1, self.state + 1])
        reward = 0
        if new_state == 0:
            done = True
        elif new_state == self.nStates + 1:
            reward = 1
            done = True
        else:
            done = False
        self.state = new_state
        return self.state, reward, done

# Helper function to generate primes
def generate_primes(n):
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = all(candidate % p != 0 for p in primes)
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes

# Helper function to generate true values based on the number of states for the random walk problem
def generate_true_values(num_states):
    return np.array([(i + 1) / (num_states + 1) for i in range(num_states)])

# Monte Carlo with a constant alpha 
def mc_constant_alpha(num_steps, num_states, alpha_values, gamma=1.0):
    if isinstance(alpha_values, (int, float)):
        alpha_values = [alpha_values]

    true_values = generate_true_values(num_states)
    initial_value = 0.0
    num_runs = 100
    
    rms_errors_all = np.zeros((len(alpha_values), num_steps))
    final_values = np.zeros((len(alpha_values), num_runs, num_states + 2))
    for i, alpha in enumerate(alpha_values):
        for run in range(num_runs):
            V = np.full(num_states + 2, initial_value, dtype=float)
            env = Env(num_states)
            for _ in range(num_steps):
                state = env.reset()
                trajectory = [state]
                rewards = []
                while True:
                    new_state, reward, done = env.step()
                    trajectory.append(new_state)
                    rewards.append(reward)
                    if done:
                        break
                    state = new_state
                
                G = 0
                for t in reversed(range(len(trajectory) - 1)):
                    state = trajectory[t]
                    G = gamma * G + rewards[t]
                    V[state] += alpha * (G - V[state])
                
                rms_error = np.sqrt(np.mean((V[1:num_states + 1] - true_values) ** 2))
                rms_errors_all[i, _] += rms_error
            final_values[i, run] = V
    rms_errors_all /= num_runs
    avg_final_values = np.mean(final_values, axis=1)
    return rms_errors_all, avg_final_values

# Monte Carlo with harmonic sequence of alphas
def mc_harmonic(num_steps, num_states, gamma=1.0):
    true_values = generate_true_values(num_states)
    initial_value = 0.0
    num_runs = 100
    
    rms_errors_all = np.zeros(num_steps)
    final_values = np.zeros((num_runs, num_states + 2))
    for run in range(num_runs):
        V = np.full(num_states + 2, initial_value, dtype=float)
        state_visit_counts = np.zeros(num_states + 2)
        env = Env(num_states)
        for _ in range(num_steps):
            state = env.reset()
            trajectory = [state]
            rewards = []
            while True:
                new_state, reward, done = env.step()
                trajectory.append(new_state)
                rewards.append(reward)
                if done:
                    break
                state = new_state
            
            G = 0.0
            for t in reversed(range(len(trajectory) - 1)):
                state = trajectory[t]
                G = gamma * G + rewards[t]
                state_visit_counts[state] += 1
                alpha = 1 / state_visit_counts[state]
                V[state] += alpha * (G - V[state])
                
                # Debugging output to trace values
                if run == 0 and _ < 10:  # Limiting output for readability
                    print(f"Time step {_}, State {state}, Visit Count {state_visit_counts[state]}, Alpha {alpha:.4f}, V[{state}] {V[state]:.4f}")
            
            rms_error = np.sqrt(np.mean((V[1:num_states + 1] - true_values) ** 2))
            rms_errors_all[_] += rms_error
        final_values[run] = V
    rms_errors_all /= num_runs
    avg_final_values = np.mean(final_values, axis=0)
    return rms_errors_all, avg_final_values

# Monte Carlo with prime sequence of alphas
def mc_prime(num_steps, num_states, gamma=1.0):
    true_values = generate_true_values(num_states)
    initial_value = 0.0
    num_runs = 100
    
    max_visits = num_steps * num_states  # This ensures we have enough primes
    primes = generate_primes(max_visits)
    
    rms_errors_all = np.zeros(num_steps)
    final_values = np.zeros((num_runs, num_states + 2))
    for run in range(num_runs):
        V = np.full(num_states + 2, initial_value, dtype=float)
        state_visit_counts = np.zeros(num_states + 2)
        env = Env(num_states)
        for _ in range(num_steps):
            state = env.reset()
            trajectory = [state]
            rewards = []
            while True:
                new_state, reward, done = env.step()
                trajectory.append(new_state)
                rewards.append(reward)
                if done:
                    break
                state = new_state
            
            G = 0
            for t in reversed(range(len(trajectory) - 1)):
                state = trajectory[t]
                G = gamma * G + rewards[t]
                state_visit_counts[state] += 1
                visit_index = int(state_visit_counts[state])
                if visit_index > max_visits:
                    max_visits = visit_index
                    primes.extend(generate_primes(visit_index + 100))  # Ensure we have enough primes
                current_alpha = 1 / primes[visit_index - 1]
                V[state] += current_alpha * (G - V[state])
            
            rms_error = np.sqrt(np.mean((V[1:num_states + 1] - true_values) ** 2))
            rms_errors_all[_] += rms_error
        final_values[run] = V
    rms_errors_all /= num_runs
    avg_final_values = np.mean(final_values, axis=0)
    return rms_errors_all, avg_final_values

# TD(N) with a constant alpha 
def td_n(alpha_values, num_steps, num_states, n_values, gamma=1.0):
    true_values = generate_true_values(num_states)
    initial_value = 0.0
    num_runs = 100

    rms_errors_all = np.zeros((len(n_values), len(alpha_values), num_steps))
    final_values = np.zeros((len(n_values), len(alpha_values), num_states + 2))
    for run in range(num_runs):
        for j, n in enumerate(n_values):
            for i, alpha in enumerate(alpha_values):
                V = np.full(num_states + 2, initial_value, dtype=float)
                env = Env(num_states)
                for step in range(num_steps):
                    state = env.reset()
                    trajectory = []
                    rewards = []
                    t = 0
                    while t < n + 1:
                        new_state, reward, done = env.step()
                        trajectory.append(state)
                        rewards.append(reward)
                        if done:
                            break
                        state = new_state
                        t += 1
                    G = sum([gamma**k * rewards[k] for k in range(len(rewards))])
                    for t in range(len(trajectory)):
                        state = trajectory[t]
                        V[state] += alpha * (G - V[state])
                        G = (G - rewards[t]) / gamma
                    rms_error = np.sqrt(np.mean((V[1:num_states + 1] - true_values) ** 2))
                    rms_errors_all[j, i, step] += rms_error
                final_values[j, i, :] += V
    rms_errors_all /= num_runs
    final_values = np.mean(final_values, axis=1)  # Average the final values over all runs
    return rms_errors_all, final_values

# TD(N) with harmonic sequence of alphas
def td_n_harmonic(steps, num_states, n_values, gamma=1.0):
    true_values = generate_true_values(num_states)
    initial_value = 0.0  # Initialize all values to 0.0 (float)
    num_runs = 100
    
    rms_errors_all = np.zeros((len(n_values), steps))
    final_values = np.zeros((len(n_values), num_runs, num_states + 2))  # Include terminal states
    for run in range(num_runs):
        for j, n in enumerate(n_values):
            V = np.full(num_states + 2, initial_value, dtype=float)  # Include terminal states
            env = Env(num_states)
            state_visit_counts = np.zeros(num_states + 2)  # Track visit counts for each state
            for step in range(steps):
                state = env.reset()
                trajectory = [state]
                eligibility = np.zeros(num_states + 2, dtype=float)
                while True:
                    new_state, reward, done = env.step()
                    trajectory.append(new_state)
                    eligibility *= gamma
                    eligibility[state] += 1.0 
                    td_error = reward + gamma * V[new_state] - V[state]
                    state_visit_counts[state] += 1
                    current_alpha = 1 / state_visit_counts[state]  # Harmonic sequence for alpha
                    V += current_alpha * td_error * eligibility
                    if len(trajectory) == n + 2:
                        eligibility[trajectory[0]] -= gamma ** n
                        trajectory.pop(0)
                    if done:
                        break
                    state = new_state
                rms_error = np.sqrt(np.mean((V[1:num_states + 1] - true_values) ** 2))
                rms_errors_all[j, step] += rms_error  # Store RMS error for each episode
            final_values[j, run] = V  # Store the final values for the current run and n
    rms_errors_all /= num_runs
    avg_final_values = np.mean(final_values, axis=1)  # Average the final values over all runs
    return rms_errors_all, avg_final_values

# TD(N) with a prime sequence of alphas
def td_n_prime(steps, num_states, n_values, gamma=1.0):
    true_values = generate_true_values(num_states)
    initial_value = 0.0  # Initialize all values to 0.0 (float)
    num_runs = 100
    
    primes = generate_primes(steps)
    
    rms_errors_all = np.zeros((len(n_values), steps))
    final_values = np.zeros((len(n_values), num_runs, num_states + 2))  # Include terminal states
    for run in range(num_runs):
        for j, n in enumerate(n_values):
            V = np.full(num_states + 2, initial_value, dtype=float)  # Include terminal states
            env = Env(num_states)
            state_visit_counts = np.zeros(num_states + 2)  # Track visit counts for each state
            for step in range(steps):
                state = env.reset()
                trajectory = [state]
                eligibility = np.zeros(num_states + 2, dtype=float)
                while True:
                    new_state, reward, done = env.step()
                    trajectory.append(new_state)
                    eligibility *= gamma
                    eligibility[state] += 1.0 
                    td_error = reward + gamma * V[new_state] - V[state]
                    state_visit_counts[state] += 1
                    i = state_visit_counts[state]
                    if i <= len(primes):
                        current_alpha = 1 / (primes[int(i) - 1] ** 2)  # Harmonic sequence of primes
                    else:
                        current_alpha = 0  # If we run out of primes, stop updating
                    V += current_alpha * td_error * eligibility
                    if len(trajectory) == n + 2:
                        eligibility[trajectory[0]] -= gamma ** n
                        trajectory.pop(0)
                    if done:
                        break
                    state = new_state
                rms_error = np.sqrt(np.mean((V[1:num_states + 1] - true_values) ** 2))
                rms_errors_all[j, step] += rms_error  # Store RMS error for each episode
            final_values[j, run] = V  # Store the final values for the current run and n
    rms_errors_all /= num_runs
    avg_final_values = np.mean(final_values, axis=1)  # Average the final values over all runs
    return rms_errors_all, avg_final_values

# TD(lambda) with a constant alpha 
def td_lambda(alpha_values, num_steps, num_states, lambda_values, gamma=1.0):
    true_values = generate_true_values(num_states)
    initial_value = 0.0
    num_runs = 100

    # Adjust final_values to match the expected 3D shape
    rms_errors_all = np.zeros((len(lambda_values), len(alpha_values), num_steps))
    final_values = np.zeros((len(lambda_values), len(alpha_values), num_runs, num_states + 2))

    for run in range(num_runs):
        for j, lam in enumerate(lambda_values):
            for i, alpha in enumerate(alpha_values):
                V = np.full(num_states + 2, initial_value, dtype=float)
                env = Env(num_states)
                for step in range(num_steps):
                    state = env.reset()
                    eligibility = np.zeros(num_states + 2, dtype=float)
                    while True:
                        new_state, reward, done = env.step()
                        td_error = reward + gamma * V[new_state] - V[state]
                        eligibility[state] += 1.0
                        V += alpha * td_error * eligibility
                        eligibility *= gamma * lam
                        if done:
                            break
                        state = new_state
                    rms_error = np.sqrt(np.mean((V[1:num_states + 1] - true_values) ** 2))
                    rms_errors_all[j, i, step] += rms_error
                final_values[j, i, run] = V  # Store the final values per run
    rms_errors_all /= num_runs
    avg_final_values = np.mean(final_values, axis=1)  # Average over runs

    print(f'Debug: rms_errors_all shape: {rms_errors_all.shape}')  # Debug statement
    return rms_errors_all, avg_final_values

# def td_lambda(alpha_values, num_steps, num_states, lambda_values, gamma=1.0):
#     true_values = generate_true_values(num_states)
#     initial_value = 0.0
#     num_runs = 100

#     rms_errors_all = np.zeros((len(lambda_values), len(alpha_values)))
#     final_values = np.zeros((len(lambda_values), num_runs, num_states + 2))  # Ensure this shape matches V
#     for run in range(num_runs):
#         for j, lam in enumerate(lambda_values):
#             for i, alpha in enumerate(alpha_values):
#                 V = np.full(num_states + 2, initial_value, dtype=float)
#                 env = Env(num_states)
#                 for _ in range(num_steps):
#                     state = env.reset()
#                     eligibility = np.zeros(num_states + 2, dtype=float)
#                     while True:
#                         new_state, reward, done = env.step()
#                         eligibility *= gamma * lam
#                         eligibility[state] += 1.0
#                         td_error = reward + gamma * V[new_state] - V[state]
#                         V[state] += alpha * td_error * eligibility[state]
#                         if done:
#                             break
#                         state = new_state
#                 rms_error = np.sqrt(np.mean((V[1:num_states + 1] - true_values) ** 2))
#                 rms_errors_all[j, i] += rms_error
#             final_values[j, run] = V 
#     rms_errors_all /= num_runs
#     avg_final_values = np.mean(final_values, axis=1)
#     return rms_errors_all, avg_final_values

# TD(lambda) with harmonic sequence of alphas
def td_lambda_harmonic(num_states, lambda_values, steps, gamma=1.0):
    true_values = generate_true_values(num_states)
    initial_value = 0.0  # Initialize all values to 0.0 (float)
    num_runs = 100
    
    rms_errors_all = np.zeros((len(lambda_values), steps))
    final_values = np.zeros((len(lambda_values), num_runs, num_states + 2))  # Include terminal states
    for run in range(num_runs):
        for j, lam in enumerate(lambda_values):
            V = np.full(num_states + 2, initial_value, dtype=float)  # Include terminal states
            env = Env(num_states)
            state_visit_counts = np.zeros(num_states + 2)  # Track visit counts for each state
            for step in range(steps):
                state = env.reset()
                eligibility = np.zeros(num_states + 2, dtype=float)
                while True:
                    new_state, reward, done = env.step()
                    eligibility *= gamma * lam
                    eligibility[state] += 1.0 # ACCUMULATING TRACE
                    state_visit_counts[state] += 1
                    current_alpha = 1 / state_visit_counts[state]  # Harmonic sequence for alpha
                    td_error = reward + gamma * V[new_state] - V[state]
                    V += current_alpha * td_error * eligibility
                    if done:
                        break
                    state = new_state
                rms_error = np.sqrt(np.mean((V[1:num_states + 1] - true_values) ** 2))
                rms_errors_all[j, step] += rms_error  # Store RMS error for each episode
            final_values[j, run] = V # Store the final values for the current run and lambda
    rms_errors_all /= num_runs
    avg_final_values = np.mean(final_values, axis=1)  # Average the final values over all runs
    return rms_errors_all, avg_final_values

# TD(lambda) with prime sequence of alphas
def td_lambda_prime(num_states, lambda_values, steps, gamma=1.0):
    true_values = generate_true_values(num_states)
    initial_value = 0.0  # Initialize all values to 0.0 (float)
    num_runs = 100
    
    rms_errors_all = np.zeros((len(lambda_values), steps))
    final_values = np.zeros((len(lambda_values), num_runs, num_states + 2))  # Include terminal states
    primes = generate_primes(steps)  # Generate prime numbers
    for run in range(num_runs):
        for j, lam in enumerate(lambda_values):
            V = np.full(num_states + 2, initial_value, dtype=float)  # Include terminal states
            env = Env(num_states)
            state_visit_counts = np.zeros(num_states + 2)  # Track visit counts for each state
            for step in range(steps):
                state = env.reset()
                eligibility = np.zeros(num_states + 2, dtype=float)
                while True:
                    new_state, reward, done = env.step()
                    eligibility *= gamma * lam
                    eligibility[state] += 1.0 # ACCUMULATING TRACE
                    state_visit_counts[state] += 1
                    if state_visit_counts[state] <= len(primes):
                        current_alpha = 1 / (primes[int(state_visit_counts[state]) - 1] ** 2)
                    else:
                        current_alpha = 1 / (primes[-1] ** 2)
                    td_error = reward + gamma * V[new_state] - V[state]
                    V += current_alpha * td_error * eligibility
                    if done:
                        break
                    state = new_state
                rms_error = np.sqrt(np.mean((V[1:num_states + 1] - true_values) ** 2))
                rms_errors_all[j, step] += rms_error  # Store RMS error for each episode
            final_values[j, run] = V # Store the final values for the current run and lambda
    rms_errors_all /= num_runs
    avg_final_values = np.mean(final_values, axis=1)  # Average the final values over all runs
    return rms_errors_all, avg_final_values

## PLOTS

# Function to plot true vs estimated values
def plot_true_vs_estimated(true_values, estimated_values, method_name, alpha=None, n=None, lam=None):
    states = np.arange(1, len(true_values) + 1)
    
    # Ensure that estimated_values is correctly flattened if it has an extra dimension
    if estimated_values.ndim > 1:
        estimated_values = estimated_values.flatten()

    plt.figure(figsize=(10, 6))
    plt.bar(states - 0.2, true_values, width=0.4, label='True Values', color='blue')
    plt.bar(states + 0.2, estimated_values, width=0.4, label='Estimated Values', color='orange')
    
    plt.xlabel('State')
    plt.ylabel('Value')
    
    title = f'True vs Estimated Values for {method_name}'
    if alpha is not None:
        title += f' (alpha={alpha})'
    if n is not None:
        title += f' (n={n})'
    if lam is not None:
        title += f' (lambda={lam})'
    
    plt.title(title)
    plt.xticks(states)
    plt.legend()
    plt.grid(True)
    plt.show()

# PLOT WITH MC OVERLAID 
def plot_with_overlay(method, num_states, alpha_values, n_values=None, lambda_values=None, steps=100, harmonic_alpha=False, prime_sequence=False, overlay=True):
    # Ensure num_states is an integer
    if isinstance(num_states, list):
        if len(num_states) == 1:
            num_states = num_states[0]
        else:
            raise ValueError("num_states should be a single integer value.")

    true_values = generate_true_values(num_states)

    # Determine the appropriate MC method based on the alpha type
    if overlay:
        if harmonic_alpha:
            rms_errors_mc, _ = mc_harmonic(steps, num_states)
            mc_label = 'MC Harmonic Sequence'
        elif prime_sequence:
            rms_errors_mc, _ = mc_prime(steps, num_states)
            mc_label = 'MC Prime Sequence'
        else:
            rms_errors_mc, _ = mc_constant_alpha(steps, num_states, alpha_values)
            mc_label = None  # We'll handle labels within the loop for constant alpha

        # Standardize MC results to 2D for plotting
        rms_errors_mc = rms_errors_mc.reshape((1, steps)) if rms_errors_mc.ndim == 1 else rms_errors_mc
    else:
        rms_errors_mc = None

    # Plot TD(n) results
    if method == 'tdn' and n_values is not None:
        if harmonic_alpha:
            rms_errors_all, _ = td_n_harmonic(steps, num_states, n_values)
            label_base = 'TD(n) Harmonic Sequence'
            for i, n in enumerate(n_values):
                plt.plot(range(steps), rms_errors_all[i], label=f'{label_base} n={n}, alpha=Harmonic')
        elif prime_sequence:
            rms_errors_all, _ = td_n_prime(steps, num_states, n_values)
            label_base = 'TD(n) Prime Sequence'
            for i, n in enumerate(n_values):
                plt.plot(range(steps), rms_errors_all[i], label=f'{label_base} n={n}, alpha=Prime')
        else:
            rms_errors_all, _ = td_n(alpha_values, steps, num_states, n_values)
            label_base = 'TD(n) Constant Alpha'
            for i, n in enumerate(n_values):
                for j, alpha in enumerate(alpha_values):
                    plt.plot(range(steps), rms_errors_all[i, j], label=f'{label_base} n={n}, alpha={alpha}')

    # Plot TD(lambda) results
    elif method == 'tdlambda' and lambda_values is not None:
        if harmonic_alpha:
            rms_errors_all, _ = td_lambda_harmonic(num_states, lambda_values, steps)
            label_base = 'TD(lambda) Harmonic Sequence'
            for i, lam in enumerate(lambda_values):
                plt.plot(range(steps), rms_errors_all[i], label=f'{label_base} λ={lam}, alpha=Harmonic')
        elif prime_sequence:
            rms_errors_all, _ = td_lambda_prime(num_states, lambda_values, steps)
            label_base = 'TD(lambda) Prime Sequence'
            for i, lam in enumerate(lambda_values):
                plt.plot(range(steps), rms_errors_all[i], label=f'{label_base} λ={lam}, alpha=Prime')
        else:
            rms_errors_all, _ = td_lambda(alpha_values, steps, num_states, lambda_values)
            label_base = 'TD(lambda) Constant Alpha'
            for i, lam in enumerate(lambda_values):
                for j, alpha in enumerate(alpha_values):
                    plt.plot(range(steps), rms_errors_all[i, j], label=f'{label_base} λ={lam}, alpha={alpha}')

    # Overlay the MC results, if applicable
    if overlay and rms_errors_mc is not None:
        if mc_label is None:  # Handling constant alpha case for MC
            for i, alpha in enumerate(alpha_values):
                plt.plot(range(steps), rms_errors_mc[i], label=f'MC Constant Alpha={alpha}', linestyle='--', color='black')
        else:
            plt.plot(range(steps), rms_errors_mc[0], label=mc_label, linestyle='--', color='black')

    plt.xlabel('Steps')
    plt.ylabel('RMS Error')
    plt.title(f'{method.upper()} with {"Harmonic Sequence α" if harmonic_alpha else "Prime Sequence α" if prime_sequence else "Constant α"}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plots for the selected method / values
def plot_for_varying_alpha(method, num_states, alpha_values, n_values=None, lambda_values=None, steps=100, harmonic_alpha=False, prime_sequence=False, overlay=False):
    true_values = generate_true_values(num_states)
    states = ['State {}'.format(i) for i in range(1, num_states + 1)]

    if method == 'tdn':
        if harmonic_alpha:
            rms_errors_all, avg_final_values = td_n_harmonic(steps, num_states, n_values)
            for i, n in enumerate(n_values):
                plt.plot(range(1, steps + 1), rms_errors_all[i], label=f'n={n}')
                print(f'\nMethod: TD(n), n={n}, Steps: {steps}')
                print('State Values vs True Values:')
                for state, final_value, true_value in zip(states, avg_final_values[i][1:num_states + 1], true_values):
                    print(f'{state}: Estimated Value = {final_value:.3f}, True Value = {true_value:.3f}')
        elif prime_sequence:
            rms_errors_all, avg_final_values = td_n_prime(steps, num_states, n_values)
            for i, n in enumerate(n_values):
                plt.plot(range(1, steps + 1), rms_errors_all[i], label=f'n={n}')
                print(f'\nMethod: TD(n) Prime Sequence, n={n}, Steps: {steps}')
                print('State Values vs True Values:')
                for state, final_value, true_value in zip(states, avg_final_values[i][1:num_states + 1], true_values):
                    print(f'{state}: Estimated Value = {final_value:.3f}, True Value = {true_value:.3f}')
        else:
            rms_errors_all, avg_final_values = td_n(alpha_values, steps, num_states, n_values)
            for j, alpha in enumerate(alpha_values):
                for i, n in enumerate(n_values):
                    plt.plot(range(1, steps + 1), rms_errors_all[i, j], label=f'n={n}, alpha={alpha}')
                    print(f'\nMethod: TD(n), n={n}, Steps: {steps}')
                    print(f'\nAlpha = {alpha}')
                    print('State Values vs True Values:')
                    for state, final_value, true_value in zip(states, avg_final_values[i, j, 1:num_states + 1], true_values):
                        print(f'{state}: Estimated Value = {final_value:.3f}, True Value = {true_value:.3f}')    
    elif method == 'tdlambda':
        if harmonic_alpha:
            rms_errors_all, avg_final_values = td_lambda_harmonic(num_states, lambda_values, steps)
            for i, lam in enumerate(lambda_values):
                plt.plot(range(1, steps + 1), rms_errors_all[i], label=f'lambda={lam}')
                print(f'\nMethod: TD(lambda), lambda={lam}, Steps: {steps}')
                print('State Values vs True Values:')
                for state, final_value, true_value in zip(states, avg_final_values[i][1:num_states + 1], true_values):
                    print(f'{state}: Estimated Value = {final_value:.3f}, True Value = {true_value:.3f}')
        elif prime_sequence:
            rms_errors_all, avg_final_values = td_lambda_prime(num_states, lambda_values, steps)
            for i, lam in enumerate(lambda_values):
                plt.plot(range(1, steps + 1), rms_errors_all[i], label=f'lambda={lam}')
                print(f'\nMethod: TD(lambda) Prime Sequence, lambda={lam}, Steps: {steps}')
                print('State Values vs True Values:')
                for state, final_value, true_value in zip(states, avg_final_values[i][1:num_states + 1], true_values):
                    print(f'{state}: Estimated Value = {final_value:.3f}, True Value = {true_value:.3f}')
        else:
            rms_errors_all, avg_final_values = td_lambda(alpha_values, steps, num_states, lambda_values)
            for i, lam in enumerate(lambda_values):
                for j, alpha in enumerate(alpha_values):
                    plt.plot(range(1, steps + 1), rms_errors_all[i, j], label=f'lambda={lam}, alpha={alpha}')
                    print(f'\nMethod: TD(lambda), lambda={lam}, Steps: {steps}')
                    print(f'\nAlpha = {alpha}')
                    print('State Values vs True Values:')
                    for state, final_value, true_value in zip(states, avg_final_values[i, j, 1:num_states + 1], true_values):
                        print(f'{state}: Estimated Value = {final_value:.3f}, True Value = {true_value:.3f}')
    elif method == 'mc':
        if harmonic_alpha:
            rms_errors_all, avg_final_values = mc_harmonic(steps, num_states)
            # BELOW????? [i] in avg_final values? 
            plt.plot(range(1, steps + 1), rms_errors_all, label=f'{steps} Steps')
            print(f'\nMethod: MC, Steps: {steps}')
            print('State Values vs True Values:')
            for state, final_value, true_value in zip(states, avg_final_values[1:num_states + 1], true_values):
                print(f'{state}: Estimated Value = {final_value:.3f}, True Value = {true_value:.3f}')
        elif prime_sequence:
            rms_errors_all, avg_final_values = mc_prime(steps, num_states)
            plt.plot(range(1, steps + 1), rms_errors_all, label=f'{steps} Steps')
            print(f'\nMethod: MC Prime Sequence, Steps: {steps}')
            print('State Values vs True Values:')
            for state, final_value, true_value in zip(states, avg_final_values[1:num_states + 1], true_values):
                print(f'{state}: Estimated Value = {final_value:.3f}, True Value = {true_value:.3f}')
        else:
            rms_errors_all, avg_final_values = mc_constant_alpha(steps, num_states, alpha_values)
            for i, alpha in enumerate(alpha_values):
                plt.plot(range(1, steps + 1), rms_errors_all[i], label=f'alpha={alpha}')
                print(f'\nMethod: MC, Steps: {steps}, Alpha={alpha}')
                print('State Values vs True Values:')
                for state, final_value, true_value in zip(states, avg_final_values[i, 1:num_states + 1], true_values):
                    print(f'{state}: Estimated Value = {final_value:.3f}, True Value = {true_value:.3f}')
    
    plt.xlabel('Steps')
    # plt.xlabel('Alpha' if not harmonic_alpha else 'Steps')
    plt.ylabel('RMS Error')
    plt.title(f'{method.upper()} with {"Harmonic Sequence " if harmonic_alpha else "Prime Sequence " if prime_sequence else "Constant "}α')
    plt.legend()
    plt.grid(True)
    plt.show()

# Visualize random walk 
def visualize_random_walk(num_states, max_steps=100):
    env = Env(num_states)
    states = []
    current_state = env.reset()

    fig, ax = plt.subplots(figsize=(8, 6))
    line, = ax.plot([], [], marker='o', color='b', linestyle='-', lw=2)
    
    ax.set_ylim(0, num_states + 1)  # The y-axis represents the states
    ax.set_xlim(0, 10)  # Start with a small x-axis range and expand dynamically
    ax.set_yticks(range(1, num_states + 1))
    ax.set_xticks([])  # Start without x-ticks and add them as needed
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('State')
    ax.set_title('Random Walk: State Over Time')

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        nonlocal current_state
        
        states.append(current_state)
        new_state, reward, done = env.step()
        
        # Update the current state before plotting and printing
        current_state = new_state

        # Update the plot with the new state
        line.set_data(range(len(states)), states)
        
        # Adjust the x-axis to fit the growing number of steps
        ax.set_xlim(0, len(states) + 1)
        ax.set_xticks(range(0, len(states) + 1, max(1, len(states) // 10)))
        
        ax.scatter(len(states) - 1, states[-1], color='red', s=100)
        
        # Debug print statements
        print(f"Step {len(states)}: Current State = {current_state}")
        
        if current_state == 1 or current_state == num_states:
            # Add the final state before stopping
            states.append(current_state)
            line.set_data(range(len(states)), states)
            ax.scatter(len(states) - 1, states[-1], color='red', s=100)
            print("Absorbing state reached. Stopping animation.")
            ani.event_source.stop()

        return line,

    ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, repeat=False, interval=500)
    plt.show()

# Step-by-step user input
def main():
    visualize = input("Do you want to visualize a random walk? (yes/no): ").strip().lower()
    if visualize == 'yes':
        states = int(input("Enter the number of states in the random walk problem: ").strip())
        visualize_random_walk(num_states=states)
    else: 
        method = input("Enter method (mc, tdn, tdlambda): ").strip()
        steps = int(input("Enter the number of steps: ").strip())
        states = int(input("Enter the number of states in the random walk problem: ").strip())

        alpha_type = input("Enter alpha type (constant-alpha, harmonic-alpha, prime-sequence-alpha): ").strip()

        if method == 'tdn':
            n_values = list(map(int, input("Enter n values (space-separated): ").strip().split()))
            lambda_values = None
        elif method == 'tdlambda':
            lambda_values = list(map(float, input("Enter lambda values (space-separated): ").strip().split()))
            n_values = None
        else:
            n_values = None
            lambda_values = None

        parser = argparse.ArgumentParser(description='Random Walk Methods Visualization')
        parser.add_argument('--overlay', action='store_true', help='Overlay MC plot on TD plots')
        args = parser.parse_args()

        if alpha_type == "constant-alpha":
            alpha_values = list(map(float, input("Enter alpha values (space-separated): ").strip().split()))
            harmonic_alpha = False
            prime_sequence = False
        elif alpha_type == "harmonic-alpha":
            alpha_values = [0]
            harmonic_alpha = True
            prime_sequence = False
        elif alpha_type == "prime-sequence-alpha":
            alpha_values = [0]
            harmonic_alpha = False
            prime_sequence = True
        else:
            raise ValueError("Invalid alpha type specified.")

        # Use the --overlay flag from argparse
        overlay = args.overlay

        # Overlay MC results if specified and method is not MC
        if overlay:
            plot_with_overlay(
        method=method,
        num_states=states,
        alpha_values=alpha_values,
        n_values=n_values,
        steps=steps,
        harmonic_alpha=harmonic_alpha,
        prime_sequence=prime_sequence,
        overlay=True  # Set to True to include MC overlay
    )
        else:
            plot_for_varying_alpha(
                method,
                num_states=states,
                alpha_values=alpha_values,
                n_values=n_values,
                lambda_values=lambda_values,
                steps=steps,
                harmonic_alpha=harmonic_alpha,
                prime_sequence=prime_sequence,
                overlay=overlay
            )

    # After plotting the RMS error, generate the true vs estimated plots
        true_values = generate_true_values(states)
        # TD(n) - True VS Estimated Values Plot
        if method == 'tdn':
            if harmonic_alpha:
                _, avg_final_values = td_n_harmonic(steps, states, n_values)
                alpha_label = 'Harmonic Sequence'
                for i, n in enumerate(n_values):
                    plot_true_vs_estimated(true_values, avg_final_values[i][1:states + 1], 'TD(n)', alpha=alpha_label, n=n)
            elif prime_sequence:
                _, avg_final_values = td_n_prime(steps, states, n_values)
                alpha_label = 'Prime Sequence'
                for i, n in enumerate(n_values):
                    plot_true_vs_estimated(true_values, avg_final_values[i][1:states + 1], 'TD(n)', alpha=alpha_label, n=n)
            else:
                _, avg_final_values = td_n(alpha_values, steps, states, n_values)
                for i, n in enumerate(n_values):
                    for j, alpha in enumerate(alpha_values):
                        plot_true_vs_estimated(true_values, avg_final_values[i][1:states + 1], 'TD(n)', alpha=alpha, n=n)
        # TD Lambda - True VS Estimated Values Plot
        elif method == 'tdlambda':
            if harmonic_alpha:
                _, avg_final_values = td_lambda_harmonic(states, lambda_values, steps)
                alpha_label = 'Harmonic Sequence'
                for i, lam in enumerate(lambda_values):
                    plot_true_vs_estimated(true_values, avg_final_values[i][1:states + 1], 'TD(lambda)', alpha=alpha_label, lam=lam)
            elif prime_sequence:
                _, avg_final_values = td_lambda_prime(states, lambda_values, steps)
                alpha_label = 'Prime Sequence'
                for i, lam in enumerate(lambda_values):
                    plot_true_vs_estimated(true_values, avg_final_values[i][1:states + 1], 'TD(lambda)', alpha=alpha_label, lam=lam)
            else:
                _, avg_final_values = td_lambda(alpha_values, steps, states, lambda_values)
                for i, lam in enumerate(lambda_values):
                    for j, alpha in enumerate(alpha_values):
                        estimated_values = avg_final_values[i, j, 1:states + 1]
                        plot_true_vs_estimated(true_values, estimated_values, 'TD(lambda)', alpha=alpha, lam=lam)
        # Monte Carlo - True VS Estimated Values Plot
        elif method == 'mc':
            if harmonic_alpha:
                _, avg_final_values = mc_harmonic(steps, states)
                alpha_label = 'Harmonic Sequence'
                plot_true_vs_estimated(true_values, avg_final_values[1:states + 1], 'MC', alpha=alpha_label)
            elif prime_sequence:
                _, avg_final_values = mc_prime(steps, states)
                alpha_label = 'Prime Sequence'
                plot_true_vs_estimated(true_values, avg_final_values[1:states + 1], 'MC', alpha=alpha_label)
            else:
                _, avg_final_values = mc_constant_alpha(steps, states, alpha_values)
                for i, alpha in enumerate(alpha_values):
                    estimated_values = avg_final_values[i, 1:states + 1]
                    plot_true_vs_estimated(true_values, estimated_values, 'MC', alpha=alpha)
                return
            
if __name__ == '__main__':
    main()
