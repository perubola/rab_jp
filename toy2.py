# TFA for second toy model
# has a bifurcation 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog
from scipy.integrate import solve_ivp
from tqdm import tqdm 

np.random.seed(42)

# we have x1, x2, x3, ATP, and ADP
N = np.array([[1, -1, -1, 0, 0],
              [0, 1, 0, -1, 0],
              [0, 0, 1, 0, -1],
              [0, -1, 1, 0, 0],
              [0, 1, -1, 0, 0]])

mu_B1 = 0.5
mu_B2 = 0.0
mu_B3 = 1.0
# columns 2 and 3 correspond to the reactions 
mu0 = np.array([0.4, 0.3, 0.1, 0.3, 0.2])  # for the x_i
mu_bd = np.array([mu_B1, 0, 0, mu_B2, mu_B3])

def get_G(x, N, mu0, mu_bd):
    mu = mu0 + np.log(x)
    G = N.T @ mu - mu_bd
    return G

def flux_vector(x, N, mu0, k, mu_bd):
    g = get_G(x, N, mu0, mu_bd)
    vb1, k1 , k2, vb2, vb3 = k 
    v = np.zeros_like(g)
    v[0] = vb1 * (1 - np.exp(g[0]))  # b1 to x1 
    v[1] = k1 * x[0] * (1 - np.exp(g[1]))  # x1 to x3
    v[2] = k2 * x[0] * (1 - np.exp(g[2]))  # x1 to x2
    v[3] = vb2 * x[1] * (1 - np.exp(g[3]))  # x2 to b2 
    v[4] = vb3 * x[2]* (1 - np.exp(g[4]))  # x3 to b3 
    return v

# finding k for s.s.
def calculate_parameters(x0, v0, N, mu0, mu_bd):
    k0 = np.ones_like(v0)
    k = v0 / flux_vector(x0, N, mu0, k0, mu_bd)
    return k

def flux_samples(num_samples, a=[0.1, 1]):
    sample = np.zeros((num_samples, 5))
    for n in range(num_samples):
        alpha = np.random.uniform(*a)
        v0 = [1, alpha * 1 , alpha * 1,  1-alpha, 1]
        sample[n] = v0
    return sample

# Generate for thermodynamically consistent x0
# Find x s.t. dG_v < 0
def concentration_samples(num_samples, N, mu0, mu_bd, step_size=0.01, eps=1e-3, dropout=100):
    # find initial condition
    # TODO get samples that are in a thermodynamically feasible range
    b = -mu_bd + N.T @ mu0
    c  = np.zeros_like(mu0)
    res = linprog(c, A_ub=N.T, b_ub= -b -eps)
    z0 = res.x  # inital point inside constraint
    if not res.success:
        raise RuntimeError("Problem infeasible")
    tot_steps = num_samples * dropout
    i = 0
    samples = np.zeros((tot_steps, mu0.shape[0]))
    while i < tot_steps:
        u = np.random.randn(*mu0.shape)
        u = u/np.linalg.norm(u)

        z = z0 + u * step_size
        
        if all(N.T @ z <= -b -eps):
            samples[i, :] = z
            z0 = z
            i += 1
    return np.exp(samples[::dropout, :])  # :: gives every 100

num_samples = 10000  # Choose how many x0 vectors you want to test
x0_samples = concentration_samples(num_samples, N, mu0, mu_bd)  # comes in groups of 3
# We constructruct the flux vectors for each x0
v0_samples = flux_samples(num_samples)
# Calculate the k values for each x0 and v0
k_samples = np.zeros((num_samples, 5))
for i in range(num_samples):
    x0 = x0_samples[i]
    v0 = v0_samples[i]
    k_samples[i] = calculate_parameters(x0, v0, N, mu0, mu_bd)
# Calc dG
dG = np.zeros((num_samples, 5))
for i in range(num_samples):
    x0 = x0_samples[i]
    dG[i] = get_G(x0, N, mu0, mu_bd=mu_bd)



def flux_sys(t, x, N, mu0, k, mu_bd):
    v = flux_vector(x=x, k=k, N=N, mu0=mu0, mu_bd=mu_bd)
    return N @ v


def plot_time_series(x0, t_span, N, mu0, k, mu_bd):
    solution = solve_ivp(lambda t, x: flux_sys(t, x, N, mu0, k, mu_bd), t_span, x0, method='BDF')
    x1 = solution.y[0]
    x2 = solution.y[1]
    x3 = solution.y[2]
    atp = solution.y[3]
    adp = solution.y[4]

    plt.figure(figsize=(10, 8))
    plt.plot(solution.t, x1, label='x1')
    plt.plot(solution.t, x2, label='x2')
    plt.plot(solution.t, x3, label='x3')
    plt.plot(solution.t, atp, label='atp')
    plt.plot(solution.t, adp, label='adp')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('Time Series')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
t_span = [0, 10]
idx = 0
x0_sys = x0_samples[idx]
k_sys = k_samples[idx]

def generate_perturbed_initial_conditions(x0, perturbation_strength=0.1):
    # Perturb each component of x0 by a random amount within ±perturbation_strength% of each component's value
    perturbed_x0 = x0 * np.abs((1 + np.random.uniform(-perturbation_strength, perturbation_strength, size=x0.shape)))
    # perturbed_x0 = x0 * 2 
    return perturbed_x0


x0_test = generate_perturbed_initial_conditions(x0_sys, perturbation_strength=0.5)
plot_time_series(x0_test, t_span, N, mu0, k=k_sys, mu_bd=mu_bd)