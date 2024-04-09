import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog
from scipy.linalg import null_space
from concurrent.futures import ProcessPoolExecutor
#from tqdm import tqdm
np.random.seed(42)
# stoichiometric matrix
N = np.array([[1, -1, 0, -1, 0],
              [0, 1, -1, 0, 0],
              [0, 0, 1, 1, -1]])
# PARAMS
# we choose mu for boundary
mu_R1 = 1
mu_R2 = 0
# we then sample the other mu's in between these
# mu0_1, mu0_2, mu0_3 = np.random.uniform(mu_R2, mu_R1, 3)
# mu0 = np.array([mu0_1, mu0_2, mu0_3])
# for now let's use a predetermined mu
mu0 = np.array([0.4, 0.3, 0.2])
mu_bd = np.array([mu_R1,0, 0, 0, mu_R2])
# helper functions
def get_G(x, N, mu0, mu_bd):
    mu = mu0 + np.log(x)
    G = N.T @ mu - mu_bd
    return G
def flux_vector(x, N, mu0, k, mu_bd):
    g = get_G(x, N, mu0, mu_bd=mu_bd)
    vb1, k1, k2, k3, vb2 = k
    v = np.zeros_like(g)  # Initialize flux vector with zeros
    v[0] = vb1 * (1 - np.exp(g[0]))  # Reaction from R1
    v[1] = k1 * x[0] * (1 - np.exp(g[1]))  # x1 to x2
    v[2] = k2 * x[1] * (1 - np.exp(g[2]))  # x2 to x3
    v[3] = k3 * x[2] * (1 - np.exp(g[3]))  # x3 to R2
    v[4] = vb2 * (1 - np.exp(g[4]))  # Reaction to R2
    return v
# ### PART 2: FINDING K for a set of steady state concentration x0 and flux vectors v0
def calculate_parameters( x0, v0, N, mu0, mu_bd):
    k0 = np.ones_like(v0)
    k = v0 / flux_vector(x0, N, mu0, k0, mu_bd)
    return k
# Note that each steady-state flux needs to be in N*v = 0
# generates flux samples from reference
def flux_samples(num_samples, a=[0.1,1] ):
    sample = np.zeros((num_samples, 5))
    for n in range(num_samples):
        alpha = np.random.uniform(*a)
        v0 = [1, alpha * 1 , (1-alpha) * 1,  1-alpha, 1]
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


num_samples = 10  # Choose how many x0 vectors you want to test
x0_samples = concentration_samples(num_samples, N, mu0, mu_bd)
# We constructruct the flux vectors for each x0
v0_samples = flux_samples(10)
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
# Note DW: Your idea here is interesting but the idea is we back calculate the k values from the flux vector
# ### PART 2: FINDING K for a range of x0
# def objective_function(k, x0, N, mu0, BD, vb1, vb2):
#     v = flux_vector(x0, N, mu0, k, BD, vb1, vb2)
#     # Calculate deviation from steady state (N * v should be close to zero at steady state)
#     deviation = N @ v
#     # Objective is the norm of this deviation
#     cost = np.linalg.norm(deviation)
#     return cost
# # Function to run optimization for a given x0
# def optimize_for_x0(x0):
#     #NOTE: Here is where you change initial k guess
#     k_initial_guess = np.array([0.1, 0.1, 0.1])
#     result = minimize(objective_function, k_initial_guess, args=(x0, N, mu0, BD, vb1, vb2))
#     if result.success:
#         return (x0, result.x)  # Return the initial x0 and the optimized k values
#     else:
#         return (x0, None)
# # Generate a list of x0 vectors for testing
# num_samples = 10  # Choose how many x0 vectors you want to test
# x0_samples = [np.random.uniform(low=0.001, high=100.0, size=3) for _ in range(num_samples)]
# # Initialize a list to hold successful optimizations
# # stores as (x, k)
# successful_optimizations = []
# # Loop through the x0 vectors and optimize for each
# for i in tqdm(range(len(x0_samples))):
#     x0 = x0_samples[i]
#     result = optimize_for_x0(x0)
#     if result[1] is not None:  # Check if optimization was successful
#         successful_optimizations.append(result)
#         print(f"Successful optimization for x0 = {result[0]}: optimized k = {result[1]}")
#     # else:
#     #     print(f"Optimization failed for x0 = {result[0]}")
# # Summary
# print(f"Total successful optimizations: {len(successful_optimizations)}/{len(x0_samples)}")
