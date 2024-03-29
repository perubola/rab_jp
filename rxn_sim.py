import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize 
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

np.random.seed(42)

# stoichiometric matrix
N = np.array([[1, -1, 0, -1, 0],
              [0, -1, -1, 0, 0],
              [0, 0, 1, 1, -1]])


# PARAMS 
# we choose mu for boundary
mu_R1 = 100
mu_R2 = 0 

vb1 = 0.1
vb2 = 0.01 
# we then sample the other mu's in between these 
# mu0_1, mu0_2, mu0_3 = np.random.uniform(mu_R2, mu_R1, 3)
# mu0 = np.array([mu0_1, mu0_2, mu0_3])

# for now let's use a predetermined mu
mu0 = np.array([20, 80, 20])
BD = np.array([vb1,0, 0, 0, vb2])

# helper functions
def get_G(x, N, mu0, BD):
    mu = mu0 + np.log(x)
    G = N.T @ mu + BD
    return G


def flux_vector(x, N, mu0, k, BD, vb1, vb2):
    g = get_G(x, N, mu0, BD=BD)

    k1, k2, k3 = k

    v = np.zeros_like(g)  # Initialize flux vector with zeros
    v[0] = vb1 * (1 - np.exp(g[0]))  # Reaction from R1
    v[1] = k1 * x[0] * (1 - np.exp(g[1]))  # x1 to x2
    v[2] = k2 * x[1] * (1 - np.exp(g[2]))  # x2 to x3
    v[3] = k3 * x[2] * (1 - np.exp(g[3]))  # x3 to R2
    v[4] = vb2 * (1 - np.exp(g[4]))  # Reaction to R2
    return v


### PART 2: FINDING K for a range of x0
def objective_function(k, x0, N, mu0, BD, vb1, vb2):
    v = flux_vector(x0, N, mu0, k, BD, vb1, vb2)
    # Calculate deviation from steady state (N * v should be close to zero at steady state)
    deviation = N @ v
    # Objective is the norm of this deviation
    cost = np.linalg.norm(deviation)
    return cost

# Function to run optimization for a given x0
def optimize_for_x0(x0):
    #NOTE: Here is where you change initial k guess
    k_initial_guess = np.array([0.1, 0.1, 0.1])
    result = minimize(objective_function, k_initial_guess, args=(x0, N, mu0, BD, vb1, vb2))
    if result.success:
        return (x0, result.x)  # Return the initial x0 and the optimized k values
    else:
        return (x0, None)

# Generate a list of x0 vectors for testing
num_samples = 10  # Choose how many x0 vectors you want to test
x0_samples = [np.random.uniform(low=0.001, high=100.0, size=3) for _ in range(num_samples)]

# Initialize a list to hold successful optimizations
# stores as (x, k)
successful_optimizations = []

# Loop through the x0 vectors and optimize for each
for i in tqdm(range(len(x0_samples))):
    x0 = x0_samples[i]
    result = optimize_for_x0(x0)
    if result[1] is not None:  # Check if optimization was successful
        successful_optimizations.append(result)
        print(f"Successful optimization for x0 = {result[0]}: optimized k = {result[1]}")
    # else:
    #     print(f"Optimization failed for x0 = {result[0]}")

# Summary
print(f"Total successful optimizations: {len(successful_optimizations)}/{len(x0_samples)}")
