import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
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
    # x = np.maximum(x, 1e-10)  # trying to catch 0 or negatives
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

### PART 3 (for real this time): PLOTTING SOLUTIONS



def flux_sys(t, x, N, mu0, k, mu_bd):
    v = flux_vector(x=x, k=k, N=N, mu0=mu0, mu_bd=mu_bd)
    return N @ v


def plot_time_series(x0, t_span, N, mu0, k, mu_bd):
    solution = solve_ivp(lambda t, x: flux_sys(t, x, N, mu0, k, mu_bd), t_span, x0, method='BDF')
    x1 = solution.y[0]
    x2 = solution.y[1]
    x3 = solution.y[2]


    plt.figure(figsize=(10, 8))
    plt.plot(solution.t, x1, label='x1')
    plt.plot(solution.t, x2, label='x2')
    plt.plot(solution.t, x3, label='x3')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('Time Series')
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_phase_space_with_arrows(x0, t_span, N, mu0, k, mu_bd, x0_sys, num_points=20, arrows=False):
    """
    2D phase space (x1, x2) with optional quiver plot
    """
    # Integrate the system over the time span
    solution = solve_ivp(lambda t, x: flux_sys(t, x, N, mu0, k, mu_bd), t_span, x0, method='BDF')

    # Plot the trajectories and the quiver
    plt.figure(figsize=(10, 8))
    plt.plot(solution.y[0], solution.y[1], 'r-', label='Trajectory')  # Trajectory in phase space
    plt.plot(x0[0], x0[1], 'ro', label="Starting Point")
    plt.plot(x0_sys[0], x0_sys[1],'bo', label="Stable Point")

    if arrows:
        # Grid for the quiver plot
        x1_min, x1_max = min(solution.y[0]), max(solution.y[0])
        x2_min, x2_max = min(solution.y[1]), max(solution.y[1])
        x1_vals = np.linspace(x1_min, x1_max, num_points)
        x2_vals = np.linspace(x2_min, x2_max, num_points)

        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        U, V = np.zeros_like(X1), np.zeros_like(X2)
        # Calculate the direction vectors at each point on the grid
        for i in range(num_points):
            for j in range(num_points):
                x_sample = np.array([X1[i, j], X2[i, j], 1])  # Assuming a 3D system with x3 normalized to 1
                dxdt = flux_sys(0, x_sample, N, mu0, k, mu_bd)[:2]  # We only take the first two derivatives
                U[i, j], V[i, j] = dxdt
        plt.quiver(X1, X2, U, V, color='blue', width=0.0025, scale=100)  # Flow directions

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Phase Space Plot with Flow Directions')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_3d_phase_space_with_arrows(x0, t_span, N, mu0, k, mu_bd, x_sys, num_points=5, arrows=False):
    # Integrate the system over the time span
    solution = solve_ivp(lambda t, x: flux_sys(t, x, N, mu0, k, mu_bd), t_span, x0, method='BDF')

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(solution.y[0], solution.y[1], solution.y[2], 'r-', label='Trajectory')
    ax.plot(x0[0], x0[1], x0[2], 'ro', label="Initial x0")
    ax.plot(x_sys[0], x_sys[1], x_sys[2], 'bo', label="Optimized x0")

    if arrows:
        # Create a 3D grid for the quiver plot
        min_vals = np.min(solution.y, axis=1)
        max_vals = np.max(solution.y, axis=1)
        grid_vals = [np.linspace(min_vals[i], max_vals[i], num_points) for i in range(3)]
        X1, X2, X3 = np.meshgrid(*grid_vals)
        U, V, W = np.zeros_like(X1), np.zeros_like(X2), np.zeros_like(X3)

        for i in range(num_points):
            for j in range(num_points):
                for h in range(num_points):
                    x_sample = np.array([X1[i, j, h], X2[i, j, h], X3[i, j, h]])
                    dxdt = flux_sys(0, x_sample, N, mu0, k, mu_bd)
                    U[i, j, h], V[i, j, h], W[i, j, h] = dxdt
        ax.quiver(X1, X2, X3, U, V, W, length=0.01, linewidth=0.5, arrow_length_ratio=0.3)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title(f'3D Phase Space with Flow Directions \n x0_stable: {x0_sys}')
    ax.legend()
    plt.show()

# Example usage
t_span = [0, 10]
idx = 4
x0_sys = x0_samples[idx]
k_sys = k_samples[idx]

def generate_perturbed_initial_conditions(x0, perturbation_strength=0.1):
    # Perturb each component of x0 by a random amount within ±perturbation_strength% of each component's value
    perturbed_x0 = x0 * (1 + np.random.uniform(-perturbation_strength, perturbation_strength, size=x0.shape))
    return perturbed_x0

x0_test = generate_perturbed_initial_conditions(x0_sys, perturbation_strength=0.9)
plot_time_series(x0_test, t_span, N, mu0, k=k_sys, mu_bd=mu_bd)
plot_phase_space_with_arrows(x0_test, t_span, N, mu0, k_sys, mu_bd, x0_sys)
plot_3d_phase_space_with_arrows(x0_test, t_span, N, mu0, k=k_sys, mu_bd=mu_bd, x_sys=x0_sys)


