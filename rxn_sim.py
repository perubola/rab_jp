import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(42)

# helper functions
def get_G(x, N, mu0, BD):
    mu = mu0 + np.log(x)
    G = N.T * mu + BD
    return G



# stoichiometric matrix
N = np.array([[1, -1, 0, -1, 0],
              [0, -1, -1, 0, 0],
              [0, 0, 1, 1, -1]])


# PARAMS 
# we choose mu for boundary
mu_R1 = 10
mu_R2 = 0

vb1 = 1
vb2 = 1
# we then sample the other mu's in between these 
mu0_1, mu0_2, mu0_3 = np.random.uniform(0, 10, 3)
mu0 = np.array({mu0_1, mu0_2, mu0_3})


def flux_vector(x, N, mu0, k):
    g = get_G(x, N, mu0, BD=0)

    k1, k2, k3 = k
    v1 = vb1 * (1-np.exp(g[0]))
    v1 = k1* x * (1-np.exp(g[1]))
    v1 = k2 * x * (1-np.exp(g[2]))
    v1 = k3 * x * (1-np.exp(g[3]))
    v1 = vb2 * (1-np.exp(g[4]))

