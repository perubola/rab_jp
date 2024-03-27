# simulation of the kinetics of a toy metabolic map 
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# reaction constants 
k1 = 1 
k2 = 1 
r12 = 1
r13 = 1
r23 = 1

# initial concentrations 
x1_0 = 0
x2_0 = 0 
x3_0 = 0 


# initial reservoir values 
R1 = 2 
R2 = 0

# chemical potentials
muR1 = 10
muR2 = 0

np.random.seed(42)
mu1, mu2, mu3 = np.random.uniform(muR2, muR1, 3)
# mu1 = 7
# mu2 = 1
# mu3 = 9

# thermo 
dg13 = mu3 - mu1 
dg12 = mu2 - mu1 
dg23 = mu3 - mu2  # check if this sign is right 


# initial vector
z0 = [x1_0, x2_0, x3_0]

def toydt(t, z, k1, k2, r12, r13, r23):
    # z in this case will be a vector containing x1, x2, x3, and reservoir values 
    x1, x2, x3 = z
    v13 = r13*x1 - r13*x3*np.exp(dg13)
    v12 = r12*x1 - r12*x2*np.exp(dg12)
    v23 = r23*x2 - r23*x3*np.exp(dg23) 
    dx1dt = -v12 - v13 + k1*(R1-x1)
    dx2dt = v12 + v23 + k2*(R2-x2)
    dx3dt = v13 - v23  



    return [dx1dt, dx2dt, dx3dt]

tf = 3

sol = solve_ivp(toydt, [0, tf], z0, args=(k1, k2, r12, r13, r23), dense_output=True)
t = np.linspace(0, tf, 300)
z = sol.sol(t)

plt.plot(t, z.T)
plt.xlabel('t')
plt.ylabel('conc')
plt.legend(['x1', 'x2', 'x3'], shadow=True)
plt.suptitle(f"$\mu_1$ = {mu1:.2f}, $\mu_2$ = {mu2:.2f}, $\mu_3$ = {mu3:.2f}")
plt.show()


