# simulation of the kinetics of a toy metabolic map 
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# reaction constants 
k1 = 10
k2 = 10
r12 = 5
r13 = 1
r23 = 1

# initial concentrations 
x1_0 = 5
x2_0 = 0
x3_0 = 0


# initial reservoir values 
R1 = 10
R2 = 0

# initial vector
z0 = [x1_0, x2_0, x3_0]

def toydt(t, z, k1, k2, r12, r13, r23):
    # z in this case will be a vector containing x1, x2, x3, and reservoir values 
    x1, x2, x3 = z
    dx1dt = (-r12 - r13)*x1 + k1*(R1-x1) + r12*x2 + r13*x3
    dx2dt = (-r12 - r23)*x2 + k2*(R2-x2) + r12*x1 + r23*x3 
    dx3dt = r13*x1 + r23*x2 - (r13 + r23)*x3



    return [dx1dt, dx2dt, dx3dt]

tf = 5

sol = solve_ivp(toydt, [0, tf], z0, args=(k1, k2, r12, r13, r23), dense_output=True)
t = np.linspace(0, tf, 300)
z = sol.sol(t)

plt.plot(t, z.T)
plt.xlabel('t')
plt.ylabel('conc')
plt.legend(['x1', 'x2', 'x3'], shadow=True)
plt.show()


