import numpy as np
from scipy.integrate import odeint

# Function to calcualte the basin of attraction for the steady state
def basins_of_attratction(parameters ,vars=[0,1], log2_concentation_range=[-1,1],
                          n_points=10, t_max=500, concentration_noise=0.0,
                          n_time_points=1000, clamped=False):
    # Generate a list of points that are on boundary of a box in the concentration space
    # The box is defined by the log2_concentration_range
    # The number of points is defined by n_points
    if len(log2_concentation_range) == 2:
        log_concentrations = np.linspace(log2_concentation_range[0], log2_concentation_range[1], n_points)
        concentrations = 2**log_concentrations
        # Create a list of all the points on the boundary of the box
        boundary_points = []
        for i in range(n_points):
            for j in range(n_points):
                if i == 0 or i == n_points-1 or j == 0 or j == n_points-1:
                    boundary_points.append([concentrations[i], concentrations[j]])
    if len(log2_concentation_range) == 4:
        log_concentrations_x = np.linspace(log2_concentation_range[0], log2_concentation_range[1], n_points)
        log_concentrations_y = np.linspace(log2_concentation_range[2], log2_concentation_range[3], n_points)
        concentrations_x = 2**log_concentrations_x
        concentrations_y = 2**log_concentrations_y
        # Create a list of all the points on the boundary of the box
        boundary_points = []
        for i in range(n_points):
            for j in range(n_points):
                if i == 0 or i == n_points-1 or j == 0 or j == n_points-1:
                    boundary_points.append([concentrations_x[i], concentrations_y[j]])
    # Time vector
    t = np.linspace(0,t_max,n_time_points)
    # Calculate the trajectories for each of the boundary points
    results = []
    for point in boundary_points:
        X0 = reference_state(concentration_noise)
        X0[vars[0]] = point[0]
        X0[vars[1]] = point[1]
        if not 3 in vars:
            # Calculate the insulin concentration
            X0[3] = insulin(X0[0], 1.0, 0.0)
        else:
            X0[3] = point[vars.index(3)] * insulin(1.0, 1.0, 0.0)
        if clamped:
            result = odeint(equation_clamped,X0, t, args=(parameters,))
        else:
            result = odeint(equation,X0, t, args=(parameters,))
        results.append(result)
    return t, results

def equation(x,t,p):
     N,mu0,k = p
     dxdt = N * v(x,p)
     return [dxdt] (edited) 



