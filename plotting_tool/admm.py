import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


# Constants and parameters
D = 100  # Total demand
c = np.array([1, 5, 2])  # Cost coefficients for each generator
capacities = np.array([40, 50, 40])  # Capacity constraints for each generator
rho = 0.01  # Penalty parameter

# Initial values
x = np.array([0.0, 0.0, 0.0], dtype=float)  # Starting with a feasible production level for each generator
lambda_hat = 0  # Dual variables for each generator

# Optimization problem for each generator
def solve_qp(c_i, p_max_i, lambda_hat, x_i_old, x_ave_old, rho, D):
    # Define the variable for generator's production level
    x_i = cp.Variable()
    
    # Define the quadratic cost function for the generator
    cost = c_i * x_i + lambda_hat * (x_i) + (rho / 2) * cp.square(x_i - (x_i_old- 3*x_ave_old) - D)
    
    # Define the constraints for the generator
    constraints = [0 <= x_i, x_i <= p_max_i]
    
    # Set up the problem and solve it
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    
    # If problem is solved, return the value. If not, return previous value.
    return x_i.value

# ADMM iteration
max_iter = 300
x_history = [np.copy(x)]
lambda_history = [np.copy(lambda_hat)]  # Initialize lambda history


for iteration in range(max_iter):
    # Save current x values before they are updated
    x_old = np.copy(x)
    x_avg_old = np.mean(x_old)
    lambda_hat_old = np.copy(lambda_hat)
    
    # x-update step for each generator
    for i in range(3):
        x[i] = solve_qp(c[i], capacities[i], lambda_hat, x_old[i],  x_avg_old,  rho, D)
    
    x_avg = np.mean(x)
    
    lambda_hat += rho * (-D + sum(x))
    #or 
    # lambda_hat += rho * (D - 3*x_avg)
    lambda_history.append(np.copy(lambda_hat))
    
    x_history.append(np.copy(x))


    # Check for lambda change for termination condition
    if np.linalg.norm(lambda_hat - lambda_hat_old) < 1e-4:
        break
    

x_history = np.array(x_history)
lambda_history = np.array(lambda_history)


# Print results
print("Final Production levels:", x)
print("Total production:", np.sum(x))
print("Total demand:", D)
print("Iterations:", iteration + 1)

# Plotting
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(x_history[:, i], label=f'Generator {i+1}')
plt.xlabel('Iteration')
plt.ylabel('Production Level')
plt.title('Convergence of Production Levels by Generator')
plt.legend()
plt.grid(True)
plt.show()

#recall the sing of lambda is based on the decomposition
print("price_clearing:", -lambda_hat)
plt.plot(-lambda_history)
