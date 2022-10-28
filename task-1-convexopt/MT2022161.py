''' LEAST NORM PROBLEM '''

# Import packages.
import cvxpy as cp
import numpy as np

#cost function
def norm(x):
    out=0
    for i in range(x.shape[0]):
        out=out+x[i]**2
    return out

# Generate a random non-trivial quadratic program.
m = 15
n = 15
np.random.seed(1)
A = np.random.randn(m, n)
B = np.random.randn(m)
# Define and solve the CVXPY problem.
x = cp.Variable(n)
cost=norm(x)
prob = cp.Problem(cp.Minimize(cost),
                 [A @ x == B])
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
