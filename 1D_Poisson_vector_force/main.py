import torch
from torch_sparse_solve import solve
import matplotlib.pyplot as plt
import os

# number of spatial elements
n = 50
# mesh size
h = 1.0 / n
# right hand side
# set dtype to torch.float64 to avoid numerical issues
b_true = torch.pi**2 * torch.sin(torch.pi * torch.linspace(0, 1, n+1)[1:-1]).reshape(1, n-1, 1).type(torch.float64) 
b_guess = torch.ones(1, n-1, 1, dtype=torch.float64).requires_grad_()

def poisson_matrix():
    # NOTE: In the following, the boundary nodes u(0) = u(1) = 0 
    #       are not included in the linear system.

    # define the 1D Poisson finite difference matrix as torch.sparse:
    # write [-1 / h**2, 2 / h**2, 1 / h**2] on the diagonals using torch.sparse.spdiags
    diagonals = torch.ones((3, n-1), dtype=torch.float64) * 1.0 / h**2
    diagonals[0, :] *= -1.
    diagonals[2, :] *= -1.
    diagonals[1, :] *= 2.

    A = torch.sparse.spdiags(
        diagonals=diagonals,
        offsets=torch.tensor([-1, 0, 1]),
        shape=(n-1, n-1)
    )

    # reshape A to have batch dimension 1
    return A.unsqueeze(0)

# prepare the true solution and the linear system
A = poisson_matrix()
u_true = solve(A, b_true)

# prepare the initial guess
u_guess = solve(A, b_guess)

# prepare optimization
iter = 0
MAX_ITER = 1000
optimizer = torch.optim.Rprop([b_guess], lr=0.1)
print("Optimizing the right-hand side of the 1D Poisson equation...")
print(f"Number of parameters: {b_guess.numel()}")

# Lists to store iteration numbers and corresponding loss values
iterations = []
losses = []

# optimize the RHS as long as u_true and u_guess are not close enough
while torch.norm(u_true - u_guess) > 1e-6 and iter < MAX_ITER:
    iter += 1
   
    loss = torch.norm(u_true - u_guess).item()
    
    # Append iteration and loss to lists
    iterations.append(iter)
    losses.append(loss)

    print(f"Iteration {iter}: Loss = {loss}")

    # zero the gradients
    optimizer.zero_grad()

    # solve the linear system
    u_guess = solve(A, b_guess)

    # compute the loss
    loss = torch.norm(u_true - u_guess) + 0.099 * torch.norm(b_guess)

    # backpropagate
    loss.backward()

    # update the RHS
    optimizer.step()

if not os.path.exists("Results"): 
    os.makedirs("Results")

# Plot iteration vs loss
plt.figure()
plt.plot(iterations, losses, label="Loss over iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Loss function vs Iteration")
plt.legend()
plt.savefig("Results/loss.png")
plt.clf()

# prepare the full solutions
u_full_true = torch.zeros(n+1, dtype=torch.float64)
u_full_true[1:-1] = u_true.squeeze()
u_full_guess = torch.zeros(n+1, dtype=torch.float64)
u_full_guess[1:-1] = u_guess.squeeze()

# plot the solution
x = torch.linspace(0, 1, n+1)
plt.title("Inverse recovery of the RHS of the 1D Poisson equation")
plt.plot(x, u_full_true.detach().numpy(), label=f"True solution")
plt.plot(x, u_full_guess.detach().numpy(), label=f"Recovered solution")
plt.legend()
plt.savefig("Results/solution.png")
plt.clf()

plt.title("Inverse recovery of the RHS of the 1D Poisson equation")
plt.plot(x[1:-1], b_true.detach().numpy().flatten(), label=f"True RHS")
plt.plot(x[1:-1], b_guess.detach().numpy().flatten(), label=f"Recovered RHS")
plt.legend()
plt.savefig("Results/rhs.png")
plt.clf()

# print solution at the midpoint
print(f"u_true({x[n//2]}) = {u_full_true[n//2]}")
print(f"u_guess({x[n//2]}) = {u_full_guess[n//2]}")
