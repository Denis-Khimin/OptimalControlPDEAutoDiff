import torch
from torch_sparse_solve import solve
import matplotlib.pyplot as plt
import numpy as np
import os

# number of spatial elements
n_h = 150 # n_h+1 spatial DoFs; n_h-1 interior spatial DoFs
# spatial mesh size
h = 1.0 / n_h
# number of time steps
n_k = 50 # n_k temporal DoFs + initial condition
# time step size
k = 0.5 / n_k
# number of unconstrained space-time DoFs
n = (n_h-1) * n_k

_t = torch.linspace(0, 0.5, n_k+1)
_x = torch.linspace(0, 1, n_h+1)

# initial condition
u0_true = torch.sin(torch.pi * _x[1:-1]).reshape(1, n_h-1, 1).type(torch.float64)
u0_guess = torch.ones_like(u0_true).requires_grad_()

f = [
    torch.Tensor(
        (torch.exp(-0.5 * tq) * (0.5 + torch.pi**2 + (torch.pi**2-0.5)*tq) ) * torch.sin(torch.pi * _x[1:-1])
    ).reshape(1, n_h-1, 1).type(torch.float64)
    for tq in _t[1:]
]

def heat_matrix():
    # NOTE: In the following, the boundary nodes u(0) = u(1) = 0 
    #       are not included in the linear system.

    # define the system matrix for 1+1D heat equation with finite differences in time and space
    diagonals = torch.zeros((3, n_h-1), dtype=torch.float64)
    diagonals[0, :] = -1.0 / h**2
    diagonals[2, :] = -1.0 / h**2
    diagonals[1, :] = 1. / k + 2.0 / h**2

    return torch.sparse.spdiags(
        diagonals=diagonals,
        offsets=torch.tensor([-1, 0, 1]),
        shape=(n_h-1, n_h-1)
    ).unsqueeze(0)

# solve the heat equation with backward Euler time stepping
A_k = heat_matrix()

def time_stepping(u0):
    # full space-time solution
    _u = [torch.zeros(1, (n_h-1), 1, dtype=torch.float64) for _ in range(n_k+1)]
    # add the initial condition to _u
    _u[0] = u0

    # time stepping loop for true solution
    for i in range(n_k):
        # solve the linear systems and update _u
        _u[i+1] = solve(A_k, f[i] + (1. / k) * _u[i])

    return _u

def mse_time(u1, u2):
    mse = 0.
    for i in range(n_k+1):
        mse += (1. / (n_k+1)) * torch.norm(u1[i] - u2[i])
    return mse

# prepare optimization
iter = 0
MAX_ITER = 500
optimizer = torch.optim.Rprop([u0_guess], lr=0.1)
print("Optimizing the initial condition of the 1+1D heat equation...")
print(f"Number of parameters: {u0_guess.numel()}")

# initialize full true and guessed solution trajectories
u_true = time_stepping(u0_true)
u_guess = time_stepping(u0_guess)

# Lists to store iteration numbers and corresponding loss values
iterations = []
losses = []

# optimize the initial condition as long as u_true and u_guess are not close enough
while mse_time(u_true, u_guess) > 1e-6 and iter < MAX_ITER:
    iter += 1

    loss = mse_time(u_true, u_guess)
    
    # Append iteration and loss to lists
    iterations.append(iter)
    losses.append(loss.detach().numpy())

    print(f"Iteration {iter}: Loss = {loss}")

    # zero the gradients
    optimizer.zero_grad()

    # solve the linear system
    u_guess = time_stepping(u0_guess)

    # compute the loss
    loss = torch.norm(mse_time(u_true, u_guess)) + 0.1 * torch.norm(u0_guess)

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
plt.savefig("Results/loss_history.png")

# plot u for the entire space-time domain (meshgrid)
u_meshgrid = torch.zeros((n_k+1, n_h-1))
for i in range(n_k+1):
    u_meshgrid[i, :] = u_guess[i].flatten()
u_meshgrid = torch.flip(u_meshgrid, [0])
plt.title("Solution of the 1+1D heat equation with time stepping")
plt.imshow(u_meshgrid.detach().numpy(), aspect='auto', extent=[_x[1], _x[-2], _t[0], _t[-1]])
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.savefig("Results/solution.png")
plt.clf()

# plot the initial condition
plt.title("Optimized initial condition of the 1+1D heat equation with time stepping")
plt.plot(_x[1:-1], u0_guess.detach().numpy().flatten(), label="Numerical solution")
plt.plot(_x[1:-1], u0_true.detach().numpy().flatten(), label="True solution")
plt.legend()
plt.savefig("Results/initial_condition.png")
