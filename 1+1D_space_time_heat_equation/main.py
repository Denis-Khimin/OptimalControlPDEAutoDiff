import torch
from torch_sparse_solve import solve
import matplotlib.pyplot as plt
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

# right-hand side
f = torch.kron(
    torch.exp(-0.5 * _t[1:]) * (0.5 + torch.pi**2 + (torch.pi**2-0.5)*_t[1:]),
    torch.sin(torch.pi * _x[1:-1])
).reshape(1, n_k*(n_h-1), 1).type(torch.float64)

def heat_matrix():
    # NOTE: In the following, the spatial boundary nodes u(0, t) = u(1, t) = 0 
    #       are not included in the linear system.

    # define the 1+1D heat finite difference matrix as torch.sparse:
    diagonals = torch.zeros((4, n), dtype=torch.float64)
    diagonals[0, :] = -1. / k
    diagonals[1, :] = -1. / h**2
    diagonals[3, :] = -1. / h**2
    diagonals[2, :] = 1. / k + 2. / h**2
    
    for i in range(n_k-1):
        diagonals[1, (i+1)*(n_h-1)] = 0.   # outside of diagonal blocks of size (n_h-1) x (n_h-1)
        diagonals[3, (i+1)*(n_h-1)+1] = 0. # outside of diagonal blocks of size (n_h-1) x (n_h-1)

    return torch.sparse.spdiags(
        diagonals=diagonals,
        offsets=torch.tensor([-n_h+1,-1, 0, 1]),
        shape=(n, n)
    ).unsqueeze(0)

# prepare the true solution and the linear system
A = heat_matrix()
b_true = torch.Tensor(f).type(torch.float64)
b_true[:, :n_h-1, :] += (1. / k) * u0_true
u_true = solve(A, b_true)

# prepare the initial guess
b_guess = torch.ones_like(b_true).requires_grad_()
u_guess = solve(A, b_guess)

# prepare optimization
iter = 0
MAX_ITER = 1000
optimizer = torch.optim.Rprop([b_guess], lr=0.1)
print("Optimizing the initial condition of the 1+1D heat equation...")
print(f"Number of parameters: {b_guess.numel()}")

# Lists to store iteration numbers and corresponding loss values
iterations = []
losses = []

if not os.path.exists("Results"): 
    os.makedirs("Results")

# optimize the initial condition and RHS as long as u_true and u_guess are not close enough
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
    loss = torch.norm(u_guess-u_true) + 0.01 * torch.norm(b_guess)

    # backpropagate
    loss.backward()

    # update the RHS
    optimizer.step()

# Plot iteration vs loss
plt.figure()
plt.plot(iterations, losses, label="Loss over iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Loss function vs Iteration")
plt.legend()
plt.savefig("Results/loss_history.png")
plt.clf()

# plot the solution at the center of the spatial domain
u_mid = [u_guess[0, i*(n_h-1)+n_h//2, 0].item() for i in range(n_k)]
plt.title("Solution of the 1+1D heat equation at the center of the spatial domain")
plt.plot(_t[1:], u_mid, label="Numerical solution")
plt.plot(_t[1:], torch.exp(-0.5 * _t[1:]) * (1.0 + _t[1:]), label="True solution")
plt.legend()
plt.savefig("Results/solution_center.png")
plt.clf()

# plot u for the entire space-time domain (meshgrid)
u_meshgrid = torch.flip(u_guess.reshape(n_k, n_h-1), [0])
plt.title("Solution of the 1+1D heat equation")
plt.imshow(u_meshgrid.detach().numpy(), aspect='auto', extent=[_x[1], _x[-2], _t[1], _t[-1]])
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.savefig("Results/solution.png")
plt.clf()

# plot the right-hand side
b_meshgrid = torch.flip(b_guess.reshape(n_k, n_h-1), [0])
plt.title("Optimized RHS of the 1+1D heat equation")
plt.imshow(b_meshgrid.detach().numpy(), aspect='auto', extent=[_x[1], _x[-2], _t[1], _t[-1]])
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.savefig("Results/rhs.png")
