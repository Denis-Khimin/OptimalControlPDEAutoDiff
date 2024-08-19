import torch
import math
import numpy as np
import os

# Import fenics and override necessary data structures with fenics_adjoint
from fenics import *
from fenics_adjoint import *
import matplotlib.pyplot as plt

import torch_fenics

class Heat(torch_fenics.FEniCSModule):
    def __init__(self):
        super().__init__()

        # create mesh and create function space
        self.mesh = UnitSquareMesh(10, 10)
        self.V = FunctionSpace(self.mesh, 'P', 1)
        self.dofs = self.V.tabulate_dof_coordinates()
        print(f"Number of DoFs: {self.V.dim()}")

    def solve(self, u0):
        # temporal discretization
        t = 0.
        T = 1.
        k = 0.01

        # get time steps
        num_steps = int(T/k)
        _u = [None for _ in range(num_steps+1)]

        # u_n: solution from last time step
        u_n = u0
        _u[0] = u_n.copy(deepcopy=True)
        
        # variational problem
        u = Function(self.V) # u = u_{n+1}: current solution
        v = TestFunction(self.V)

        # right hand side
        f = Expression('(-2*t*exp(pow(t, 2)) + exp(t)*sin(pi*x[0])*sin(pi*x[1]) + exp(pow(t, 2)) + 2*pow(pi, 2)*exp(pow(t, 2)))*exp(-2*pow(t, 2) + t)*sin(pi*x[0])*sin(pi*x[1])', degree=4, t=t, pi=math.pi) 
        f_old = Expression('(-2*t*exp(pow(t, 2)) + exp(t)*sin(pi*x[0])*sin(pi*x[1]) + exp(pow(t, 2)) + 2*pow(pi, 2)*exp(pow(t, 2)))*exp(-2*pow(t, 2) + t)*sin(pi*x[0])*sin(pi*x[1])', degree=4, t=t, pi=math.pi) 

        F = (u-u_n)*v*dx + 0.5*k*dot(grad(u)+grad(u_n), grad(v))*dx + 0.5*k*(pow(u, 2)+pow(u_n, 2))*v*dx - 0.5*k*(f_old+f)*v*dx 

        self.u_mid = [u0(Point(0.5, 0.5))]
        self.t_values = [t]
        i = 0
        while(t+k <= T+1e-8):
            # Update current time
            t += k

            # Compute solution
            f.t = t
            f_old.t = t-k
            solve(F == 0, u, DirichletBC(self.V, Constant(0), lambda _, on_boundary: on_boundary))

            # save value at midpoint
            self.u_mid.append(u(Point(0.5, 0.5)))
            self.t_values.append(t)

            # Update previous solution
            u_n.assign(u)
            i += 1
            _u[i] = u_n.copy(deepcopy=True)

        return tuple(_u) # torch_fenics needs to return a tuple in solve()

    def input_templates(self):
        return Function(self.V)

if __name__ == '__main__':
    # Construct the FEniCS module
    heat = Heat()

    if not os.path.exists("Results"):
        os.makedirs("Results")

    # get location of DoFs
    dofs = torch.tensor(heat.dofs, dtype=torch.float64)
    u0_true = torch.sin(math.pi * dofs[:,0]) * torch.sin(math.pi * dofs[:,1])
    uT_true = torch.sin(math.pi * dofs[:,0]) * torch.sin(math.pi * dofs[:,1])

    # return the true solution trajectory as vtk files
    vtkfile = File('Results/heat_solution_true.pvd')
    u = Function(heat.V)
    for t in np.linspace(0, 1, 100):
        _t = torch.tensor([t], dtype=torch.float64)
        u.vector()[:] = torch.sin(math.pi * dofs[:,0]) * torch.sin(math.pi * dofs[:,1]) * torch.exp(_t-_t*_t)
        vtkfile << u

    # perform optimization of u0_guess
    u0_guess = torch.autograd.Variable(
        torch.zeros(1, heat.V.dim(), dtype=torch.float64),
        requires_grad=True
    )    
    u_guess = heat(u0_guess)
    uT_guess = u_guess[-1]

    # prepare optimization
    iter = 0
    MAX_ITER = 100
    optimizer = torch.optim.Rprop([u0_guess], lr=0.1)
    print("Optimizing the initial condition in the nonlinear heat equation...")
    print(f"Number of parameters: {u0_guess.numel()}")

    # Lists to store iteration numbers and corresponding loss values
    iterations = []
    losses = []

    # optimize the parameters as long as u0_true and u0_guess, as well as uT_true and uT_guess are not close enough
    while torch.norm(u0_true - u0_guess) + torch.norm(uT_true - uT_guess) > 1e-6 and iter < MAX_ITER:
        iter += 1

        loss = (torch.norm(u0_true - u0_guess) + torch.norm(uT_true - uT_guess)).detach().numpy()

        # Append iteration and loss to lists
        iterations.append(iter)
        losses.append(loss)

        print(f"Iteration {iter}: Loss = {loss}")

        # zero the gradients
        optimizer.zero_grad()

        # solve the heat equation
        u_guess = heat(u0_guess)
        uT_guess = u_guess[-1]

        # compute the loss
        loss = torch.norm(u0_true - u0_guess) + torch.norm(uT_true - uT_guess) + 0.1 * torch.norm(u0_guess)

        # backpropagate
        loss.backward()
        
        # update the parameters
        optimizer.step()

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

    # plot the solution at the midpoint
    plt.title("Solution at center of domain")
    plt.plot(heat.t_values, heat.u_mid, label="Numerical solution")
    plt.plot(heat.t_values, [math.exp(_t-_t*_t) for _t in heat.t_values], label="True solution")
    plt.legend()
    # save the plot
    plt.savefig("Results/heat_solution_center_trajectory.png")

    # return the recovered solution trajectory as vtk files
    vtkfile = File('Results/heat_solution_guess.pvd')
    u = Function(heat.V)
    for i, _u in enumerate(u_guess):
        u.vector()[:] = _u.detach().numpy().flatten()
        vtkfile << u

    
