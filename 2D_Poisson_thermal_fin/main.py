import torch
import numpy as np
from fenics import *
from fenics_adjoint import *
import matplotlib.pyplot as plt
import torch_fenics
import os

if not os.path.exists("Results"): 
    os.makedirs("Results")

class Poisson(torch_fenics.FEniCSModule):
    def __init__(self):
        super().__init__()

        # Load mesh from xml file and create function space
        self.mesh = Mesh("thermal_fin_mesh.xml")
        self.V = FunctionSpace(self.mesh, 'P', 1)
        print(f"Number of DoFs: {self.V.dim()}")

        # Load subdomains from xml file
        self.subdomains = MeshFunction("size_t", self.mesh, "thermal_fin_subdomains.xml")
        self.dx = Measure('dx', domain=self.mesh, subdomain_data=self.subdomains)

        # mark boundaries for boundary conditions
        self.neumann = CompiledSubDomain("near(x[1], 0.) && on_boundary")
        self.robin = CompiledSubDomain("!near(x[1], 0.) && on_boundary")
        # create facet marker
        self.facet_marker = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
        self.facet_marker.set_all(0)
        self.neumann.mark(self.facet_marker, 1)
        self.robin.mark(self.facet_marker, 2)
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.facet_marker)

    def solve(self, mu):
        # Create trial and test functions
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        # Construct bilinear form:
        # * subdomain integrals with different heat conductivities mu[i] = k_i
        a = mu[0] * inner(grad(u), grad(v)) * self.dx(1) + mu[1] * inner(grad(u), grad(v)) * self.dx(2) + mu[2] * inner(grad(u), grad(v)) * self.dx(3) + mu[3] * inner(grad(u), grad(v)) * self.dx(4) + mu[4] * inner(grad(u), grad(v)) * self.dx(5)

        # * boundary integral for Robin boundary condition with heat transfer coefficient mu[5] = Bi
        a += mu[5] * u * v * self.ds(2)

        # Construct linear form
        L = Constant(1.) * v * self.ds(1)

        # Solve the Poisson equation
        u = Function(self.V)
        solve(a == L, u)

        # Return the solution
        return u

    def input_templates(self):
        # Declare templates for the inputs to Poisson.solve
        return Constant((0, 0, 0, 0, 0, 0))
    
    def plot(self, u, plot_=True, title='Poisson solution', save=False, filename='Results/solution.pvd'):
        # Plot the solution
        _u = Function(self.V)
        _u.vector().set_local(u.detach().numpy().flatten())

        # Create matplotlib figure
        if plot_:
            c = plot(_u, title='Temperature')
            plt.colorbar(c)
            plt.show()

        # Save the solution as a pvd file
        if save:
            vtkfile = File(filename)
            vtkfile << _u


if __name__ == '__main__':
    # Construct the FEniCS model
    poisson = Poisson()

    if not os.path.exists("Results"):
        os.makedirs("Results")    

    # mu = [k0, k1, k2, k3, k4, Bi] -> parameters in PDE which are to be learned
    mu_true = torch.tensor([[0.1, 8.37317446377103, 6.572276066240383, 0.46651735398635275, 1.8835410659596712, 0.01]], dtype=torch.float64)

    # get the true solution of the Poisson equation
    u_true = poisson(mu_true)

    # plot the true solution
    poisson.plot(u_true, plot_=False, title="Reference solution", save=True, filename='Results/thermal_fin_true_solution.pvd')

    # get a reference mu for regularization
    mu_reference = torch.tensor([[1., 1., 1., 1., 1., 0.1]], dtype=torch.float64)

    # perform optimization of mu_guess
    mu_guess = torch.tensor(0.5 * torch.ones(1, 6, dtype=torch.float64), requires_grad=True) # = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    u_guess = poisson(mu_guess)

    # prepare optimization
    iter = 0
    MAX_ITER = 100
    optimizer = torch.optim.Rprop([mu_guess], lr=0.01)
    print("Optimizing the parameters in the thermal fin problem...")
    print("Number of parameters: {mu_guess.numel()}")

    # Lists to store iteration numbers and corresponding loss values
    iterations = []
    losses = []

    # optimize the parameters as long as u_true and u_guess are not close enough at the Neumann boundary
    # loss_history = []
    while torch.norm(u_true - u_guess) > 1e-6 and iter < MAX_ITER:
        iter += 1

        loss = torch.norm(u_true - u_guess).detach().numpy()
    
        # Append iteration and loss to lists
        iterations.append(iter)
        losses.append(loss)

        print(f"Iteration {iter}: Loss = {loss}")

        # zero the gradients
        optimizer.zero_grad()

        # solve the Poisson equation
        u_guess = poisson(mu_guess)

        # compute the loss
        loss = torch.norm(u_true - u_guess) + 0.1 * torch.norm((mu_guess - mu_reference) / mu_reference)

        # backpropagate
        loss.backward()
        
        # update the parameters
        optimizer.step()
        
        # apply constraints
        with torch.no_grad():
            mu_guess.clamp_(
                min=torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.01], dtype=torch.float64), 
                max=torch.tensor([10., 10., 10., 10., 10., 1.], dtype=torch.float64)
            )
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

    # plot the recovered solution
    poisson.plot(u_guess, plot_=False, title="Recovered solution", save=True, filename='Results/thermal_fin_recovered_solution.pvd')
    print(f"Recovered parameters: {mu_guess}")
    print(f"True parameters: {mu_true}")
