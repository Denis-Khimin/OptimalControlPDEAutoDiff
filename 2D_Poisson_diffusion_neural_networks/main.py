import torch
# Import fenics and override necessary data structures with fenics_adjoint
from fenics import *
from fenics_adjoint import *
import matplotlib.pyplot as plt
import math
import argparse
import torch_fenics

plt.rcParams['text.usetex'] = True

class Poisson(torch_fenics.FEniCSModule):
    def __init__(self, n_elements=40):
        super().__init__()

        # create unit square mesh
        self.mesh = UnitSquareMesh(n_elements, n_elements)
        self.V = FunctionSpace(self.mesh, 'P', 1)
        print(f"Number of DoFs: {self.V.dim():,}")

        # get the DoFs in self.V
        self.dofs = self.V.tabulate_dof_coordinates().reshape(-1, 2)

    def solve(self, kappa):
        # Create trial and test functions
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        # right hand side
        f = Expression('-6*pi*sin(pi*x[0])*cos(pi*x[1]) + 2*pi*pi*(2*x[0]+3*x[1]*x[1]+1)*sin(pi*x[0])*sin(pi*x[1]) - 2*pi*sin(pi*x[1])*cos(pi*x[0])', degree=2, pi=math.pi)

        # homogeneous Dirichlet boundary conditions
        bc = DirichletBC(self.V, Constant(0), lambda _, on_boundary: on_boundary)

        # Construct bilinear form:
        a = dot(kappa * grad(u), grad(v))*dx
        L = f*v*dx

        # Solve the Poisson equation
        u = Function(self.V)
        solve(a == L, u, bc)

        # Return the solution
        return u

    def input_templates(self):
        return Function(self.V)
    
    def plot(self, u, plot_=True, title='Poisson solution', save=False, filename='solution.pvd'):
        # Plot the solution
        _u = Function(self.V)
        _u.vector().set_local(u.detach().numpy().flatten())

        # Create matplotlib figure
        if plot_:
            c = plot(_u, title=title)
            plt.colorbar(c)
            plt.show()

        # Save the solution as a pvd file
        if save:
            vtkfile = File(filename)
            vtkfile << _u

def data_driven_training(nn, n_elements=40, name="data", save=True):
    # Construct the FEniCS model
    poisson = Poisson(n_elements=n_elements)

    # Define the input and target
    input = torch.tensor(poisson.dofs, dtype=torch.float64)
    target = 1. + 2. * input[:, 0] + 3. * input[:, 1]**2
    target = target.reshape(-1, 1)

    # Define the loss function
    loss_fn = torch.nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.1)

    # Train the neural network
    for i in range(1000):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = nn(input)
        loss = loss_fn(output, target)

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Print the loss
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss.item()}")

    # plot the neural network solution, the target and the error
    u_nn = nn(input).detach()
    poisson.plot(u_nn, title=r'$\kappa_{NN}^{guess}$', plot_=True, save=save, filename=f"poisson_{name}_nn_solution.pvd")
    poisson.plot(target, title=r'$\kappa^{true}$', plot_=True, save=save, filename=f"poisson_{name}_target_solution.pvd")
    poisson.plot((u_nn - target)**2, title=r'$(\kappa^{true} - \kappa_{NN}^{guess})^2$', plot_=True, save=save, filename=f"poisson_{name}_error.pvd")

    return None

def physics_informed_training(nn, n_elements=40, name="physics", learning_rate=0.1, save=True):
    # Construct the FEniCS model
    poisson = Poisson(n_elements=n_elements)

    # Define the input and target
    input = torch.tensor(poisson.dofs, dtype=torch.float64)
    target = 1. + 2. * input[:, 0] + 3. * input[:, 1]**2
    target = target.reshape(-1, 1)
    u_true = poisson(target.T)

    # Define the optimizer
    optimizer = torch.optim.Rprop(nn.parameters(), lr=learning_rate)

    print("MSE before training:", torch.nn.MSELoss()(nn(input), target).item())

    # Train the neural network
    for i in range(1000):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        kappa = nn(input).T
        # solve the Poisson equation
        u_guess = poisson(kappa)

        # compute the loss
        loss = torch.norm(u_true - u_guess) # NOTE: possible to also add data loss here, i.e. compare kappa with target

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Print the loss
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss.item()}")

    print("MSE after training:", torch.nn.MSELoss()(nn(input), target).item())

    # plot the neural network solution, the target and the error
    u_nn = nn(input).detach()
    poisson.plot(u_nn, title=r'$\kappa_{NN}^{guess}$', plot_=True, save=save, filename=f"poisson_{name}_nn_solution.pvd")
    poisson.plot(target, title=r'$\kappa^{true}$', plot_=True, save=save, filename=f"poisson_{name}_target_solution.pvd")
    poisson.plot((u_nn - target)**2, title=r'$(\kappa^{true} - \kappa_{NN}^{guess})^2$', plot_=True, save=save, filename=f"poisson_{name}_error.pvd")

    return None

if __name__ == '__main__':
    # load arguments from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_type", type=str, default="physics_informed", choices=["data_driven", "physics_informed", "mixed"], help="How to train the neural network: 'data_driven' or 'physics_informed' or 'mixed'?")
    args = parser.parse_args()

    # Define the neural network
    nn = torch.nn.Sequential(
        torch.nn.Linear(2, 20),
        torch.nn.Sigmoid(),
        torch.nn.Linear(20, 1)
    ).to(torch.float64)
    print("Number of parameters in the neural network:", sum(p.numel() for p in nn.parameters()))


    if args.train_type == "data_driven":
        data_driven_training(nn, n_elements=40, save=False) # train NN purely on data
    elif args.train_type == "mixed":
        data_driven_training(nn, n_elements=4, name="pretrain", save=False) # pre-training NN on coarse data
        physics_informed_training(nn, n_elements=40, name="finetune", save=False) # fine-tuning NN on physics on fine mesh
    elif args.train_type == "physics_informed":
        physics_informed_training(nn, n_elements=40, learning_rate=0.001, save=False) # train NN purely on physics
    else:
        raise ValueError(f"Invalid training type: {args.train_type}")
