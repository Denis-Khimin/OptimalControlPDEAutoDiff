import torch
import numpy as np
# Import fenics and override necessary data structures with fenics_adjoint
from fenics import *
from fenics_adjoint import *
import matplotlib.pyplot as plt
import os
import torch_fenics

set_log_level(30) # suppress FEniCS output
parameters["reorder_dofs_serial"] = False # No DoF reordering -> easier solution vector manipulation

class FluidStructureInteraction(torch_fenics.FEniCSModule):
    def __init__(self):
        super().__init__()

        # load mesh, subdomain and boundaries
        self.load_domain()

        # create function space
        element = {
            "u": VectorElement("Lagrange", self.mesh.ufl_cell(), 2),
            "v": VectorElement("Lagrange", self.mesh.ufl_cell(), 2),
            "p": FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
        }
        self.V = FunctionSpace(self.mesh, MixedElement(*element.values()))
        self._U = self.V.sub(0)
        self._V = self.V.sub(1)
        self._P = self.V.sub(2)
        print(f"Numer of DoFs: {self.V.dim():,} ({self._U.dim():,} + {self._V.dim():,} + {self._P.dim():,})")

        self.dof_at_tip = 10730 # beam tip DoF for this mesh
        print("DoF at tip of elastic beam:", self.dof_at_tip)

    def load_domain(self):
        # load mesh from xml file
        self.mesh = Mesh("fsi_mesh.xml")

        # load subdomain from xml file
        self.subdomains = MeshFunction("size_t", self.mesh, "fsi_subdomains.xml")
        
        # boundaries
        inflow = CompiledSubDomain("near(x[0], 0.) && on_boundary")
        wall = CompiledSubDomain("(near(x[1], 0.) || near(x[1], 0.41)) && on_boundary")
        outflow = CompiledSubDomain("near(x[0], 2.5) && on_boundary")
        cylinder = CompiledSubDomain("on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3")
        beam_dirichlet = CompiledSubDomain("on_boundary && x[0]<0.3 && x[0]>0.2458257 && x[1]>0.1 && x[1]<0.3")

        self.facet_marker = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
        self.facet_marker.set_all(0)
        inflow.mark(self.facet_marker, 1)
        wall.mark(self.facet_marker, 2)
        outflow.mark(self.facet_marker, 3)
        cylinder.mark(self.facet_marker, 4)
        beam_dirichlet.mark(self.facet_marker, 5)

    def input_templates(self):
        return Constant(0.)
    
    def solve(self, mu):
        # Define boundary conditions
        bc_u_inflow = DirichletBC(self._U, Constant((0, 0)), self.facet_marker, 1)
        bc_u_walls = DirichletBC(self._U, Constant((0, 0)), self.facet_marker, 2)
        bc_u_outflow = DirichletBC(self._U, Constant((0, 0)), self.facet_marker, 3)
        bc_u_cylinder = DirichletBC(self._U, Constant((0, 0)), self.facet_marker, 4)
        bc_u_beam = DirichletBC(self._U, Constant((0, 0)), self.facet_marker, 5)
        inflow_parabola = ('1.5*0.2*4.0*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')
        bc_v_inflow = DirichletBC(self._V, Expression(inflow_parabola, degree=2), self.facet_marker, 1)
        bc_v_walls = DirichletBC(self._V, Constant((0, 0)), self.facet_marker, 2)
        bc_v_cylinder = DirichletBC(self._V, Constant((0, 0)), self.facet_marker, 4)
        bc_v_beam = DirichletBC(self._V, Constant((0, 0)), self.facet_marker, 5)
        bc_u = [bc_u_inflow, bc_u_walls, bc_u_outflow, bc_u_cylinder, bc_u_beam]
        bc_v = [bc_v_inflow, bc_v_walls, bc_v_cylinder, bc_v_beam]
        bc_p = []
        bc = bc_u + bc_v + bc_p

        # material parameters
        mu_s     = mu                                     # 2nd Lame coefficient (solid)
        nu_s     = 0.4                                    # Poisson ratio        (solid)
        lambda_s = 2.0 * mu_s * nu_s / (1.0 - 2.0 * nu_s) # 1st Lame coefficient (solid)
        rho_s    = 1.0e3                                  # density              (solid)
        nu_f     = 1.0e-3                                 # kinematic viscosity  (fluid)
        rho_f    = 1.0e3                                  # density              (fluid)
        mu_f     = nu_f * rho_f                           # dynamic viscosity    (fluid)
        # extension parameters
        alpha_u  = 1.0e-12
        alpha_v  = 1.0e3 
        alpha_p  = 1.0e-12

        # integration measures
        dx = Measure("dx", domain=self.mesh, subdomain_data=self.subdomains)
        dx_solid = dx(1) # integrate over solid domain
        dx_fluid = dx(2) # integrate over fluid domain
        
        # split functions
        U = Function(self.V)
        (u, v, p) = split(U)
        Psi = TestFunction(self.V)
        (psi_u, psi_v, psi_p) = split(Psi)

        # parameters for variational form
        I     = Identity(2)
        F_hat = I + grad(u)
        E_hat = 0.5 * (F_hat.T * F_hat - I)
        J_hat = det(F_hat)
        # stress tensors
        sigma_f = -p * I + mu_f * (grad(v) * inv(F_hat) + inv(F_hat).T * grad(v).T)
        sigma_s = 2.0 * mu_s * E_hat + lambda_s * tr(E_hat) * I

        # weak form
        # fluid equations
        fluid_convection        = inner(rho_f * J_hat * grad(v) * inv(F_hat) * v, psi_v) * dx_fluid
        fluid_momentum          = inner(J_hat * sigma_f * inv(F_hat).T, grad(psi_v)) * dx_fluid
        fluid_incompressibility = inner(div(J_hat * inv(F_hat) * v), psi_p) * dx_fluid
        fluid_u_extension       = inner(alpha_u * grad(u), grad(psi_u)) * dx_fluid

        # solid equations
        solid_momentum          = inner(F_hat * sigma_s, grad(psi_v)) * dx_solid
        solid_v_extension       = alpha_v * inner(v, psi_u) * dx_solid 
        solid_p_extension       = alpha_p * (inner(grad(p), grad(psi_p)) + inner(p, psi_p)) * dx_solid 
        
        F = fluid_convection + fluid_momentum + fluid_incompressibility + fluid_u_extension + solid_momentum + solid_v_extension + solid_p_extension

        # Compute Jacobian
        J = derivative(F, U)

        # Create solver
        problem = NonlinearVariationalProblem(F, U, bc, J)
        solver  = NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1e-8
        prm['newton_solver']['relative_tolerance'] = 1e-7
        prm['newton_solver']['maximum_iterations'] = 25
        prm['newton_solver']['relaxation_parameter'] = 1.0
        prm['newton_solver']['linear_solver'] = 'mumps'

        solver.solve()

        # print(U(0.6, 0.2)) # solution at the tip of the beam

        return (
            project(u, self._U.collapse()),
            project(v, self._V.collapse()),
            project(p, self._P.collapse())
        )
    
    def save(self, U, name_prefix="test_"):
        # Save the solution as pvd
        _U = Function(self.V)
        _U.vector().set_local(U.detach().numpy().flatten())
        _u, _v, _p = _U.split(deepcopy=True)
        _u.rename("u", "u")
        _v.rename("v", "v")
        _p.rename("p", "p")
        vtkfile_u = File(os.path.join("Results", name_prefix + "u.pvd"))
        vtkfile_u << _u
        vtkfile_v = File(os.path.join("Results", name_prefix + "v.pvd"))
        vtkfile_v << _v
        vtkfile_p = File(os.path.join("Results", name_prefix + "p.pvd"))
        vtkfile_p << _p

if __name__ == '__main__':
    # Construct the FEniCS model
    fsi = FluidStructureInteraction()

    if not os.path.exists("Results"):
        os.makedirs("Results")

    mu_true = torch.tensor([[5.0e5]], dtype=torch.float64)
    mu_guess = torch.tensor([[5.0e3]], dtype=torch.float64, requires_grad=True)

    # compute the reference solution
    u_true, v_true, p_true  = fsi(mu_true)
    uy_tip_true = u_true[0, fsi.dof_at_tip, 1]
    print(f"True mu: {mu_true.item():.5e}")
    print(f"True y-deformation at beam tip: {uy_tip_true.item():.5e}")
    # save the reference solution
    U_true = torch.cat((u_true.reshape(1,-1), v_true.reshape(1,-1), p_true), dim=1).flatten()
    fsi.save(U_true, "true_")

    # compute the initial guess
    u_guess, v_guess, p_guess  = fsi(mu_guess)
    uy_tip_guess = u_guess[0, fsi.dof_at_tip, 1]
    print(f"Initial guess mu: {mu_guess.item():.5e}")
    print(f"Initial guess y-deformation at beam tip: {uy_tip_guess.item():.5e}")
    # save the initial guess
    U_guess = torch.cat((u_guess.reshape(1,-1), v_guess.reshape(1,-1), p_guess), dim=1).flatten()
    fsi.save(U_guess, "initial_")

    # prepare optimization
    iter = 0
    MAX_ITER = 50
    optimizer = torch.optim.Adam([mu_guess], lr=1.5e7)
    print("Optimizing the Lame-parameter in the Fluid-Structure Interaction problem...")
    print(f"Number of parameters: {mu_guess.numel()}")

    # optimize the parameters the true y-deformation at the beam tip and the guess are too far apart
    loss_history = []
    gradient_history = []
    mu_history = [mu_guess.item()]
    while iter < MAX_ITER:
        iter += 1
        error = torch.pow(uy_tip_true-uy_tip_guess, 2)
        if error < 1e-13:
            loss_history.append(error.item())
            print(f"Reached sufficient accuracy: Error = {error.item():.5e}")
            break # reached sufficient accuracy
        print(f"Iteration {iter}: Error = {error.item():.5e}")

        # zero the gradients
        optimizer.zero_grad()

        # compute the loss
        loss = torch.pow(uy_tip_true-uy_tip_guess, 2)
        loss_history.append(loss.item())

        # backpropagate
        loss.backward()
        print(f"  Gradient: {mu_guess.grad.item():.5e}")
        gradient_history.append(mu_guess.grad.item())

        # update the parameter
        optimizer.step()
        print(f"  New guess mu: {mu_guess.item():.5e}")
        mu_history.append(mu_guess.item())

        # solve the FSI equations
        u_guess, v_guess, p_guess = fsi(mu_guess)
        uy_tip_guess = u_guess[0, fsi.dof_at_tip, 1]
        print(f"  Current guess y-deformation at beam tip: {uy_tip_guess.item():.5e}")

    # save the final guess
    U_guess = torch.cat((u_guess.reshape(1,-1), v_guess.reshape(1,-1), p_guess), dim=1).flatten()
    fsi.save(U_guess, "final_")

    # plot loss history
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Loss history")
    plt.savefig("Results/loss_history.png")
    plt.clf()

    # plot gradient history
    plt.plot([-g for g in gradient_history])
    plt.xlabel("Iteration")
    plt.ylabel("Negative gradient")
    plt.yscale("log")
    plt.title("Gradient history")
    plt.savefig("Results/gradient_history.png")
    plt.clf()

    # plot mu history
    plt.plot(mu_history)
    plt.xlabel("Iteration")
    plt.ylabel("Parameter")
    plt.yscale("log")
    plt.title("Lame-parameter history")
    plt.savefig("Results/mu_history.png")
