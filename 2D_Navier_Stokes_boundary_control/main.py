import torch
import numpy as np
# Import fenics and override necessary data structures with fenics_adjoint
from fenics import *
from fenics_adjoint import *
import matplotlib.pyplot as plt
import torch_fenics
import os

set_log_level(30) # suppress FEniCS output
parameters["reorder_dofs_serial"] = False # No DoF reordering -> easier solution vector manipulation

class NavierStokes(torch_fenics.FEniCSModule):
    def __init__(self):
        super().__init__()

        # load mesh from schaefer_turek_2D.xml
        self.mesh = Mesh("schaefer_turek_2D.xml") 
        
        # create function space
        element = {
            "v": VectorElement("Lagrange", self.mesh.ufl_cell(), 2),
            "p": FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
        }
        self.V = FunctionSpace(self.mesh, MixedElement(*element.values()))
        self._V = self.V.sub(0)
        self._P = self.V.sub(1)

        # create function space for control
        self.W = FunctionSpace(self.mesh, 'P', 1)
        print(f"Number of DoFs (linear): {self.W.dim()}")

        # Problem data
        self.inflow_parabola = ('4.0*0.3*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')
        self.nu = Constant(0.001)

        # Define boundaries
        self.inflow   = 'near(x[0], 0)'
        self.outflow  = 'near(x[0], 2.2)'
        self.walls    = 'on_boundary && (near(x[1], 0) || near(x[1], 0.41)) && (x[0]<0.2 || x[0]>0.3)'
        self.cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'
        self.control  = 'on_boundary && (near(x[1], 0) || near(x[1], 0.41)) && (x[0]>=0.2-1e-4 || x[0]<=0.3+1e-4)'

        # get integrator over the control boundary
        facet_marker = MeshFunction("size_t", self.mesh, 1)
        facet_marker.set_all(0)
        CompiledSubDomain(self.control).mark(facet_marker, 1)
        CompiledSubDomain(self.cylinder).mark(facet_marker, 2)
        self.ds_control = Measure("ds", subdomain_data=facet_marker, subdomain_id=1)
        self.ds_cylinder = Measure("ds", subdomain_data=facet_marker, subdomain_id=2)

        # get the DoFs in W at the control boundary
        self.control_dofs = self.W.tabulate_dof_coordinates().reshape(-1, 2)
        self.control_dofs_ids = np.where(np.logical_and(
            np.logical_and(self.control_dofs[:, 0] >= 0.2, self.control_dofs[:, 0] <= 0.3),
            np.logical_or(np.isclose(self.control_dofs[:, 1], 0.0), np.isclose(self.control_dofs[:, 1], 0.41))
        ))
        self.control_dofs = self.control_dofs[
            self.control_dofs_ids
        ]
        self.assemble_drag_tensor()

    def assemble_drag_tensor(self):
        # preassemble vector for drag tensor such that drag(u) = U_h * drag_tensor
        dU = TrialFunction(self.V) 
        dv, dp = split(dU)
        n = FacetNormal(self.mesh)
        D = 0.1
        v_bar = 2/3*4.0*0.3*0.205*(0.41 - 0.205) / pow(0.41, 2)
        self.drag_vector = assemble(
            2/(v_bar**2*D)*
            (
            - dot(dp * Identity(len(dv)), n)[0]
            + self.nu * dot(grad(dv), n)[0]
            ) * self.ds_cylinder
        ).get_local()

    def solve(self, q):
        # Define boundary conditions
        bc_v_inflow = DirichletBC(self._V, Expression(self.inflow_parabola, degree=2), self.inflow)
        bc_v_walls = DirichletBC(self._V, Constant((0, 0)), self.walls)
        bc_v_cylinder = DirichletBC(self._V, Constant((0, 0)), self.cylinder)
        bc_v = [bc_v_inflow, bc_v_walls, bc_v_cylinder]
        bc_p = []
        bc = bc_v + bc_p

        # Define trial and test functions and function at old time step
        U   = Function(self.V)
        Phi = TestFunctions(self.V)

        # Split functions into velocity and pressure components
        v, p = split(U)
        phi_v, phi_p = Phi

        # Define variational forms
        F = (
            self.nu * inner(grad(v), grad(phi_v))
            + dot(dot(grad(v), v), phi_v)
            - p * div(phi_v)
            + div(v) * phi_p
        ) * dx
        # add an integral over the control boundary
        n = FacetNormal(self.mesh)
        F -= q * inner(phi_v, n) * self.ds_control

        # Compute Jacobian
        J = derivative(F, U)

        # Create solver
        problem = NonlinearVariationalProblem(F, U, bc, J)
        solver = NonlinearVariationalSolver(problem)
        solver.solve()

        return (
            project(v, self._V.collapse()),
            project(p, self._P.collapse())
        ) 

    def input_templates(self):
        return Function(self.W)
    
    def save(self, U, filename='Results/solution.pvd'):
        # Save the solution as pvd
        _U = Function(self.V)
        _U.vector().set_local(U.detach().numpy().flatten())
        _v, _p = _U.split(deepcopy=True)

        # # plot magnitude of velocity with matplotlib
        # c = plot(sqrt(dot(_v, _v)), title='Velocity')
        # plt.colorbar(c, orientation='horizontal')
        # plt.show()

        vtkfile_v = File(filename.replace(".pvd", "_v.pvd"))
        vtkfile_v << _v
        vtkfile_p = File(filename.replace(".pvd", "_p.pvd"))
        vtkfile_p << _p

if __name__ == '__main__':
    # Construct the FEniCS model
    navier_stokes = NavierStokes()

    if not os.path.exists("Results"):
        os.makedirs("Results")

    d = navier_stokes.control_dofs_ids[0].shape[0]
    # print(navier_stokes.control_dofs)
    
    # get torch sparse matrix that maps control DoFs to global DoFs
    indices = torch.cat(
        [torch.tensor(navier_stokes.control_dofs_ids[0]).reshape(-1, 1),
        torch.arange(d).reshape(-1, 1)],
        dim=1
    )
    values = torch.ones((d,))
    q_guess_matrix = torch.sparse_coo_tensor(
        indices.t(),
        values,
        (navier_stokes.W.dim(), d),
        dtype=torch.float64
    )

    # get an initial guess for the control where only the control DoFs require a gradient
    q_guess_control = torch.zeros((d,), dtype=torch.float64, requires_grad=True)
    # add q_guess_control to the global control tensor q_guess
    q_guess = (q_guess_matrix @ q_guess_control).reshape((1,-1))
    
    # compute the solution U = (v, p) for the initial guess
    v_guess, p_guess = navier_stokes(q_guess)
    # flatten v_guess: v_x and v_y are in the same tensor
    v_guess = v_guess.reshape(1, -1)
    # concatenate velocity and pressure to a single tensor
    U_guess = torch.cat((v_guess, p_guess), dim=1).flatten()
    # save the initial guess as pvd
    navier_stokes.save(U_guess, filename='Results/navier_stokes_initial_guess.pvd')
    assert torch.isnan(U_guess).sum().item() == 0, "U_guess contains NaN values!"
    
    # compute drag value for an arbitrary solution U
    drag_vector = torch.tensor(navier_stokes.drag_vector)

    # prepare optimization
    iter = 0
    MAX_ITER = 50
    optimizer = torch.optim.Rprop([q_guess_control], lr=0.1)
    print("Optimizing the control in the Navier-Stokes problem...")
    print(f"Number of parameters: {q_guess_control.numel()}")

    # optimize the parameters as long drag is not close enough to zero
    loss_history = []
    while iter < MAX_ITER:
        iter += 1
        print(f"Iteration {iter}: Drag = {torch.pow(torch.dot(drag_vector, U_guess), 2)}")

        # zero the gradients
        optimizer.zero_grad()

        # add q_guess_control to the global control tensor q_guess
        q_guess = (q_guess_matrix @ q_guess_control).reshape((1,-1))

        # solve the Navier-Stokes equation
        v_guess, p_guess = navier_stokes(q_guess)
        v_guess = v_guess.reshape(1, -1) # now v_x and v_y are in the same tensor
        # concatenate velocity and pressure to a single tensor
        U_guess = torch.cat((v_guess, p_guess), dim=1).flatten()

        # compute the loss
        # loss = torch.dot(drag_vector, U_guess)
        loss = torch.pow(torch.dot(drag_vector, U_guess), 2)
        loss_history.append(loss.item())

        # backpropagate
        loss.backward()
        # update the parameters
        optimizer.step()

        # apply constraint (non-negativity)
        with torch.no_grad():
            q_guess_control.clamp_(
                min=torch.tensor([0.]*d, dtype=torch.float64),
                max=torch.tensor([100.]*(d-3)+[0.7, 0.7, 100.], dtype=torch.float64) # enforce q(0.3, y) <= 0.7 to avoid failure of NonlinearVariationalSolver (Newton diverges)
            )

# save the optimized flow
navier_stokes.save(U_guess, filename='Results/navier_stokes_optimized.pvd')

# plot optimal control at boundary
colors = ["red", "blue"]
for i, y in enumerate([0., 0.41]):
    X = zip(
        list(navier_stokes.control_dofs[:, 0][navier_stokes.control_dofs[:, 1] == y]),
        list(q_guess_control.detach().numpy()[navier_stokes.control_dofs[:, 1] == y])
    )
    X = sorted(X, key=lambda x: x[0])
    plt.plot(
        [x[0] for x in X], [x[1] for x in X],
        color=colors[i], label=f"Control at y={y}"
    )
plt.xlabel("x")
plt.ylabel("q(x,y)")
plt.title("Control at the control boundary")
plt.legend()
plt.savefig("Results/optimal_control_at_boundary.png")
plt.clf()

# plot loss history
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Loss history")
plt.savefig("Results/loss_history.png")
