# Optimal Control with PDEs solved by a Differentiable Solver
This repository uses differentiable PDE solvers for optimal control problems.
Therein, two different approaches are shown: 
1) using [torch_sparse_solve](https://github.com/flaport/torch_sparse_solve) to enable differentiation through sparse direct solvers
2) using [torch-fenics](https://github.com/barkm/torch-fenics) to seamlessly integrate [FEniCS](https://fenicsproject.org/) solvers with [PyTorch](https://github.com/pytorch/pytorch) and especially neural network based control

The first four examples require the [torch_sparse_solve](https://github.com/flaport/torch_sparse_solve) library, along with its [dependencies](https://github.com/flaport/torch_sparse_solve#dependencies), and [matplotlib](https://github.com/matplotlib/matplotlib),
where the last five examples require [torch-fenics](https://github.com/barkm/torch-fenics) with its [dependencies](https://github.com/barkm/torch-fenics#install) including [FEniCS](https://fenicsproject.org/).
## Numerical Examples
### Example 1: 1D Poisson with scalar-valued force (clothesline)

The goal is to determine an approximate force, denoted as $f^{\text{guess}}$, that produces a specific outcome. The qualiy of the force $f^{\text{guess}}$ is measured by the distance between the
corresponding solution of the PDE (see below) $u(f^{\text{guess}})$ and some desired solution $u(f^{\text{true}})$ for a given force $f^{\text{true}}$ which generates it. 
Mathematically, this involves minimizing a tracking-type
cost functional (or loss function) $J$, subject to constraints imposed by the 1D Poission equation on the domain [0,1]. 
More precisely, in the PDE constraint, we solve 
for a function $u \colon (0,1) \to \mathbb{R}$. The overall problem reads as
```math
\begin{align*}
\min_{f^{\text{guess}}}&\quad J(f^{\text{guess}}) := \| u(f^{\text{true}}) - u(f^{\text{guess}})\| \\
\text{ s.t.}&-\partial_x^2 u(x) = f^{\text{guess}}, \\
&\hspace{2em}u(0) = u(1) = 0.
\end{align*}
```
From a mechanical perspective, this example simulates a clothesline that deforms under gravity.
The solver tries then to determine the scalar-valued gravity from observations of the clothesline's deformation.
The solver can be found in [Example 1: 1D_Poisson_scalar_force](./1D_Poisson_scalar_force/main.py).

### Example 2: 1D Poisson with vector-valued force

This example builds on the previous one, but now the gravitational force is vector valued, i.e., it depends on the spatial point $x$. 
Instead of determining a scalar value, we aim to determine a vector-valued gravity field. 
The solver can be found in [Example 2: 1D_Poisson_vector_force](./1D_Poisson_vector_force/main.py).

### Example 3: 1+1D space-time heat equation with vector-valued force and initial condition

In the light of Example 1, the goal here is quite similar. We want to find a control, i.e., a right hand side of a PDE wich leads to a certain outcome. 
Once again, we minimize a tracking-type cost functional regularized with a Tikhonov term, where the constraint is given by the heat equation (a nonstationary PDE).
The overall problem reads as
```math
\begin{align*}
\min_{f^{\text{guess}}}\quad J(f^{\text{guess}}) := \| u&(f^{\text{true}}) - u(f^{\text{guess}})\| + \alpha \| f^{\text{guess}} \| \\
\text{ s.t.}\quad\partial_t u(x,t) -\partial_x^2 u(x,t) &= f^{\text{guess}}(x,t), \\
u(0,t) &= u(1,t) = 0,\\
u(x,0) &= u_0(x).
\end{align*}
```
In this experiment we are looking for a space-time function $u \colon (0,1) \times (0,T) \to \mathbb{R}$. 
The discretization is also performed in a space-time fashion, i.e., not in a time incremental way.

From a mechanical perspective, this example simulates the heat equation and tries to learn the right-hand side 
of the PDE along with the initial conditions. The solver can be found 
in [Example 3: 1+1D_space_time_heat_equation](./1+1D_space_time_heat_equation/main.py).

### Example 4: 1+1D time-stepping heat equation with vector-valued initial condition

This example is similar to the previous one, but instead of using space-time discretization, i.e., one big system matrix,
it employs the backward Euler time stepping scheme. As a result, only the initial condition is optimized in this case, 
not the right-hand side of the PDE. The solver can be found 
in [Example 4: 1+1D_time_stepping_heat_equation](./1+1D_time_stepping_heat_equation/main.py).

### Example 5: 2D Poisson problem for thermal fin with subdomain-dependent heat conductivities

In this example, we consider a 2D Poisson problem describing heat dissipation in a thermal fin, see [Sec. 5.1](https://epubs.siam.org/doi/10.1137/16M1081981).
The PDE constraint in strong form reads as
```math
\begin{align*}
    \sum_{i = 0}^4 - \nabla \cdot (\kappa_i 1_{\Omega_i}(x) \nabla u(x)) &= 0, \qquad \forall x \in \Omega, \\
    \kappa_i \nabla u \cdot n + Bi(u) &= 0, \qquad \forall x \in \Gamma_R \cap \Omega_i, \quad 0 \leq i \leq 4, \\
    \kappa_0 \nabla u \cdot n &= 1. \qquad \forall x \in \Gamma_N.
\end{align*}
```
The parameters that need to be learned are then $\mu^{guess} = (\kappa_0, \kappa_1, \kappa_2, \kappa_3, \kappa_4, Bi) \in \mathbb{R}^6$ and the
loss function is defined as
```math
\begin{align*}
    J(\mu^{\text{guess}}) := \|u(\mu^{\text{true}}) - u(\mu^{\text{guess}})\|_2 + 0.1 \left\|\frac{\mu^{\text{guess}}-\mu^{\text{ref}}}{\mu^{\text{ref}}}\right\|_2,
\end{align*}
```
where $\mu^{ref}$ are some reference coefficients.
The solver can be found in [Example 5: 2D_Poisson_thermal_fin](./2D_Poisson_thermal_fin/main.py).

### Example 6: 2+1D nonlinear heat equation with unknown initial condition

Here, the PDE constraint is given by the nonlinear heat equation $\partial_t u - \Delta u + u^2 = f$ with initial conditions $u(t=0) = u_0$. 
As in the previous nonstationary examples, $u$ represents a space-time function $u\colon [0,1]^2 \times (0,1) \to \mathbb{R}$.
The objective is to determine an initial condition that minimizes a specific loss function, thereby leading to the desired observations.
Unlike in Example 4, we employ a Crank-Nicolson time-stepping scheme here.

The solver can be found in [Example 6: 2+1D_nonlinear_heat_unknown_init_cond](./2+1D_nonlinear_heat_unknown_init_cond/main.py).

### Example 7: 2D Navier-Stokes with boundary control to minimize drag

In this experiment, we consider the stationary Navier-Stokes equations on a 2D rectangular domain with a cylindrical obstacle obstructing the flow. 
The objective is to determine a Neumann boundary control that minimizes the drag coefficient on the boundary of the obstacle. 
In the strong form the PDE is given as
```math
\begin{align*}
    - \nu \Delta v + \nabla p + (v \cdot \nabla)v  &= 0 \qquad \text{in }  \Omega, \\
    \nabla \cdot v &= 0 \qquad \text{in }  \Omega,
\end{align*}
```
where we have to determine the vector-valued velocity $v: \Omega \rightarrow \mathbb{R}^2$ and the scalar-valued 
pressure $p: \Omega \rightarrow \mathbb{R}$ such that the drag coefficient
```math
\begin{align*}
    C_D(v, p) = 500 \int_{\Gamma_{\text{obstacle}}}\sigma(v, p) \cdot n \cdot \begin{pmatrix}
        1 \\ 0
    \end{pmatrix}\ \mathrm{d}x.
\end{align*}
```
is minimized.
The solver can be found in [Example 7: 2D_Navier_Stokes_boundary_control](./2D_Navier_Stokes_boundary_control/main.py).

### Example 8: 2D Fluid-Structure Interaction with parameter estimation for Lamé parameter

The geometrical setting of this experiment is similar to the previous one. 
However, the PDE constraint in this case is governed by a Fluid-Structure Interaction (FSI) model. 
The objective is to determine the Lamé parameters, which are material properties that reproduce a desired observation. 
The solution variables include the vector-valued displacement, the vector-valued velocity, and the scalar-valued pressure. 
The loss function measures the difference between the current displacement and the desired displacement.

The solver can be found in [Example 8: 2D_FSI_Lame_parameters](./2D_FSI_Lame_parameters/main.py).

### Example 9: 2D Poisson with spatially-variable diffusion coefficient combined with neural networks

In our final example, we consider a 2D Poisson problem with a spatially-variable diffusion coefficient that we want to optimize.
The PDE constraint is defined as: Find $u: \Omega \subset \mathbb{R}^2 \rightarrow \mathbb{R}$ such that
```math
\begin{align*}
    - \nabla \cdot (\kappa(x,y) \nabla u(x,y)) &= f, \qquad \forall (x,y) \in \Omega, \\
    u &= 0 \qquad \forall (x,y) \in \partial \Omega.
\end{align*}
```
The true (or desired) diffusion coefficient is given by $\kappa^{true}(x,y) = 1 + 2x + 3y^2$ and the tracking-type loss function
is defined as $J(\kappa^{\text{guess}}) :=  \left\|u(\kappa^{\text{true}}) - u(\kappa^{\text{guess}})\right\|_2$.
Unlike the previous examples, our goal here is to find a network surrogate for $\kappa^{\text{guess}}$. 
To achieve this, we use a fully connected neural network with a single hidden layer containing 20 neurons 
and a sigmoid activation function, resulting in 81 trainable parameters.

The solver can be found in [Example 9: 2D_Poisson_diffusion_neural_networks](./2D_Poisson_diffusion_neural_networks/main.py).

## Authors
- [Denis Khimin](https://github.com/Denis-Khimin)
- [Julian Roth](https://github.com/mathmerizing)
- [Alexander Henkes](https://github.com/ahenkes1)
- [Thomas Wick](https://github.com/tommeswick)




