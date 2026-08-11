# Physics-Informed Neural Networks (PINN) for 1D Schrödinger Equation

An academically rigorous implementation of a **Physics-Informed Neural Network (PINN)** to solve the time-dependent one-dimensional Schrödinger equation without labeled simulation data.

## Physics Formulation

The 1D time-dependent Schrödinger equation is given by:

$$i \hbar \frac{\partial \psi(x,t)}{\partial t} = -\frac{\hbar^2}{2m} \frac{\partial^2 \psi(x,t)}{\partial x^2} + V(x)\psi(x,t)$$

Decomposing the complex wave function $\psi(x,t) = u(x,t) + i v(x,t)$ into real and imaginary parts yields two coupled real partial differential equations:

$$\frac{\partial u}{\partial t} + \frac{\hbar}{2m} \frac{\partial^2 v}{\partial x^2} - \frac{V(x)}{\hbar} v = 0$$

$$\frac{\partial v}{\partial t} - \frac{\hbar}{2m} \frac{\partial^2 u}{\partial x^2} + \frac{V(x)}{\hbar} u = 0$$

The neural network is trained by minimizing a composite loss function comprising boundary conditions, initial conditions, and the residual physics loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{IC}} + \mathcal{L}_{\text{BC}} + \lambda \mathcal{L}_{\text{physics}}$$

## Repository Structure

```text
pinn-schrodinger-solver/
├── environment.yml       # Conda environment configuration
├── train.py              # Main training pipeline and execution
├── README.md             # Project documentation
└── src/
    ├── __init__.py       # Package marker
    ├── model.py          # Multi-Layer Perceptron architecture
    └── physics.py        # Automatic differentiation & residual loss