"""
Physics-Informed Neural Network Models for 1D Wave Equation
"""
import torch
import torch.nn as nn
import config


class PINN(nn.Module):
    """
    Base Physics-Informed Neural Network

    Takes (x, t) as input and outputs u(x,t)
    """

    def __init__(self, hidden_layers, activation='tanh'):
        """
        Initialize PINN

        Args:
            hidden_layers: list of hidden layer sizes
            activation: activation function ('tanh', 'relu', 'sigmoid')
        """
        super(PINN, self).__init__()

        # Build network architecture
        # Input: (x, t) -> 2 dimensions
        # Output: u -> 1 dimension
        layers = []
        input_dim = 2
        output_dim = 1

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_layers[0]))

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())

            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

        # Output layer
        if activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())

        layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for weights"""
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, t):
        """
        Forward pass

        Args:
            x: spatial coordinate (batch_size, 1)
            t: temporal coordinate (batch_size, 1)

        Returns:
            u: predicted solution (batch_size, 1)
        """
        # Concatenate inputs
        inputs = torch.cat([x, t], dim=1)
        u = self.network(inputs)
        return u


def compute_derivatives(u, x, t):
    """
    Compute derivatives of u with respect to x and t using automatic differentiation

    Args:
        u: network output (requires_grad=True)
        x: spatial coordinate (requires_grad=True)
        t: temporal coordinate (requires_grad=True)

    Returns:
        u_x: ∂u/∂x
        u_t: ∂u/∂t
        u_xx: ∂²u/∂x²
        u_tt: ∂²u/∂t²
    """
    # First derivatives
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]

    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]

    # Second derivatives
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0]

    u_tt = torch.autograd.grad(
        u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True, retain_graph=True
    )[0]

    return u_x, u_t, u_xx, u_tt


def wave_equation_residual(u, x, t, c=1.0):
    """
    Compute the residual of the wave equation: ∂²u/∂t² - c²∂²u/∂x²

    Args:
        u: predicted solution
        x: spatial coordinate
        t: temporal coordinate
        c: wave speed

    Returns:
        residual: physics residual
    """
    _, _, u_xx, u_tt = compute_derivatives(u, x, t)

    # Wave equation: ∂²u/∂t² = c²∂²u/∂x²
    # Residual: ∂²u/∂t² - c²∂²u/∂x²
    residual = u_tt - c ** 2 * u_xx

    return residual


class Model1(nn.Module):
    """
    Model 1: Custom architecture with triplet input (x, t, u)

    Architecture:
    - Input: (x, t, u) -> 3 dimensions
    - 5 hidden layers with 100 neurons each
    - Output: (z1, z2) -> 2 dimensions
    - Loss: |z1 - z2| -> should minimize to zero
    """

    def __init__(self, config_dict=None):
        super(Model1, self).__init__()

        # Fixed architecture: 3 -> 100 -> 100 -> 100 -> 100 -> 100 -> 2
        input_dim = 3  # (x, t, u)
        hidden_dim = 100
        n_layers = 5
        output_dim = 2  # (z1, z2)

        # Build network layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # Hidden layers
        for i in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # Output layer (no activation)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for weights"""
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, t, u):
        """
        Forward pass

        Args:
            x: spatial coordinate (batch_size, 1)
            t: temporal coordinate (batch_size, 1)
            u: solution value (batch_size, 1)

        Returns:
            Z: output vector (batch_size, 2) containing (z1, z2)
        """
        # Concatenate inputs
        inputs = torch.cat([x, t, u], dim=1)  # (batch_size, 3)
        Z = self.network(inputs)  # (batch_size, 2)
        return Z

    def get_last_layer_weights(self):
        """Get weights from last hidden layer to output (z1, z2)"""
        # The last layer is self.network[-1] (the final Linear layer)
        # Returns shape: weights (2, 100), bias (2,)
        last_layer = self.network[-1]
        return last_layer.weight.data.cpu().numpy(), last_layer.bias.data.cpu().numpy()


class Model2(PINN):
    """
    Model 2: PINN with different architecture and loss function

    TODO: Customize this model based on your requirements
    - Modify architecture if needed
    - Implement custom loss function in train.py
    """

    def __init__(self, config=None):
        if config is None:
            config = config.MODEL2_CONFIG

        super(Model2, self).__init__(
            hidden_layers=config['hidden_layers'], activation=config['activation']
        )


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING MODELS")
    print("=" * 70)

    # Test Model 1
    print("\nModel 1: Triplet Input (x, t, u) → Scalar Z")
    print("-" * 70)
    model1 = Model1().to(config.DEVICE)
    print(model1)
    print(f"Number of parameters: {sum(p.numel() for p in model1.parameters()):,}")

    print("\nTesting Model 1 forward pass...")
    x = torch.randn(10, 1).to(config.DEVICE)
    t = torch.randn(10, 1).to(config.DEVICE)
    u = torch.randn(10, 1).to(config.DEVICE)

    Z = model1(x, t, u)
    print(f"Input shapes: x={x.shape}, t={t.shape}, u={u.shape}")
    print(f"Output shape: Z={Z.shape}")
    print(f"Sample Z values: {Z[:3].flatten()}")

    # Test weight extraction
    weights, bias = model1.get_last_layer_weights()
    print(f"\nLast layer weights shape: {weights.shape}")
    print(f"Last layer bias: {bias}")

    # Test Model 2
    print("\n" + "=" * 70)
    print("Model 2: Standard PINN (x, t) → u(x,t)")
    print("-" * 70)
    model2 = Model2(config.MODEL2_CONFIG).to(config.DEVICE)
    print(model2)
    print(f"Number of parameters: {sum(p.numel() for p in model2.parameters()):,}")

    print("\nTesting Model 2 forward pass...")
    x = torch.randn(10, 1, requires_grad=True).to(config.DEVICE)
    t = torch.randn(10, 1, requires_grad=True).to(config.DEVICE)

    u = model2(x, t)
    print(f"Input shapes: x={x.shape}, t={t.shape}")
    print(f"Output shape: u={u.shape}")

    # Test derivatives
    print("\nTesting automatic differentiation...")
    residual = wave_equation_residual(u, x, t, c=config.WAVE_SPEED)
    print(f"Residual shape: {residual.shape}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
