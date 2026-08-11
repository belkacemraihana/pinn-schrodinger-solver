import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_layers=4, hidden_units=50):
        super(PINN, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(nn.Tanh())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_units, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        output = self.network(inputs)
        u = output[:, 0:1]
        v = output[:, 1:2]
        return u, v