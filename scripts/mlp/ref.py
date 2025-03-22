import torch
import torch.nn.functional as F
from safetensors import safe_open

device = torch.device("cpu")
weights = safe_open("model.safetensors", framework="pt", device="cpu")


class MLP:
    def __init__(self, weights, prefix, device=None):
        self.device = device
        self.gate_weight = weights.get_tensor(f"{prefix}.fc1.weight").to(device)
        self.gate_bias = weights.get_tensor(f"{prefix}.fc1.bias").to(device)
        self.down_weight = weights.get_tensor(f"{prefix}.fc2.weight").to(device)
        self.down_bias = weights.get_tensor(f"{prefix}.fc2.bias").to(device)

        # Ensure all weights are bfloat16
        self.gate_weight = self.gate_weight.to(torch.bfloat16)
        self.gate_bias = self.gate_bias.to(torch.bfloat16)
        self.down_weight = self.down_weight.to(torch.bfloat16)
        self.down_bias = self.down_bias.to(torch.bfloat16)

    def forward(self, x):
        output = F.gelu(F.linear(x, self.gate_weight, bias=self.gate_bias))
        output = F.linear(output, self.down_weight, bias=self.down_bias)
        return output


print("Creating model")
model = MLP(weights, f"model.layers.0.mlp", device=device)

hidden_states = torch.ones(1, 1, 1, 2048).to(torch.bfloat16).to(device)

print("✨", hidden_states.sum())

print("Running model")
output = model.forward(hidden_states)
print("✨", output)
print("✨", output.sum())
print("✨", output.shape)
