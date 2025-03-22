import torch
import ttnn
from safetensors import safe_open

device = ttnn.open_device(device_id=0)
weights = safe_open("model.safetensors", framework="pt", device="cpu")


class MLP:
    def __init__(self, weights, prefix, device=None):
        self.device = device
        self.gate_weight = weights.get_tensor(f"{prefix}.fc1.weight")
        self.gate_bias = weights.get_tensor(f"{prefix}.fc1.bias")
        self.down_weight = weights.get_tensor(f"{prefix}.fc2.weight")
        self.down_bias = weights.get_tensor(f"{prefix}.fc2.bias")

        # Ensure all weights are bfloat16 (already the case)
        self.gate_weight = self.gate_weight.to(torch.bfloat16)
        self.gate_bias = self.gate_bias.to(torch.bfloat16)
        self.down_weight = self.down_weight.to(torch.bfloat16)
        self.down_bias = self.down_bias.to(torch.bfloat16)

        # transpose the weights
        self.gate_weight = self.gate_weight.T
        self.down_weight = self.down_weight.T

        # make them ttnn tensors
        self.gate_weight = ttnn.from_torch(
            self.gate_weight, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.gate_bias = ttnn.from_torch(
            self.gate_bias, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.down_weight = ttnn.from_torch(
            self.down_weight, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.down_bias = ttnn.from_torch(
            self.down_bias, layout=ttnn.TILE_LAYOUT, device=device
        )

    def forward(self, x):
        output = ttnn.linear(
            x, self.gate_weight, bias=self.gate_bias, activation="gelu"
        )
        output = ttnn.linear(output, self.down_weight, bias=self.down_bias)
        return output


print("Creating model")
model = MLP(weights, f"model.layers.0.mlp", device=device)

hidden_states = torch.ones(1, 1, 1, 2048).to(torch.bfloat16)
hidden_states = ttnn.from_torch(hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
print("✨", ttnn.sum(hidden_states))

print("Running model")
output = model.forward(hidden_states)

print("✨", output)
print("✨", ttnn.sum(output))
print("✨", output.shape)
