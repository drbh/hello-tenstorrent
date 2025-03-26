import torch
import torch.nn.functional as F
from safetensors import safe_open
import ttnn

torch_device = torch.device("cpu")
weights = safe_open("../models/model.safetensors", framework="pt", device="cpu")

#N300 mesh_device
#mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1,2))

class LightweightModule:
    """
    Lightweight version of PyTorch's nn.Module
    """

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Embedding(LightweightModule):
    def __init__(
        self,
        device,
        weights,
    ):
        super().__init__()

        self.weight = ttnn.as_tensor(
            weights.get_tensor("model.embed_tokens.weight"),
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.embedding(x, self.weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x

tt_device = ttnn.open_device(device_id=0)
embedding = Embedding(tt_device, weights)

x = torch.tensor([32], dtype=torch.int32)
x = ttnn.as_tensor(x, device=tt_device)

y = embedding(x)
print(f"✨ {y}") 
