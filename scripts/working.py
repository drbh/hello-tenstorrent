import torch
import ttnn

input_tensor = torch.Tensor([[ 330.,  345.,  360.]]).to(torch.float16)

reference_sum = input_tensor.sum().item()

# now open the tenstorrent device and sum
device = ttnn.open_device(device_id=0)
ttnn_tensor = ttnn.from_torch(input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
ttnn_sum = ttnn.sum(ttnn_tensor)
ttnn_sum_torch_output = ttnn.to_torch(ttnn_sum).item()
ttnn.close_device(device)

print(f"\nInput tensor: {input_tensor}\n")
print("Ref:\t", reference_sum)
print("TTNN:\t", ttnn_sum_torch_output)

# Input tensor: tensor([[330., 345., 360.]])