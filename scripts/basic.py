import torch
import ttnn

device = ttnn.open_device(device_id=0)

torch_tensor = torch.rand(32, 32, dtype=torch.float32)
ttnn_tensor_cpu = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)

# simply multiply the tensor by itself
ttnn_tensor = ttnn.mul(ttnn_tensor_cpu, ttnn_tensor_cpu)

ttnn.close_device(device)

print("🔥 Success!")