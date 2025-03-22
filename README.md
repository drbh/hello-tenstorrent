# hello tenstorrent

This is a tiny repo that explore running simple computation on Tenstorrent devices.

Additional this repo serves as a starting point to quickly get up and running with Python and ttnn since the library currently require a specific version of Python and specific dependency versions.

```bash
uv run scripts/basic.py

# 2025-02-27 18:46:04.964 | DEBUG    | ttnn.library_tweaks:prepare_dir_as_metal_home:54 - Existing installation of 0.55.0+any detected
# 2025-02-27 18:46:04.988 | DEBUG    | ttnn:<module>:82 - Initial ttnn.CONFIG:
# Config{cache_path=/root/.cache/ttnn,model_cache_path=/root/.cache/ttnn/models,tmp_dir=/tmp/ttnn,enable_model_cache=false,enable_fast_runtime_mode=true,throw_exception_on_fallback=false,enable_logging=false,enable_graph_report=false,enable_detailed_buffer_report=false,enable_detailed_tensor_report=false,enable_comparison_mode=false,comparison_mode_should_raise_exception=false,comparison_mode_pcc=0.9999,root_report_path=generated/ttnn/reports,report_name=std::nullopt,std::nullopt}
# 2025-02-27 18:46:05.535 | DEBUG    | ttnn.decorators:operation_decorator:807 - Should ttnn.pearson_correlation_coefficient be migrated to C++?
# 2025-02-27 18:46:05.536 | DEBUG    | ttnn.decorators:operation_decorator:807 - Should ttnn.Conv1d be migrated to C++?
# 2025-02-27 18:46:05.536 | DEBUG    | ttnn.decorators:operation_decorator:807 - Should ttnn.conv2d be migrated to C++?
# 2025-02-27 18:46:05.536 | DEBUG    | ttnn.decorators:operation_decorator:807 - Should ttnn.unsqueeze_to_4D be migrated to C++?
# 2025-02-27 18:46:05.536 | DEBUG    | ttnn.decorators:operation_decorator:807 - Should ttnn.from_torch be migrated to C++?
# 2025-02-27 18:46:05.536 | DEBUG    | ttnn.decorators:operation_decorator:807 - Should ttnn.to_torch be migrated to C++?
# 2025-02-27 18:46:05.536 | DEBUG    | ttnn.decorators:operation_decorator:807 - Should ttnn.to_device be migrated to C++?
# 2025-02-27 18:46:05.536 | DEBUG    | ttnn.decorators:operation_decorator:807 - Should ttnn.from_device be migrated to C++?
# 2025-02-27 18:46:05.537 | DEBUG    | ttnn.decorators:operation_decorator:807 - Should ttnn.allocate_tensor_on_device be migrated to C++?
# 2025-02-27 18:46:05.537 | DEBUG    | ttnn.decorators:operation_decorator:807 - Should ttnn.copy_host_to_device_tensor be migrated to C++?
# 2025-02-27 18:46:05.537 | DEBUG    | ttnn.decorators:operation_decorator:807 - Should ttnn.deallocate be migrated to C++?
# 2025-02-27 18:46:05.537 | DEBUG    | ttnn.decorators:operation_decorator:807 - Should ttnn.reallocate be migrated to C++?
# 2025-02-27 18:46:05.537 | DEBUG    | ttnn.decorators:operation_decorator:807 - Should ttnn.load_tensor be migrated to C++?
# 2025-02-27 18:46:05.537 | DEBUG    | ttnn.decorators:operation_decorator:807 - Should ttnn.dump_tensor be migrated to C++?
# 2025-02-27 18:46:05.537 | DEBUG    | ttnn.decorators:operation_decorator:807 - Should ttnn.as_tensor be migrated to C++?
# 2025-02-27 18:46:05.538 | DEBUG    | ttnn.decorators:operation_decorator:807 - Should ttnn.conv_transpose2d be migrated to C++?
# 2025-02-27 18:46:05.540 | DEBUG    | ttnn.decorators:operation_decorator:807 - Should ttnn.conv2d be migrated to C++?
# 2025-02-27 18:46:05.540 | DEBUG    | ttnn.decorators:operation_decorator:807 - Should ttnn.Conv1d be migrated to C++?
#                  Device | INFO     | Opening user mode device driver
#   Detecting chips (found 2)                                                                                                                                           
# 2025-02-27 18:46:05.600 | INFO     | SiliconDriver   - Opened PCI device 0; KMD version: 1.31.0, IOMMU: disabled
# 2025-02-27 18:46:05.601 | INFO     | SiliconDriver   - Detected PCI devices: [0]
# 2025-02-27 18:46:05.601 | INFO     | SiliconDriver   - Using local chip ids: {0} and remote chip ids {1}
# 2025-02-27 18:46:05.612 | INFO     | SiliconDriver   - Software version 6.0.0, Ethernet FW version 6.10.0 (Device 0)
# 2025-02-27 18:46:05.612 | INFO     | SiliconDriver   - Software version 6.0.0, Ethernet FW version 6.10.0 (Device 1)
#                   Metal | INFO     | Initializing device 0. Program cache is NOT enabled
#                   Metal | INFO     | AI CLK for device 0 is:   1000 MHz
#            BuildKernels | INFO     | GIT_COMMIT_HASH not found
#                  Always | FATAL    | Cannot get the device from a tensor with host storage
#                   ... TODO ... (fix this)
#                   Metal | INFO     | Closing device 0
#                   Metal | INFO     | Disabling and clearing program cache on device 0
#                  Device | INFO     | Closing user mode device drivers
```


## Running with patched tt_transformer

```bash
uv sync
make setup-patch
export PYTHONPATH=/root/hello-tenstorrent/tt-metal-vendored/
# or maybe export PYTHONPATH=/root/hello-tenstorrent/tt-metal-vendored/models/
uv run scripts/ttransform.py
```

Compare/explore torch and ttnn simple mlp implementation

```bash
uv run scripts/mlp/ttmlp.py
# Creating model
# ✨ ttnn.Tensor([[[[2048.00000]]]], shape=Shape([1, 1, 1, 1]), dtype=DataType::BFLOAT16, layout=Layout::TILE)
# Running model
# ✨ ttnn.Tensor([[[[-0.05713, -0.40820,  ...,  0.35156,  0.51562]]]], shape=Shape([1, 1, 1, 2048]), dtype=DataType::BFLOAT16, layout=Layout::TILE)
# ✨ ttnn.Tensor([[[[52.50000]]]], shape=Shape([1, 1, 1, 1]), dtype=DataType::BFLOAT16, layout=Layout::TILE)
# ✨ ttnn.Shape([1, 1, 1[32], 2048])
```

```bash
uv run scripts/mlp/ref.py
# Creating model
# ✨ tensor(2048., dtype=torch.bfloat16)
# Running model
# ✨ tensor([[[[-0.0461, -0.4004, -0.9844,  ...,  0.1709,  0.3594,  0.5352]]]],
#        dtype=torch.bfloat16)
# ✨ tensor(49.5000, dtype=torch.bfloat16)
# ✨ torch.Size([1, 1, 1, 2048])
```
