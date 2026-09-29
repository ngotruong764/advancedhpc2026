from numba import cuda

# Get the current active GPU device
device = cuda.get_current_device()

# Get the name of the device
device_name = device.name

print(f"Device Name: {device_name}")

# Iterate through all available GPUs
for gpu in cuda.gpus:
    print(f"GPU ID: {gpu.id}")

    # Core information
    print(f"Name: {gpu.name}")

    mp_count = gpu.MULTIPROCESSOR_COUNT
    print(f"Multiprocessor Count (SMs): {mp_count}")
    print(f"Compute capability: {gpu.compute_capability}")
    print(f"Estimated CUDA Core Count: {mp_count*64}")

    free_mem, total_mem = cuda.current_context().get_memory_info()
    total_mem_gb = total_mem / (1024 ** 3)
    print(f"Total Memory Size: {total_mem_gb:.2f} GB ({total_mem} bytes)")