import os
import torch

def cuda_display():
    os.system("nvidia-smi")

def cuda_init():
    if torch.cuda.is_available():
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\n======================================")
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Compute Capability: {torch.cuda.get_device_capability(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
            print(f"Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 2:.2f} MB")
            print(f"Tensor Cores Support: {torch.cuda.get_device_properties(i).major > 7}")
            print(f"======================================\n")
        torch.cuda.set_device(0)
        torch.cuda.synchronize()

    else:
        print("No GPU available. Using CPU instead.")


if __name__ == "__main__":
    cuda_init()
    cuda_display()