import torch
from torch.cuda.amp import autocast
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.cuda.get_device_name(0))

if torch.cuda.is_available():
    print("CUDA devices:")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    print("Current CUDA device:", torch.cuda.current_device())