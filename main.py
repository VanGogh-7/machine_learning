import torch
import tensorflow

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"PyTorch Version: {torch.__version__}")
print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
print(f"Tensorflow Version: {tensorflow.__version__}")

if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")

