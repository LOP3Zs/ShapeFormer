#!/bin/bash
echo "🔍 Kiểm tra CUDA trên Vast.ai:"

# Kiểm tra GPU hardware
echo "📊 GPU Hardware:"
nvidia-smi

echo -e "\n🐍 Python CUDA check:"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'CUDNN version: {torch.backends.cudnn.version()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo -e "\n🔧 NVCC check:"
nvcc --version

echo -e "\n📦 CUDA environment:"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PATH: $PATH" | grep -o '[^:]*cuda[^:]*'
