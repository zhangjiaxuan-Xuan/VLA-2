#!/usr/bin/zsh
# start MPS service

# point to specific GPU device (if multiple GPUs are present)
export CUDA_VISIBLE_DEVICES=0

# set MPS pipe and log directories (must be writable)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log

if [ ! -d "${CUDA_MPS_PIPE_DIRECTORY}" ] || [ ! -d "${CUDA_MPS_LOG_DIRECTORY}" ]; then
    mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
fi

# start MPS control daemon (in background)
nvidia-cuda-mps-control -d

echo "MPS service started."
echo "PIPE_DIR=${CUDA_MPS_PIPE_DIRECTORY}, LOG_DIR=${CUDA_MPS_LOG_DIRECTORY}"
