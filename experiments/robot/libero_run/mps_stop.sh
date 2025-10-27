#!/usr/bin/zsh
# stop MPS service

# send quit command to control daemon
echo quit | nvidia-cuda-mps-control

echo "MPS service stopped."
