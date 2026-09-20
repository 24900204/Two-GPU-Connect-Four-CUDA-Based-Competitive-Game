#!/usr/bin/env bash
set -e
make
nvidia-smi -L
./two_gpu_connect4 --gpu0 0 --gpu1 1
