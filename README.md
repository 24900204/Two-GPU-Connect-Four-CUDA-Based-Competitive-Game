# Two-GPU Connect Four — CUDA Capstone

This project implements Connect Four with two CUDA-enabled GPUs as competitors. GPU 0 controls Player X and GPU 1 controls Player O. On each turn, a CUDA kernel evaluates the seven possible columns in parallel and the selected GPU returns the best legal move.

## Code Description
- CUDA kernel evaluates legal moves in parallel.
- `cudaSetDevice()` selects the GPU for each competitor.
- Host/device memory copies transfer the current board.
- Two host threads represent the two GPU competitors.
- A simple heuristic rewards immediate wins, blocks immediate opponent wins, and prefers the center.

## Demonstration / Visualization
The terminal prints the board after every move and reports which GPU selected the move.

## Requirements
- NVIDIA CUDA Toolkit / `nvcc`
- Linux
- At least two CUDA-capable GPUs for the full demonstration

## Build and run
```bash
make
./run.sh
```
or:
```bash
./two_gpu_connect4 --gpu0 0 --gpu1 1
```

The program checks that two different CUDA devices are available. The `results/` folder is only a template: replace it with REAL execution logs and screenshots from your own two-GPU run.
