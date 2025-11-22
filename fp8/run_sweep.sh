#!/bin/bash
set -euo pipefail

model_name=${1:-}
if [[ -z "$model_name" ]]; then
  echo "Usage: $0 <model_name> [cuda_devices]"
  echo "Example: $0 my-model \"1,2\""
  exit 1
fi

# optional: pass CUDA devices as second arg (default "1,2")
CUDA_DEVICES=${2:-"1,2"}
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

# sweep parameters
seq_lengths=(2048 4096 8192)
precisions=(bf16 fp8)

for seq in "${seq_lengths[@]}"; do
  for prec in "${precisions[@]}"; do
    echo "=== running: model=${model_name} seq=${seq} precision=${prec} (CUDA=${CUDA_VISIBLE_DEVICES}) ==="
    accelerate launch \
      --mixed_precision "$prec" \
      fp8_benchmark.py \
      "$model_name" \
      --sequence-length "$seq" \
      --precision "$prec"
  done
done
