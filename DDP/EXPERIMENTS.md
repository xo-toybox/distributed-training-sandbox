# Experiments

```sh
modal volume get ddp-trace-vol / ./ddp_traces --force
tensorboard --logdir ddp_traces
```

`./scripts/profile.sh all --run-name baseline` 
- 20251108-031405-baseline (bs8): 43.81%, 7.76%, 7.32%
- 20251108-044451-bs32: 96.71%, 74.79%, 40.77%
- 20251108-044053-bs64: 95.76% utilization, 84.4% SM efficiency, 49.1% occupancy
- 20251108-043817-bs128: OOM

