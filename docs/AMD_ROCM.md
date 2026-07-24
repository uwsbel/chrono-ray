# ChronoRay on AMD GPUs (ROCm) and Ray

This note is for **PyTorch ROCm** + **Ray Tune** on AMD Instinct (or other ROCm) hosts. It complements the upstream PyTorch ROCm and Ray installation guides.

## Device selection

Use **`ROCR_VISIBLE_DEVICES`** to select AMD GPUs. Do **not** use `CUDA_VISIBLE_DEVICES` on ROCm-only machines.

## Ray 2.45+ and HIP

Before **`import ray`** (or before any library that imports Ray transitively), set:

```text
RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
```

Setting **both** is forward-compatible across Ray versions that renamed or split these flags.

The **`ChronoRay`** package applies these defaults when it is imported (see `ChronoRay.rocm_ray_env.prepare_rocm_ray_env`). You may also call `prepare_rocm_ray_env()` explicitly at the very top of your own driver script.

Optional (helps some ROCm + accelerator layouts):

```text
RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=1
```

## `prepare_rocm_ray_env()` behavior

The helper also clears **`ROCR_VISIBLE_DEVICES`** and **`CUDA_VISIBLE_DEVICES`** before the first Ray import so Ray does not assert against a pre-set ROCm device list. After **`ray.init`** or **`ray start`**, you may need to **re-export** `ROCR_VISIBLE_DEVICES` for workloads that use **RCCL** or other libraries that read that variable directly.

## Ray Cluster / workers

When bringing up a Ray cluster on a multi-GPU node, use:

```bash
ray start --head --num-gpus=8
```

Adjust **`--num-gpus`** to match visible devices. If you omit **`--num-gpus`**, Ray may report **zero** GPUs even when ROCm devices are present.

## Multi-node

Reconcile **Ray’s** GPU visibility with **RCCL** / **distributed** jobs: document which process sets `ROCR_VISIBLE_DEVICES` (launcher vs worker) and test with a minimal `torch.distributed` or RCCL health check independent of Tune when debugging.

## Further reading

- [PyTorch ROCm install matrix](https://pytorch.org/get-started/locally/)
- [Ray cluster docs](https://docs.ray.io/en/latest/cluster/getting-started.html)
