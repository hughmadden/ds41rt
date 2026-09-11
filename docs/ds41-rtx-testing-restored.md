# RTX testing restored with isolated matching libraries

On 2026-09-11, the loaded NVIDIA kernel module was `595.71.05`, while unattended upgrades had replaced the installed compute libraries with `595.91.07`. Ordinary NVML calls and NVIDIA-runtime container startup reported a driver/library mismatch. Both RTX devices were idle.

The exact Ubuntu compute package for the loaded kernel remains available from the Ubuntu security archive:
[libnvidia-compute-595-server 595.71.05-0ubuntu0.26.04.1 amd64](https://security.ubuntu.com/ubuntu/pool/restricted/n/nvidia-graphics-drivers-595-server/libnvidia-compute-595-server_595.71.05-0ubuntu0.26.04.1_amd64.deb).

It was downloaded to `/tmp/ds41-driver-5957105/compute.deb` and extracted with `dpkg-deb -x` into `/tmp/ds41-driver-5957105/root`. No package was installed, system library changed, kernel module reloaded or host rebooted. `LD_LIBRARY_PATH` pointing at the extracted `usr/lib/x86_64-linux-gnu` restored NVML visibility. A local Torch allocation/arithmetic/synchronization check on GPU 0 passed.

Development containers can use the extracted libraries with the ordinary `runc` runtime and explicit GPU device access:

```sh
docker run --rm   --device=/dev/nvidia0 --device=/dev/nvidiactl --device=/dev/nvidia-uvm   -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES=0   -v /tmp/ds41-driver-5957105/root/usr/lib/x86_64-linux-gnu:/driver:ro   -e LD_LIBRARY_PATH=/driver:/usr/local/cuda/lib64   ...
```

The omitted arguments are the fixture/native/model mounts and entry point. GPU 0 is UUID `GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0`. This isolated development setup does not repair the installed driver state or qualify the normal release `--gpus` launch path. If the loaded kernel changes, these libraries must be revalidated before reuse.

## Executed checks

Using `ds41rt-coordinator-dev:latest` and native library `/tmp/ds41-v41-aot/cmake/libds41rt_native.so`:

- Actual cache-bank lifecycle: two cycles of 16 requests, all 44 component leases and stale/foreign/release/reuse guards passed.
- Real router rebinding: 166 byte-exact score/ID/routing comparisons across 40 layers at 1/80 rows and layers 0/20/39 at 4096 rows passed.
- CUDA graph ownership: 240 captures and 480 replays, two independent lanes, 40 owners and changed payloads passed.
- New retained real index-query rebinding test: **48 byte-exact packed-query, scale and head-weight comparisons** across all eight index layers, rows 1/80/4096 and two changed-input cycles passed in **16.62 seconds**. Input addresses remain stable; the second cycle retains each layer's graph handle; invalid-row execution unpublishes output and valid replay recovers.

The router/cache/graph run reports eight Rust tests passing, but three planning tests deliberately skipped because their environment variables were unset; two tests exercised CPU progress guards. The three CUDA tests above actually executed. Planning tests were independently qualified in earlier records.

The index-query check compares captured/rebound execution to fresh uncaptured owners using the same official weights and capacity. It qualifies rebinding/capture equivalence, not a new independent reference arithmetic audit, full index candidate reuse or the combined model layer. Those integration checks are now available to run on RTX and remain open.

All fixture processes exited successfully and no compute process remained after the tests. Logs: `/tmp/ds41-rtx-restored-tests.log` and `/tmp/ds41-index-reuse-rtx-test.log`. Package/library/source hashes are in [the evidence record](ds41-rtx-testing-restored.json).
