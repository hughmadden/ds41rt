# io_uring for coordinator experiments

`seccomp-io-uring.json` is the Moby default profile from commit
[`61eaf32614c7c71b60bd8927d3e6a4ffc8ff1f31`](https://github.com/moby/profiles/blob/61eaf32614c7c71b60bd8927d3e6a4ffc8ff1f31/seccomp/default.json),
with one additional unconditional allow rule for `io_uring_setup`,
`io_uring_enter`, and `io_uring_register`. It retains the upstream default-deny
policy and other rules. This is a pinned upstream profile, not an export of the
installed Docker daemon's built-in profile. Review the base when updating Docker.
The upstream profile is Copyright The Moby Authors, Apache-2.0; its
[license](seccomp-io-uring.LICENSE) is included alongside the profile.

Add this argument when creating an experimental container from the repo root:

```bash
--security-opt "seccomp=$PWD/docker/seccomp-io-uring.json"
```

An existing container must be recreated to change its seccomp policy; restarting
it retains the original policy. No host sysctl change is needed on raptor:
`/proc/sys/kernel/io_uring_disabled` is already `0`. No `--privileged`, extra
capabilities, or global Docker configuration change is required.

A disposable x86-64 capability check (no model load):

```bash
docker run --rm --network none \
  --security-opt "seccomp=$PWD/docker/seccomp-io-uring.json" \
  --entrypoint python3 ds41rt-coordinator-dev:latest -c '
import ctypes, os
libc = ctypes.CDLL(None, use_errno=True)
params = ctypes.create_string_buffer(120)
fd = libc.syscall(425, 8, ctypes.byref(params))
assert fd >= 0, os.strerror(ctypes.get_errno())
os.close(fd)
print("io_uring setup succeeds")
'
```

On raptor/Docker 29.1.3 this succeeded with the custom profile; the current
serving container's default profile returned EPERM. This only verifies ring
creation. Additional disposable-container checks completed 432 official Engram
rows byte-exactly with ordinary io_uring, registered buffers, and io_uring followed
by mmap gather. Registration uses the existing unlimited memlock setting.
This does not switch the runtime's Engram implementation to io_uring.
The live APIs continue to use their selected mmap/prefetch/gather implementation.

Docker documents custom profiles in its
[seccomp guide](https://docs.docker.com/engine/security/seccomp/).
