# Install

```bash
# One-time, on a fresh machine:
curl -LsSf https://astral.sh/uv/install.sh | sh

# One-time, on a new cluster:
cd /lustre/fast/fast/zqiu/
git clone --recurse-submodules https://github.com/zqiu24/slm-research.git

# Per-person env (sibling of slm-research/, ~15-25 min on first install).
# NOT user zqiu? The default cache + TMPDIR break for you (perms + flock) —
# read "uv cache must be on a flock-capable filesystem" below FIRST.
export UV_LINK_MODE=symlink
bash slm-research/install_slm_env.sh <your_name>
cd <your_name>
source .venv/bin/activate
```

The env folder is a sibling of the slm-research clone; editable installs point back at the clone so many envs can share one source tree. See [install_slm_env.sh](install_slm_env.sh) for the full recipe.

## uv cache must be on a flock-capable filesystem (not Lustre)

`install_slm_env.sh` defaults `UV_CACHE_DIR` to `~zqiu/.cache/...` and `TMPDIR`
to `/lustre/fast/fast/zqiu/tmp`. Both fail for anyone who isn't `zqiu`, for two
separate reasons:

- **Permissions:** the default cache lives under zqiu's mode-700 home → `EACCES`.
- **File locking:** our Lustre mounts (`/lustre/...`) are mounted **without
  `flock`**, so uv cannot lock its cache when **building from source**. The first
  `git+https://` dependency (TransformerEngine) dies with:

  ```
  failed to lock .../sdists-v9/.../.lock: Function not implemented (os error 38)
  ```

  This is sneaky: PyPI wheels (torch + the CUDA stack, ~2 GB) install fine
  because they don't take that lock, so the build only blows up ~5 min in at the
  first git dependency — not at the start. (uv locks several cache subdirs during
  a source build — `git-v0`, `sdists-v9`, `built-wheels-v*` — so relocating one
  isn't enough; the *whole* cache has to move off Lustre.)

**Fix — put the entire uv cache and build scratch on a local, flock-capable disk
and use `symlink` mode:**

```bash
export UV_CACHE_DIR=/tmp/uv_cu13_<you>   # local XFS: supports flock; fast for the source builds
export TMPDIR=/tmp/uv_tmp_<you>          # local build scratch
export UV_LINK_MODE=symlink              # installs are instant metadata-only links
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
bash slm-research/install_slm_env.sh <your_name>
```

Don't be tempted by `UV_LINK_MODE=copy` "to make the venv self-contained": with
the venv on Lustre, copying the ~8 GB torch/CUDA stack file-by-file onto Lustre
crawls at ~10–20 MB/min (tens of minutes → hours). `symlink` keeps installs
instant.

**Caveat (the trade-off):** with the cache on node-local `/tmp`, the finished
`.venv` symlinks into `/tmp`. It works great on the build node, but `/tmp` is
node-local and cleared on reboot — so on another node or after a reboot, just
re-run the installer (the warm `/tmp` cache makes a rebuild quick). If you need a
venv that survives without a rebuild, point `UV_CACHE_DIR` at a *persistent*
flock-capable filesystem instead — `$HOME` (cluster-home) works and is shared
across nodes, at the cost of slower many-small-file I/O than `/tmp`.

Check whether any path supports flock before using it as a cache:

```bash
python3 - <<'PY'
import fcntl, os
fd = open("/tmp/.flock_probe", "w")
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); print("flock OK")
except OSError as e:
    print("flock FAIL:", os.strerror(e.errno))
PY
```

## After someone bumps the Megatron pin

`git pull` only brings the parent commit, not the new submodule code:

```bash
cd /lustre/fast/fast/zqiu/slm-research
git pull && git submodule update --init --recursive
```

Or set once: `git config --global submodule.recurse true`.

## Bumping the Megatron pin

See [docs/megatron_pin.md](docs/megatron_pin.md).
