# Install

```bash
# One-time, on a fresh machine:
curl -LsSf https://astral.sh/uv/install.sh | sh

# One-time, on a new cluster:
cd /lustre/fast/fast/zqiu/
git clone --recurse-submodules https://github.com/zqiu24/slm-research.git

# Per-person env (sibling of slm-research/, ~15-25 min on first install):
export UV_LINK_MODE=symlink
bash slm-research/install_slm_env.sh <your_name>
cd <your_name>
source .venv/bin/activate
```

The env folder is a sibling of the slm-research clone; editable installs point back at the clone so many envs can share one source tree. See [install_slm_env.sh](install_slm_env.sh) for the full recipe.

## After someone bumps the Megatron pin

`git pull` only brings the parent commit, not the new submodule code:

```bash
cd /lustre/fast/fast/zqiu/slm-research
git pull && git submodule update --init --recursive
```

Or set once: `git config --global submodule.recurse true`.

## Bumping the Megatron pin

See [docs/megatron_pin.md](docs/megatron_pin.md).
