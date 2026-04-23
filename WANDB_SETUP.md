# W&B Setup Guide for the Research Team

**Companion to `SPEC.md`** — this document covers the operational setup of Weights & Biases for the research infrastructure described in the main spec. It assumes you've read the spec and are now implementing Section 8 (W&B infrastructure).

**Version 1.0** — April 2026

---

## 0. Quick summary

- **Hosting**: self-hosted W&B server on a small VM in Germany, accessed via Tailscale/VPN from everywhere including China.
- **Mode**: all training jobs run `WANDB_MODE=offline`, synced post-hoc by a cron job.
- **Structure**: one entity, projects organized by *purpose* (ablations, champion, final, sandboxes), runs grouped by `config_hash` so seeds aggregate automatically.
- **Setup time**: ~1 week for one engineer including team onboarding.
- **Cost**: ~€30/month for the VM + free W&B self-hosted license for small teams.

---

## 1. Decision: self-hosted vs cloud

Two options. For this team, **self-hosted wins** — the reasoning below is specific to the setup (researchers in China + Germany + air-gapped HPC).

### Option A: W&B cloud + offline sync

Jobs run `WANDB_MODE=offline`, metrics are written to local disk, a periodic `wandb sync` pushes to `wandb.ai`.

**Pros**: zero infrastructure to maintain. W&B cloud has a polished UI, integrated reports, alerts. Free tier or cheap "Teams" tier covers a small team.

**Cons**:
- Syncing from China is slow and sometimes rate-limited.
- Dense 7B training logs can produce multi-GB offline directories; syncing over a flaky link is painful.
- Research-IP-adjacent metadata (configs, loss curves, "which method beats baseline by how much") sits on a third-party's servers.

### Option B: Self-hosted W&B server in Germany (recommended)

Deploy `wandb/local` as a Docker container on a cheap German VM. Team syncs to it over VPN.

**Pros**:
- Fast from anywhere with VPN access; China access is dramatically better than W&B cloud.
- HPC nodes have no internet egress anyway — you must run offline-then-sync regardless, so adding a self-hosted sync target costs nothing extra in ergonomics.
- Research metadata stays on your infrastructure.
- Cheap and low-maintenance once set up.

**Cons**:
- One VM to maintain (patches, disk space, backups).
- Need a real domain with HTTPS.
- Some advanced features (W&B Launch, Automations) require enterprise licensing.

**This guide assumes Option B.** If you choose Option A, skip Sections 2-5 and adapt Sections 6-10 by setting `WANDB_BASE_URL=https://api.wandb.ai`.

---

## 2. Infrastructure setup

### 2.1 Provision the VM

**Minimum spec**: 4 vCPU, 16 GB RAM, 200 GB SSD. Expand disk as runs accumulate (~50-100 GB/year for 100 runs/month).

**Providers**:
- **Hetzner Cloud** (Germany, cheap): ~€15/month for a CX31 instance. Located in Nuremberg or Falkenstein.
- **OVH**: similar pricing.
- **Your own HPC/cluster infra**: if you have a management node with a persistent IP, that also works.

**OS**: Ubuntu 22.04 LTS.

**Also needed**:
- A domain name (e.g., `wandb.yourcompany.com`). Cloudflare, Namecheap, or any registrar.
- A DNS A record → VM's public IP.
- If not using Tailscale: firewall ports 80 and 443 open.

### 2.2 Install Docker

```bash
# On the VM
sudo apt-get update
sudo apt-get install -y \
    docker.io docker-compose-plugin \
    curl ca-certificates

sudo systemctl enable --now docker

# Add your user to the docker group (log out / back in to take effect)
sudo usermod -aG docker $USER
```

Verify: `docker run --rm hello-world`.

### 2.3 Install wandb/local

```bash
# State directory (survives container restarts)
sudo mkdir -p /var/wandb
sudo chown $USER:$USER /var/wandb

docker pull wandb/local:latest

docker run -d \
  --name wandb-local \
  --restart unless-stopped \
  -p 8080:8080 \
  -v /var/wandb:/vol \
  -e HOST=https://wandb.yourcompany.com \
  -e LICENSE=<your-license-key> \
  wandb/local:latest
```

**Obtaining a license**: W&B offers a free self-hosted license for small teams.
- Apply at <https://wandb.ai/site/enterprise-trial> or contact their sales team.
- For a 5-person startup you qualify for the free tier.
- You'll receive the license key by email; paste it into the `LICENSE` environment variable above.

Verify the container is running:
```bash
docker ps | grep wandb-local
docker logs wandb-local | tail -20
```

### 2.4 Set up HTTPS with Caddy

Caddy auto-provisions Let's Encrypt TLS certificates. Minimal config:

```bash
sudo apt-get install -y caddy
```

`/etc/caddy/Caddyfile`:
```
wandb.yourcompany.com {
    reverse_proxy localhost:8080
}
```

```bash
sudo systemctl restart caddy
sudo systemctl enable caddy
```

Visit `https://wandb.yourcompany.com` in a browser — you should see the W&B login page with a valid TLS certificate.

### 2.5 Access control

Two layers, both recommended.

**Layer 1: W&B authentication.** On first visit, create the admin account. Then:
- Go to **System Settings → Authentication**.
- Enable email-domain restrictions (only `@yourcompany.com` can sign up).
- Disable open signup.
- Optionally enable SAML/SSO if you have corporate SSO.

**Layer 2: VPN-only access (strongly recommended).**

Don't expose the VM to the public internet. Put it behind a VPN; restrict ports 80/443 to VPN CIDR.

**For startups without existing VPN infrastructure: use Tailscale.**

On the VM:
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --hostname=wandb-vm
```

Then from your Tailscale admin console:
1. Enable MagicDNS.
2. Optionally: set `wandb.yourcompany.com` to resolve to the Tailscale IP via the TS DNS override.
3. Install Tailscale on every team member's laptop.
4. Once Tailscale is enabled on the VM, close public ports:

```bash
sudo ufw default deny incoming
sudo ufw allow 22/tcp  # SSH (or restrict to Tailscale subnet)
sudo ufw allow in on tailscale0  # allow all Tailscale traffic
sudo ufw enable
```

Tailscale's WireGuard tunnels are fast and reliable from China.

### 2.6 Backups

W&B state lives in `/var/wandb`. Back it up weekly to offsite storage:

```bash
# /etc/cron.weekly/wandb-backup
#!/bin/bash
BACKUP_DIR=/backup/wandb
DATE=$(date +%Y%m%d)
mkdir -p $BACKUP_DIR
tar czf $BACKUP_DIR/wandb-$DATE.tar.gz -C /var wandb
# Rotate: keep 8 weeks
find $BACKUP_DIR -name 'wandb-*.tar.gz' -mtime +56 -delete
# Optional: rsync to S3 / Hetzner Storage Box / etc.
```

```bash
sudo chmod +x /etc/cron.weekly/wandb-backup
```

Don't skip this. Losing a year of experiment metadata because of a disk failure is recoverable from W&B-stored state files; losing it because you never backed up is not.

---

## 3. Entity and project setup

### 3.1 Create the entity

Once logged in as admin:

1. Create an **entity** (W&B UI calls this a "team"): `<yourcompany>-research`. This is the shared namespace all projects live under.
2. Settings → invite team members by email (they sign up, you approve).
3. Grant "Member" role to everyone; reserve "Admin" for 1-2 people.

### 3.2 Create the projects

Inside the entity, create these projects (UI: "Create new project"):

**Ablation projects** — shared, everyone writes, everyone reads:
```
pretrain-ablations-300m      (optional; if you use the 300M smoke-test scale)
pretrain-ablations-600m
pretrain-ablations-1_2b
pretrain-ablations-2_4b
pretrain-ablations-7b
```

**Reference / deliverable projects** — shared, writes restricted to lead by social convention:
```
pretrain-champion
pretrain-final-1_2b
pretrain-final-2_4b
```

**Sandbox projects** — one per researcher, writes by the owner, reads by everyone:
```
sandbox-alice
sandbox-bob
sandbox-zeju
```

**Project privacy**: inside an entity, projects default to "Team" visibility — all entity members can read and write. This is what you want for the ablation projects. For sandboxes, the social convention "don't write to someone else's sandbox" is sufficient in a small team; W&B doesn't enforce per-project write ACLs in the free tier anyway.

### 3.3 API keys

Each team member:
1. Goes to their user settings in W&B → **API Keys** → "Create new key".
2. Copies the key.
3. Puts it in `~/.netrc` on each cluster they'll use:

```
machine wandb.yourcompany.com
  login user
  password <api-key>
```

Or in an environment variable (put in `~/.bashrc` or the cluster env file):
```bash
export WANDB_API_KEY=<key>
```

**For CI and shared-service accounts**: create a dedicated W&B user (e.g., `ci-bot@yourcompany.com`), generate its API key, put it in a secrets manager. Don't use anyone's personal key for CI.

---

## 4. Client configuration

### 4.1 Per-cluster environment variables

Every machine that runs training (H800 nodes, HPC login nodes, researcher laptops) needs these:

```bash
export WANDB_BASE_URL=https://wandb.yourcompany.com
export WANDB_ENTITY=yourcompany-research
```

Put these in the per-cluster env files the launcher sources:

**`launchers/env/h800_cn.env`**:
```bash
# W&B
export WANDB_BASE_URL=https://wandb.yourcompany.com
export WANDB_ENTITY=yourcompany-research
export WANDB_DIR=/scratch/$USER/wandb       # where offline runs are written
export WANDB_MODE=offline                    # offline is the default

# TransformerEngine / CUDA pins (from cluster config)
export TE_VERSION=1.12.0
export CUDA_VERSION=12.4
```

**`launchers/env/hpc_de.env`**:
```bash
export WANDB_BASE_URL=https://wandb.yourcompany.com
export WANDB_ENTITY=yourcompany-research
export WANDB_DIR=$SCRATCH/wandb
export WANDB_MODE=offline                    # HPC has no internet; required

# HPC-specific
export TE_VERSION=1.12.0
export CUDA_VERSION=12.4
```

The launcher sources the appropriate file before invoking the training script:
```bash
source launchers/env/$CLUSTER_NAME.env
```

### 4.2 In the training script

Called from inside `launchers/submit.py` or the training entry point:

```python
import os
import wandb
from omegaconf import OmegaConf

def init_wandb(cfg):
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg._derived.config_hash,                          # seed aggregation
        job_type=cfg.wandb.job_type,                             # ablation | sandbox | ...
        tags=build_tags(cfg),                                     # person, family, scale, cluster, status, month
        config=OmegaConf.to_container(cfg, resolve=True),        # full resolved config
        mode=os.environ.get("WANDB_MODE", "offline"),            # default offline
        dir=os.environ.get("WANDB_DIR", "./wandb"),              # where to write offline runs
        name=f"{cfg.experiment.name}-{cfg.base.scale}-s{cfg.seed}",
        notes=cfg.experiment.description,                         # logged as run notes
    )

def build_tags(cfg):
    from datetime import datetime
    return [
        f"person:{os.environ.get('USER', 'unknown')}",
        f"base_family:{cfg.base.family}",
        f"family:{cfg.experiment.family}",
        f"scale:{cfg.base.scale}",
        f"cluster:{cfg.cluster.name}",
        f"precision:{cfg.precision.default}",
        f"status:{cfg.wandb.status}",                            # candidate | sandbox | ...
        f"regime:{cfg.training_regime.name}",
        f"month:{datetime.utcnow().strftime('%Y-%m')}",
    ]
```

**Rule: only the launcher calls `wandb.init`.** Individual researchers never call it directly from their training scripts. This ensures every run has `config_hash`, `config_diff_from_champion`, and the required tags. The monthly aggregation tools rely on this.

---

## 5. Offline sync

### 5.1 Why offline by default

Three reasons to make offline the default mode for all runs:

1. **HPC compute nodes have no internet.** Attempts at `WANDB_MODE=online` there will fail or silently skip metrics.
2. **China-Germany connectivity is inconsistent.** An online run that intermittently fails to post metrics loses data; an offline run writes locally and syncs when connectivity is available.
3. **Preemption survives.** An offline run interrupted at 3am has all its metrics on disk ready to sync; an online run loses in-flight metrics from the last few minutes.

Trade-off: researchers can't watch metrics in real-time during the run. For most ablation work this doesn't matter (you look at results after the run completes anyway). For final long runs where live monitoring is useful, you can set `WANDB_MODE=online` explicitly, accepting the risks.

### 5.2 The sync script

`tools/sync_wandb.py`:

```python
"""
Scan WANDB_DIR for unsynced offline runs and sync them.
Idempotent: uses a .synced marker file per run.
Safe to run from cron every 15 minutes.
"""
import os
import subprocess
import sys
from pathlib import Path

WANDB_ROOT = Path(os.environ.get("WANDB_DIR", "./wandb"))

def sync_all():
    if not WANDB_ROOT.exists():
        print(f"WANDB_DIR does not exist: {WANDB_ROOT}", file=sys.stderr)
        return 1

    synced = 0
    failed = 0
    for run_dir in sorted(WANDB_ROOT.glob("offline-run-*")):
        sync_flag = run_dir / ".synced"
        if sync_flag.exists():
            continue
        result = subprocess.run(
            ["wandb", "sync", str(run_dir)],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            sync_flag.touch()
            synced += 1
            print(f"synced: {run_dir.name}")
        else:
            failed += 1
            print(f"failed: {run_dir.name}: {result.stderr[:200]}", file=sys.stderr)

    print(f"Done: {synced} synced, {failed} failed.")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(sync_all())
```

### 5.3 Cron setup

On each cluster's login node (or a dedicated sync machine):

```bash
# crontab -e for the researcher's shared account or a sync-bot account
*/15 * * * * cd /home/research/research && source launchers/env/h800_cn.env && python -m tools.sync_wandb >> /var/log/wandb-sync.log 2>&1
```

Every 15 minutes, unsynced runs get pushed. If the link is down, the cron retries on the next tick. No runs are lost.

For HPC clusters, compute nodes don't have internet, but login nodes usually do. Run the cron on the login node against a shared scratch directory.

### 5.4 Troubleshooting sync failures

Common issues:

- **Auth failure**: `~/.netrc` missing the `wandb.yourcompany.com` entry. Add it and chmod 600.
- **Large artifact timeouts**: increase the `timeout=600` in the sync script to `timeout=3600` for runs with many checkpoints logged as artifacts.
- **VPN not up**: verify `tailscale status` (or equivalent); cron doesn't inherit interactive sessions' VPN state, so you may need Tailscale to run as a system service.
- **Disk full on VM**: check `/var/wandb` usage; expand disk or clean old projects.

---

## 6. Operational rules (`docs/wandb_conventions.md`)

Put these rules in the repo. They're social conventions that keep the system clean.

### Rule 1: Where do I log to?

| Scenario | Project | `job_type` |
|---|---|---|
| Debugging / exploring new idea | `sandbox-<yourname>` | `sandbox` |
| Ablation candidate (clean config, reproducible) | `pretrain-ablations-<scale>` | `ablation` |
| Weekly 2.4B promotion gate | `pretrain-ablations-2_4b` | `promotion_gate` |
| Monthly 7B HPC anchor | `pretrain-ablations-7b` | `extrapolation` |
| Final overtrained model | `pretrain-final-<scale>` | `final` |
| Re-running champion baseline | `pretrain-champion` | `champion_baseline` |

Runs with `job_type=sandbox` cannot land in shared `pretrain-ablations-*` projects. The launcher enforces this.

### Rule 2: Use the launcher, not raw `wandb.init`

The launcher fills in `config_hash`, `config_diff_from_champion`, `git_sha`, `megatron_sha`, `patch_set_hash`, `dataset_hash`, and all the tags automatically. Calling `wandb.init` yourself from a training script is only allowed inside the launcher's entry point. Direct calls bypass reproducibility metadata and break the monthly aggregation tools.

### Rule 3: Tag hygiene

Every run carries minimum tags:
- `person:<username>`
- `base_family:<qwen3|llama3|...>`
- `scale:<300m|600m|1_2b|2_4b|7b>`
- `cluster:<h800_cn|h100_de|a100_de|b200_de|hpc_de>`
- `status:<candidate|promoted|deprecated|sandbox>`
- `month:<YYYY-MM>`

The launcher sets these from the config. The team only flips `status` manually when something gets promoted (→ `promoted`) or rejected (→ `deprecated`).

### Rule 4: Don't delete runs

Rejected runs stay in the project with `status:deprecated`. Deleting them loses institutional memory. The end-of-month audit (`tools/validate_ladder.py`) relies on the full run history being present, including failures.

### Rule 5: Use W&B Reports, not screenshots

If you'd take a screenshot of a W&B chart to share in Slack or a document, make it a **report panel** instead. Reports are persistent, linkable, and commentable. The monthly review is a report. The weekly promotion-gate summary is a report.

### Rule 6: Set run names for readability

The launcher sets `name=f"{experiment}-{scale}-s{seed}"`. Don't override this with creative names; the uniform format makes the runs table scannable.

### Rule 7: `config_diff_from_champion` is the primary UI column

In the W&B runs table for `pretrain-ablations-*`, add `config_diff_from_champion` as a visible column. Sort by `eval/hellaswag@20B` (or whatever the primary metric is). Scan the diff column to see what each run actually changed. This is the daily "what's happening" view.

---

## 7. Daily workflow example

Alice is working on a Muon optimizer variant.

**Morning: exploring.**
```bash
source launchers/env/h800_cn.env
python -m launchers.submit \
    base/family=qwen3 base/scale=600m \
    experiment=optim/muon_hybrid \
    training_regime=ablation_40x \
    cluster=h800_cn \
    seed=1 \
    wandb.project=sandbox-alice \
    wandb.status=sandbox
```
Run lands in `sandbox-alice`, offline-written to `/scratch/alice/wandb/offline-run-*`. Cron syncs it to the server ~15 minutes later. Alice opens the sandbox project in W&B, watches the loss curve, iterates.

5-10 sandbox runs over the morning.

**Afternoon: promoting a candidate.**

One config looks promising. Launch the full ladder with seeds:
```bash
python -m launchers.sweep \
    base/family=qwen3 \
    experiment=optim/muon_hybrid \
    ladder=[600m,1_2b] \
    seeds_per_scale=[3,2] \
    training_regime=ablation_20x \
    cluster=h800_cn \
    wandb.project=pretrain-ablations \
    wandb.status=candidate
```

The sweep script expands this into 5 submit calls:
- 3× 600M runs (seeds 1, 2, 3) → `pretrain-ablations-600m`, grouped by `config_hash`
- 2× 1.2B runs (seeds 1, 2) → `pretrain-ablations-1_2b`, grouped by `config_hash`

All 5 runs share the same `config_hash` at the group level (within each scale's project). W&B automatically shows the mean ± std curve.

Alice updates `docs/experiments/muon_hybrid.md` with her hypothesis and links to the W&B group URL.

**Next morning: comparing.**

Alice opens `pretrain-ablations-1_2b`, with the runs table configured as:

| config_diff_from_champion | seeds | eval/hellaswag@20B | eval/loss@20B |
|---|---|---|---|
| `optim.type=muon_hybrid, optim.muon.lr=2e-3` | 2 | 0.438 ± 0.004 | 2.41 ± 0.02 |
| `optim.type=muon_hybrid, optim.muon.lr=3e-3` | 2 | 0.432 ± 0.006 | 2.43 ± 0.03 |
| `optim.type=muon_hybrid` (lr=1e-3) | 2 | 0.428 ± 0.003 | 2.45 ± 0.01 |
| `champion` | 3 | 0.425 ± 0.002 | 2.46 ± 0.01 |

She runs the ladder plot tool to see the 600M→1.2B slope:
```bash
python -m tools.ladder_plot \
    experiment=muon_hybrid \
    base_family=qwen3 \
    metric=eval/hellaswag@20B
```

Saves the output PNG to `docs/experiments/muon_hybrid.md`.

**Weekly: the gate.**

On Friday, the best config goes to the 2.4B weekly gate (runs on HPC if allocation available, else H800):
```bash
python -m launchers.submit \
    base/family=qwen3 base/scale=2_4b \
    experiment=optim/muon_hybrid \
    training_regime=ablation_20x \
    cluster=hpc_de \
    seed=1 \
    wandb.project=pretrain-ablations-2_4b \
    wandb.job_type=promotion_gate \
    wandb.status=candidate
```

**Monthly: the review.**

First Monday of the month:
```bash
python -m tools.gen_monthly_report month=2026-04
```

This builds a W&B report with:
1. Champion baseline at 1.2B and 2.4B (current reference).
2. Candidate runs table, sorted by primary metric.
3. Slope plots for top 3 candidates.
4. 7B anchor (previous month's HPC run) vs extrapolation.
5. Rejected candidates with notes from `docs/experiments/*.md`.

Team reviews the report in a meeting. Decisions:
- Muon hybrid promoted → merge PR, update `docs/experiments/champion_history.md`, status flipped to `promoted` on winning run, new champion kicked off.
- Other candidates: status flipped to `deprecated` if rejected, kept as `candidate` if still being iterated.

---

## 8. Setup timeline (1 week)

### Day 1: Infrastructure

- [ ] Provision VM (Hetzner or equivalent), Ubuntu 22.04, 4vCPU/16GB/200GB.
- [ ] Set up DNS for `wandb.yourcompany.com`.
- [ ] Install Docker, pull `wandb/local`, start container.
- [ ] Confirm container is reachable at `http://<vm-ip>:8080`.

### Day 2: HTTPS + VPN + W&B license

- [ ] Install Caddy, configure reverse proxy, get HTTPS working.
- [ ] Apply for W&B free self-hosted license; apply it to the running container.
- [ ] Install Tailscale on VM; add team laptops to Tailscale.
- [ ] Close public ports; verify VPN-only access works.

### Day 3: Entity and projects

- [ ] Create admin account on W&B.
- [ ] Configure email-domain restriction for signup.
- [ ] Create entity `<yourcompany>-research`.
- [ ] Invite team members; they sign up and you approve.
- [ ] Create all projects (ablations × scales, champion, finals, sandboxes per person).
- [ ] Each team member creates their API key, adds to `~/.netrc` on their workstations.

### Day 4: Client wiring

- [ ] Create `launchers/env/h800_cn.env` and `launchers/env/hpc_de.env` with `WANDB_BASE_URL`, `WANDB_ENTITY`, `WANDB_DIR`, `WANDB_MODE=offline`.
- [ ] Implement `init_wandb(cfg)` and `build_tags(cfg)` in the launcher.
- [ ] Hook into the training script via the launcher.
- [ ] Run a smoke-test training job: launch a 10-step job on h800_cn in a sandbox project, verify it appears in W&B after sync.
- [ ] Verify `config_hash`, `config_diff_from_champion`, and all tags populate correctly.

### Day 5: Sync automation + docs

- [ ] Implement `tools/sync_wandb.py`.
- [ ] Set up cron on h800_cn login node (every 15 minutes).
- [ ] Set up cron on hpc_de login node.
- [ ] Test: launch an offline run, verify auto-sync within 15 minutes.
- [ ] Write `docs/wandb_conventions.md` with the rules from Section 6.
- [ ] Team onboarding: 30-min walkthrough for all researchers.

### Ongoing (first month)

- [ ] Weekly: review new runs in `pretrain-ablations-*`, spot-check tag hygiene.
- [ ] Monthly: run `tools/validate_ladder.py` audit, run `gen_monthly_report.py`.
- [ ] Watch for VM disk usage; expand if approaching 80%.

---

## 9. Troubleshooting

**"My runs aren't showing up in W&B."**
1. Check `WANDB_MODE`: if `online`, the job must have internet to the server; if `offline`, runs only appear after sync.
2. Check `$WANDB_DIR`: offline runs write there.
3. Run `tools/sync_wandb.py` manually; check output.
4. Check `~/.netrc`: must have credentials for `wandb.yourcompany.com`.

**"Sync is failing with auth errors."**
- Regenerate API key in W&B UI, update `~/.netrc`.
- Check file permissions: `~/.netrc` must be `chmod 600`.

**"Sync is timing out on large runs."**
- Increase `timeout=600` to `timeout=3600` in `sync_wandb.py`.
- Check VPN stability.
- Check VM disk free space.

**"China team reports VPN is slow."**
- Verify Tailscale is using DERP relays effectively; check `tailscale netcheck`.
- If bad, try running a Tailscale subnet router or WireGuard directly.
- As a last resort: run a sync-proxy VM inside China that syncs to the Germany VM over a paid commercial VPN.

**"I need to delete a test run I accidentally put in a shared project."**
- Don't. Tag it `status:deprecated`. Deletion loses lineage.

**"The monthly aggregation script is missing runs."**
- `tools/validate_ladder.py` will tell you which runs are missing required metadata (`config_hash`, `patch_set_hash`, etc.).
- Usually: someone bypassed the launcher and called `wandb.init` directly. Fix the root cause; tag the malformed runs `status:deprecated`.

**"VM disk is full."**
- Identify large projects: `du -sh /var/wandb/local/data/*`.
- Archive old finished runs (W&B admin UI supports project archival).
- Expand disk (Hetzner allows online volume expansion).

---

## 10. Cost estimate

For a 5-person team running ~100 ablation runs/month:

| Item | Cost |
|---|---|
| Hetzner CX31 VM (4 vCPU, 16 GB, 200 GB) | ~€15/month |
| Bandwidth (well under Hetzner's 20 TB/month included) | €0 |
| Domain name | ~€10/year |
| Tailscale free tier (up to 100 devices) | €0 |
| W&B self-hosted license (<10 users) | Free |
| Backup storage (Hetzner Storage Box 100 GB) | ~€4/month |
| **Total** | **~€20/month** |

Storage growth: each run's metadata is tens of MB. 100 runs/month × 12 months ≈ 50-100 GB/year. A 200 GB disk comfortably handles 2-3 years. Expand when needed.

---

## 11. References

- W&B self-hosted docs: <https://docs.wandb.ai/guides/hosting>
- W&B offline mode: <https://docs.wandb.ai/guides/technical-faq/general#does-wb-work-offline>
- Tailscale setup: <https://tailscale.com/kb/1017/install>
- Caddy server: <https://caddyserver.com/docs/quick-starts/reverse-proxy>

---

## Appendix: Minimal `sync_wandb.py`

For copy-paste:

```python
#!/usr/bin/env python
"""Sync offline W&B runs to the self-hosted server."""
import os, subprocess, sys
from pathlib import Path

WANDB_ROOT = Path(os.environ.get("WANDB_DIR", "./wandb"))

def main():
    if not WANDB_ROOT.exists():
        print(f"WANDB_DIR not found: {WANDB_ROOT}", file=sys.stderr)
        return 1
    synced, failed = 0, 0
    for run_dir in sorted(WANDB_ROOT.glob("offline-run-*")):
        flag = run_dir / ".synced"
        if flag.exists():
            continue
        res = subprocess.run(
            ["wandb", "sync", str(run_dir)],
            capture_output=True, text=True, timeout=1800,
        )
        if res.returncode == 0:
            flag.touch()
            synced += 1
            print(f"synced: {run_dir.name}")
        else:
            failed += 1
            print(f"failed: {run_dir.name}: {res.stderr[:300]}", file=sys.stderr)
    print(f"Done: {synced} synced, {failed} failed.")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
```

## Appendix: Minimal `build_tags`

```python
import os
from datetime import datetime

def build_tags(cfg) -> list[str]:
    return [
        f"person:{os.environ.get('USER', 'unknown')}",
        f"base_family:{cfg.base.family}",
        f"family:{cfg.experiment.family}",
        f"scale:{cfg.base.scale}",
        f"cluster:{cfg.cluster.name}",
        f"precision:{cfg.precision.default}",
        f"status:{cfg.wandb.status}",
        f"regime:{cfg.training_regime.name}",
        f"month:{datetime.utcnow().strftime('%Y-%m')}",
    ]
```
