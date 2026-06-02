"""Per-block rotation diagnostics for POET Muon-on-Q (Stage 2).

Given a batch of block rotation matrices R = f(Q) ((num_blocks, b, b)), report:
  - g_minus_i  = ||R - I||_F  : realized per-block rotation magnitude (angle proxy;
    the spec's theta-calibration check).
  - ortho_err  = ||R R^T - I||_F : orthogonality-approximation error (validity gate
    for the cayley no-reset arms, where the Cayley-Neumann series can leave its
    convergence regime).
"""

from __future__ import annotations

import torch


def block_rotation_diagnostics(R: torch.Tensor) -> dict[str, torch.Tensor]:
    if R.dim() == 2:
        R = R.unsqueeze(0)
    R = R.to(torch.float32)
    b = R.shape[-1]
    eye = torch.eye(b, device=R.device, dtype=R.dtype)
    g_minus_i = torch.linalg.matrix_norm(R - eye, ord="fro", dim=(-2, -1))
    ortho_err = torch.linalg.matrix_norm(R @ R.transpose(-2, -1) - eye, ord="fro", dim=(-2, -1))
    return {"g_minus_i": g_minus_i, "ortho_err": ortho_err}
