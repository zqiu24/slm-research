import torch


def test_block_rotation_diagnostics_on_identity_and_rotation():
    from src.diag.rotation_diag import block_rotation_diagnostics

    # R = I (zero skew) -> angle 0, ortho-error 0
    R_eye = torch.eye(4).unsqueeze(0)
    d0 = block_rotation_diagnostics(R_eye)
    assert d0["g_minus_i"][0].item() < 1e-6
    assert d0["ortho_err"][0].item() < 1e-6

    # a real rotation: exp of a skew block -> orthogonal (ortho_err ~ 0), angle > 0
    Q = torch.zeros(1, 4, 4)
    Q[0, 0, 1], Q[0, 1, 0] = 0.5, -0.5
    R = torch.linalg.matrix_exp(Q)
    d1 = block_rotation_diagnostics(R)
    assert d1["ortho_err"][0].item() < 1e-4  # fp32 matrix_exp ortho error ~2e-5
    assert d1["g_minus_i"][0].item() > 0.1
