# poet_lie_head

POET × Pion Lie-algebra momentum with **head-aligned attention rotation**.

Attention projections (q/k/v/o) rotate their head-structured side with one
`head_dim`-sized block per head, a fixed identity permutation (block j = head j,
never resampled), and no cross-head mixing. The residual side stays a normal POET
rotation (`block_count=1` dense in dev). Requires unfused q/k/v.

Ablate against `optim/poet_lie` (dense both sides) and `optim/poet_lie_rms`.
