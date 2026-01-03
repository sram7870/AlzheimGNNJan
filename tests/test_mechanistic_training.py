import torch
import numpy as np
from AlzheimersGNN.GNN import DDGModel, BiophysicalDiffusionEvolution


def test_synthesize_sequence_and_loss():
    torch.manual_seed(0)
    B, N = 2, 6
    model = DDGModel(in_feats=3, node_dim=8, latent_dim=4)
    model.enable_biophysical_prior(covariate_dim=3, residual_hidden=16, n_steps=2)

    A0 = torch.rand(B, N, N)
    A0 = 0.5 * (A0 + A0.transpose(1, 2))
    H_seq = [torch.rand(B, N, model.node_dim) for _ in range(3)]
    z_seq = [torch.rand(B, model.latent_dim) for _ in range(3)]
    cov_seq = [torch.randn(B, 3) for _ in range(3)]

    seq = model.synthesize_adjacency_sequence(A0, H_seq, z_seq, cov_seq, steps=3)
    assert len(seq) == 3
    for A in seq:
        assert A.shape == (B, N, N)

    # mechanistic loss
    E = seq[0]
    H = H_seq[0]
    z = z_seq[0]
    cov = cov_seq[0]
    losses = model.compute_mechanistic_loss(E, H, z, covariates=cov)
    assert 'residual_l2' in losses and 'param_l2' in losses and 'total' in losses
    assert losses['total'].item() >= 0.0
