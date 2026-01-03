import numpy as np
import torch

from AlzheimersGNN.GNN import (
    BiophysicalDiffusionEvolution,
    PersonalizedSDEEvolution,
    StructuralCausalModel,
    DDGModel,
)


def test_biophysical_shape():
    torch.manual_seed(0)
    B, N = 2, 6
    node_dim = 4
    latent_dim = 3
    evo = BiophysicalDiffusionEvolution(node_dim, latent_dim, cov_dim=3, residual_hidden=16, n_steps=2)
    E = torch.rand(B, N, N)
    E = 0.5 * (E + E.transpose(1, 2))
    H = torch.rand(B, N, node_dim)
    z = torch.rand(B, latent_dim)
    cov = torch.randn(B, 3)
    E2 = evo(E, H, z, cov)
    assert E2.shape == (B, N, N)


def test_personalized_sde_shape():
    torch.manual_seed(1)
    B, N = 3, 5
    node_dim = 3
    latent_dim = 2
    evo = PersonalizedSDEEvolution(node_dim, latent_dim, cov_dim=3, residual_hidden=16, n_steps=3)
    E = torch.rand(B, N, N)
    E = 0.5 * (E + E.transpose(1, 2))
    H = torch.rand(B, N, node_dim)
    z = torch.rand(B, latent_dim)
    cov = torch.randn(B, 3)
    E2 = evo(E, H, z, cov)
    assert E2.shape == (B, N, N)


def test_scm_counterfactual():
    num_nodes = 5
    scm = StructuralCausalModel(num_nodes, phi=0.5, bias=0.0, noise_scale=0.0)
    X0 = np.zeros(num_nodes)
    A = np.eye(num_nodes)
    A_seq = [A for _ in range(3)]
    traj = scm.counterfactual_rollout(X0, A_seq)
    assert traj.shape == (4, num_nodes)
    traj2 = scm.counterfactual_rollout(X0, A_seq, do_pairs=[(0, 1.0)])
    assert traj2.shape == (4, num_nodes)
    assert np.isclose(traj2[0, 0], 1.0)


def test_ddg_run_counterfactual():
    model = DDGModel(in_feats=3, node_dim=8, latent_dim=4)
    scm = StructuralCausalModel(num_nodes=4)
    model.attach_structural_causal_model(scm)
    X0 = np.zeros(4)
    A_seq = [np.eye(4) for _ in range(2)]
    traj = model.run_counterfactual(X0, A_seq, do_pairs=[(2, 0.8)])
    assert traj.shape == (3, 4)
    assert np.isclose(traj[0, 2], 0.8)
