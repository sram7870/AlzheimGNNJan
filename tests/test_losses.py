import torch
import numpy as np
from AlzheimersGNN.GNN import compute_losses


def make_dummy_batch(B=2, T=4, N=6, F=4):
    node_features = torch.randn(B, T, N, F)
    adjacency_matrices = torch.randn(B, T, N, N).abs()
    cognitive_scores = torch.randn(B)
    return {'node_features': node_features, 'adjacency_matrices': adjacency_matrices, 'cognitive_scores': cognitive_scores}


def make_dummy_outputs(B=2, N=6, latent_dim=8):
    preds = [{
        'predicted_adjacency': torch.randn(B, N, N).abs(),
        'evolved_adjacency': torch.randn(B, N, N).abs(),
        'predicted_score': torch.randn(B, 1),
        'latent_state': torch.randn(B, latent_dim),
        'latent_mu': torch.zeros(B, latent_dim),
        'latent_logvar': torch.zeros(B, latent_dim),
        'PX': torch.sigmoid(torch.randn(B, N)),
        'PA': torch.sigmoid(torch.randn(B, N, N))
    }]
    return preds


def test_compute_losses_runs():
    batch = make_dummy_batch()
    outputs = make_dummy_outputs()
    loss, metrics = compute_losses(batch, outputs, latent_sequence=None, kl_beta=0.5)
    assert isinstance(loss.item(), float)
    assert 'loss_kl' in metrics
    assert 'loss_importance' in metrics
