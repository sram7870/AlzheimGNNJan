import numpy as np
import torch
from GNN import NeuralODEVAEGNN, node_importance_scores, edge_attribution_via_pa, roi_saliency_over_time, simulate_subject_trajectories


def make_synthetic():
    T = 3
    N = 8
    F = 4
    x = torch.randn(T, N, F, dtype=torch.float32)
    times = torch.tensor(np.linspace(0.0, 2.0, T), dtype=torch.float32)
    return x, times


def test_node_and_edge_importances_and_saliency():
    x, times = make_synthetic()
    model = NeuralODEVAEGNN(in_features=x.shape[-1], hidden_dim=16)
    # PX-based node importances
    node_scores = node_importance_scores(model, x, times, method='px')
    assert isinstance(node_scores, np.ndarray)
    assert node_scores.shape[0] == x.shape[1]
    assert np.all(node_scores >= 0)

    # Edge attributions via PA
    edge_scores = edge_attribution_via_pa(model, x, times)
    assert edge_scores is not None
    assert edge_scores.shape[0] == x.shape[1]
    assert edge_scores.shape[1] == x.shape[1]

    # ROI saliency over time
    sal = roi_saliency_over_time(model, x, times)
    assert sal.shape == (x.shape[0], x.shape[1])
    assert np.all(sal >= 0)


def test_simulate_subject_trajectories():
    x, times = make_synthetic()
    model = NeuralODEVAEGNN(in_features=x.shape[-1], hidden_dim=16)
    sim = simulate_subject_trajectories(model, x, times, n_future=2, dt=1.0)
    assert 'X_hat' in sim and 'H_ts' in sim and 'PA' in sim and 'PX' in sim
    T_sim = x.shape[0] + 2
    assert sim['X_hat'].shape == (T_sim, x.shape[1], x.shape[2])
    assert sim['H_ts'].shape[0] == T_sim
    assert sim['PA'].shape == (x.shape[1], x.shape[1])
    assert sim['PX'].shape == (x.shape[1], x.shape[2])
