import numpy as np
import torch
from GNN import NeuralODEGNN, NeuralODEVAEGNN


def make_ring_adj(N):
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        adj[i, (i+1) % N] = 1.0
        adj[(i+1) % N, i] = 1.0
    return adj


def test_neuralodegnn_irregular_times():
    N = 6
    F = 3
    T = 3
    # irregular times
    times = torch.tensor([0.0, 0.4, 2.5], dtype=torch.float32)
    x = torch.randn(T, N, F, dtype=torch.float32)

    model = NeuralODEGNN(in_channels=F, hidden_dim=16)
    adj = torch.tensor(make_ring_adj(N))
    model.set_edge_index(edge_index=None, adj=adj)

    # ensure forward works with irregular times
    out = model(x, times)
    assert torch.is_tensor(out)

    # shorter sequence (missing visits) should work
    times2 = torch.tensor([0.0, 3.0], dtype=torch.float32)
    x2 = x[:2]
    out2 = model(x2, times2)
    assert torch.is_tensor(out2)

    # probabilities
    p = model.predict_proba(x, times)
    assert (p >= 0).all() and (p <= 1).all()


def test_vaegnn_simulation_irregular():
    N = 5
    F = 4
    T = 2
    times = torch.tensor([0.0, 1.6], dtype=torch.float32)
    x = torch.randn(T, N, F, dtype=torch.float32)

    model = NeuralODEVAEGNN(in_features=F, hidden_dim=12)
    sim_times = [0.0, 0.5, 2.2, 3.7]
    sim = model.forward(x, times, adjs_list=None, return_dict=True)
    # call simulate_time_evolution convenience function (imported above if available)
    from GNN import simulate_time_evolution
    res = simulate_time_evolution(model, x, times, sim_times)
    assert 'X_hat' in res and 'H_ts' in res
    assert res['X_hat'].shape[0] == len(sim_times)
    assert res['H_ts'].shape[0] == len(sim_times)
