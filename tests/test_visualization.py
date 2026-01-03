import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import matplotlib.pyplot as plt
from GNN import plot_node_saliency_over_time, plot_edge_attribution, plot_subject_trajectories, simulate_subject_trajectories, NeuralODEVAEGNN


def test_plot_node_and_edge_and_trajectories():
    T = 4
    N = 6
    F = 3
    sal = np.abs(np.random.randn(T, N))
    fig, ax = plot_node_saliency_over_time(sal, times=np.linspace(0.0, 3.0, T))
    assert hasattr(fig, 'savefig')

    edge = np.random.randn(N, N)
    fig2, ax2 = plot_edge_attribution(edge)
    assert hasattr(fig2, 'savefig')

    # build a tiny model and simulate
    x = torch.randn(3, N, F, dtype=torch.float32)
    times = torch.tensor(np.linspace(0.0, 2.0, 3), dtype=torch.float32)
    model = NeuralODEVAEGNN(in_features=F, hidden_dim=8)
    sim = simulate_subject_trajectories(model, x, times, n_future=2, dt=1.0)
    fig3, ax3 = plot_subject_trajectories(sim)
    assert hasattr(fig3, 'savefig')

    plt.close('all')
