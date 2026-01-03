import tempfile
import os
import numpy as np
import torch
from GNN import NeuralODEVAEGNN, train_longitudinal_vae


def make_fake_subject(tmp, name='s1', T=3, N=8, F=4):
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    times = np.linspace(0.0, 1.0, T).astype(np.float32)
    x = np.random.randn(T, N, F).astype(np.float32)
    edges = np.eye(N, dtype=np.float32)
    label = np.array([1.0], dtype=np.float32)
    np.save(os.path.join(d, 'times.npy'), times)
    np.save(os.path.join(d, 'node_features.npy'), x)
    np.save(os.path.join(d, 'adjacency.npy'), np.stack([edges for _ in range(T)], axis=0))
    np.save(os.path.join(d, 'label.npy'), label)


def test_vae_forward_and_train():
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_subject(tmp, 's0')
        # load subject directly from files
        x = torch.tensor(np.load(os.path.join(tmp, 's0', 'node_features.npy')))
        times = torch.tensor(np.load(os.path.join(tmp, 's0', 'times.npy')))
        adjs = [torch.tensor(a) for a in np.load(os.path.join(tmp, 's0', 'adjacency.npy'))]
        y = torch.tensor(np.load(os.path.join(tmp, 's0', 'label.npy')))
        model = NeuralODEVAEGNN(in_features=x.shape[-1], hidden_dim=16)
        res = model(x, times, adjs, return_dict=True)
        assert 'recon' in res and 'kl' in res and 'logits' in res and 'PX' in res
        # PX shape check
        PX = res['PX']
        assert PX.shape[0] == x.shape[1] and PX.shape[1] == x.shape[2]
        # small training step with reg weights
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = [{'x': x, 'times': times, 'adjs': adjs, 'y': y}]
        losses = train_longitudinal_vae(model, [batch[0]], optim, device='cpu', beta=0.5, px_sparsity_weight=0.1, px_entropy_weight=0.1)
        assert 'recon' in losses and 'kl' in losses and 'sparsity' in losses and 'entropy' in losses
