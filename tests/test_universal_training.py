import numpy as np
import torch

from GNN import universal_train_with_early_stopping, universal_eval


class TinyModel(torch.nn.Module):
    def __init__(self, in_feats=4):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(in_feats, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))

    def forward(self, node_feats, adjs=None):
        # node_feats: (B, T, N, F) -> pool mean over T and N
        x = node_feats.mean(axis=(1, 2))
        return self.net(x).squeeze(-1)


def make_fake_batch(B=4, T=3, N=8, F=4):
    node_feats = np.random.randn(B, T, N, F).astype(np.float32)
    adjs = np.zeros((B, T, N, N), dtype=np.float32)
    y = np.random.randn(B).astype(np.float32)
    return node_feats, adjs, y


def adni_batch_forward(m, batch, device):
    node_feats, adjs, y = batch
    node_feats_t = torch.tensor(node_feats).to(device)
    adjs_t = torch.tensor(adjs).to(device)
    y_t = torch.tensor(y).to(device)
    preds = m(node_feats_t, adjs_t)
    return preds, y_t


def test_universal_trainer_smoke():
    # small synthetic train/val
    train_batches = [make_fake_batch() for _ in range(3)]
    val_batches = [make_fake_batch() for _ in range(2)]
    model = TinyModel(in_feats=4)
    trained, history = universal_train_with_early_stopping(model, train_batches, val_batches, adni_batch_forward, loss_fn=torch.nn.MSELoss(), device='cpu', lr=1e-3, max_epochs=3, patience=2, monitor='val_loss')
    assert 'train_loss' in history and len(history['train_loss']) > 0
    assert 'val_metric' in history and len(history['val_metric']) > 0
    val_res = universal_eval(trained, val_batches, device='cpu', batch_forward_fn=adni_batch_forward)
    assert 'mse' in val_res or 'preds' in val_res
