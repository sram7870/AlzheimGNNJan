import numpy as np
import torch
from GNN import ODEBlock


class LinearODEFunc(torch.nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = torch.tensor(A, dtype=torch.float32)

    def forward(self, t, h):
        # h: (N, H) -> dh/dt = h @ A
        return torch.matmul(h, self.A)


def test_ode_integration_consistency():
    N = 4
    H = 5
    # base linear operator
    np.random.seed(0)
    A = 0.5 * np.random.randn(H, H).astype(np.float32)
    func = LinearODEFunc(A)
    block = ODEBlock(func, method='rk4')

    # initial state
    h0 = torch.randn(N, H, dtype=torch.float32)

    # dense sampling
    t_dense = np.linspace(0.0, 1.0, 201)
    t_dense_t = torch.tensor(t_dense, dtype=torch.float32)
    H_dense = block(h0, t_dense_t)  # (T_dense, N, H)

    # sparse sampling (subset of dense times)
    t_sparse = np.array([0.0, 0.05, 0.23, 0.5, 0.78, 1.0], dtype=np.float32)
    t_sparse_t = torch.tensor(t_sparse, dtype=torch.float32)
    H_sparse = block(h0, t_sparse_t)

    # sample dense at sparse times (nearest indices)
    idx = [int(np.argmin(np.abs(t_dense - ts))) for ts in t_sparse]
    H_dense_sampled = H_dense[idx]  # (len(t_sparse), N, H)

    # assert close within reasonable tolerance
    assert np.allclose(H_sparse.detach().cpu().numpy(), H_dense_sampled.detach().cpu().numpy(), rtol=1e-3, atol=1e-4)


def test_solver_tolerances_no_error():
    # ensure that rtol/atol params can be set without error
    N = 2
    H = 3
    A = np.random.randn(H, H).astype(np.float32) * 0.1
    func = LinearODEFunc(A)
    # try a strict tolerance
    block = ODEBlock(func, method='rk4', rtol=1e-8, atol=1e-10)
    h0 = torch.randn(N, H, dtype=torch.float32)
    t = torch.tensor([0.0, 0.2, 0.5, 1.0], dtype=torch.float32)
    out = block(h0, t)
    assert out.shape[0] == len(t)
