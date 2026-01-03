import numpy as np
import torch
from GNN import GraphODEFunc


def test_graphodefunc_depends_on_adj():
    N = 5
    H = 6
    hidden_dim = H
    funcA = GraphODEFunc(hidden_dim, edge_index=None, adj=torch.eye(N))
    funcB = GraphODEFunc(hidden_dim, edge_index=None, adj=torch.zeros((N, N)))
    h = torch.randn(N, H)
    outA = funcA(0.0, h)
    outB = funcB(0.0, h)
    # outputs should differ when adjacency differs
    assert not np.allclose(outA.detach().cpu().numpy(), outB.detach().cpu().numpy())
