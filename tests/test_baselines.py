import torch
import numpy as np
from AlzheimersGNN.GNN import BaselineMLPLSTM, StaticGNNBaseline, GATEncoder


def test_baseline_shapes():
    B=2; T=5; N=6; F=4
    X = torch.randn(B,T,N,F)
    A = torch.abs(torch.randn(B,T,N,N))
    m1 = BaselineMLPLSTM(in_feats=F)
    y1 = m1(X)
    assert y1.shape == (B,1)

    m2 = StaticGNNBaseline(in_feats=F)
    y2 = m2(X, A)
    assert y2.shape == (B,1)

    # test GATEncoder
    gat = GATEncoder(F, 32, num_heads=4, head_dim=8)
    out = gat(X[:,0], A[:,0])
    assert out.shape == (B, N, 32)
