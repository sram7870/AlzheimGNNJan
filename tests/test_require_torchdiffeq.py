import pytest
import torch
from GNN import GraphODEFunc, ODEBlock, _try_torchdiffeq


def test_require_torchdiffeq_raises_if_missing():
    # This test is marked xfail when torchdiffeq is available; otherwise it should raise
    requires = True
    func = GraphODEFunc(8)
    blocked = ODEBlock(func, require_torchdiffeq=requires)
    t = torch.tensor([0.0, 1.0], dtype=torch.float32)
    h0 = torch.randn(4, 8)
    if _try_torchdiffeq:
        pytest.xfail('torchdiffeq is present; this test validates behavior when absent')
    else:
        import pytest
        with pytest.raises(RuntimeError):
            _ = blocked(h0, t)
