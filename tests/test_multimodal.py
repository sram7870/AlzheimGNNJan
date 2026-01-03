import numpy as np
import torch
from AlzheimersGNN.GNN import compute_multimodal_adjacency, VAEZEncoder
from AlzheimersGNN.GNN import VAEReconstructor, VAEODEReconstructor


def test_multimodal_shape():
    T = 20
    N = 8
    ts = np.random.randn(T, N).astype(np.float32)
    A = compute_multimodal_adjacency(ts)
    assert A.shape == (N, N)
    assert A.dtype == np.float32


def test_vae_output_shapes():
    B = 2
    T = 5
    N = 6
    D = 16
    latent = 8
    # create dummy node embeddings sequence (B, T, N, D)
    x = torch.randn(B, T, N, D)
    enc = VAEZEncoder(D, latent)
    z, mu, logvar = enc(x)
    assert z.shape == (B, latent)
    assert mu.shape == (B, latent)
    assert logvar.shape == (B, latent)
    # test reconstructor shape
    recon = VAEReconstructor(latent, D, out_feats=4, T_out=T)
    r = recon(z, N)
    assert r.shape == (B, T, N, 4)
    # ODE reconstructor: may fall back to RNN if torchdiffeq missing
    ode_recon = VAEODEReconstructor(latent, D, out_feats=4, T_out=T)
    r2 = ode_recon(z, N)
    assert r2.shape == (B, T, N, 4)
