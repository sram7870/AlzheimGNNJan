import numpy as np
import torch
from GNN import NeuralODEVAEGNN, ADNISubjectDataset, compute_longitudinal_importances, extract_latent_trajectories, compute_time_varying_edge_attention


def make_fake_subject():
    # create two synthetic sessions saved to temp files for ADNIDataset to read
    import tempfile, os, numpy as np
    tmp = tempfile.mkdtemp()
    # two sessions
    for i in range(2):
        T = 3
        N = 6
        F = 4
        arr = np.random.randn(T, N).astype(np.float32)
        p = os.path.join(tmp, f'sub_{i}_ts.npy')
        np.save(p, arr)
    # manifest
    manifest = os.path.join(tmp, 'manifest.csv')
    with open(manifest, 'w') as f:
        f.write('subject_id,session_id,diagnosis,timeseries_path\n')
        f.write(f'sub_0,2010-01-01,CN,{os.path.join(tmp, "sub_0_ts.npy")}\n')
        f.write(f'sub_0,2011-01-01,MCI,{os.path.join(tmp, "sub_1_ts.npy")}\n')
    return manifest, tmp


def test_longitudinal_importances_and_latents():
    manifest, tmp = make_fake_subject()
    ds = ADNISubjectDataset(manifest)
    assert len(ds) == 1
    subj = ds[0]
    model = NeuralODEVAEGNN(in_features=4, hidden_dim=8)
    out = compute_longitudinal_importances(model, subj, method='px')
    assert 'node_scores' in out and 'edge_scores' in out
    assert out['node_scores'].shape[0] == 2
    assert out['edge_scores'].shape[0] == 2

    # test latent extraction and time-varying edge attention for a session
    sess = subj['sessions'][0]
    x_time = torch.tensor(sess['node_features'], dtype=torch.float32)
    times = torch.tensor(np.arange(x_time.shape[0]), dtype=torch.float32)
    H_ts = extract_latent_trajectories(model, x_time, times)
    assert H_ts.shape[0] == x_time.shape[0]
    pa_ts = compute_time_varying_edge_attention(model, x_time, times)
    assert pa_ts.shape[0] == x_time.shape[0]

    # cleanup
    import shutil
    shutil.rmtree(tmp)
