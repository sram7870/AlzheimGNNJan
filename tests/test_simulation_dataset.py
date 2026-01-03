import importlib
import types

# Import module try canonical package name first, fallback to top-level module
try:
    GNN = importlib.import_module('AlzheimersGNN.GNN')
except Exception:
    GNN = importlib.import_module('GNN')


def test_simulate_returns_metadata():
    res = GNN.simulate_subject_sequence(num_regions=8, num_timepoints=4, seed=0)
    assert isinstance(res, tuple) and len(res) == 4
    node_feats, adjs, times, meta = res
    assert node_feats.shape[0] == 4 and adjs.shape[0] == 4
    # metadata should contain expected keys
    for k in ('degenerate_regions', 'affected_regions', 'labels', 'disease_progression'):
        assert k in meta


def test_simulated_dataset_instantiation():
    ds = GNN.SimulatedDDGDataset(num_subjects=3, num_regions=8, num_timepoints=4, seed=1)
    assert len(ds) == 3
    sample = ds[0]
    assert 'node_features' in sample and 'adjacency_matrices' in sample and 'metadata' in sample and 'cognitive_score' in sample
