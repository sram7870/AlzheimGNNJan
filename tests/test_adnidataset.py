import os
import tempfile
import numpy as np
import csv
import scipy.io
import sys
import types

# Create a working pandas stub for tests BEFORE any imports
class MockGroupBy:
    def __init__(self, data, groupby_col):
        self.data = data
        self.groupby_col = groupby_col
    
    def __iter__(self):
        groups = {}
        indices_map = {}
        for i, row in enumerate(self.data):
            key = row[self.groupby_col]
            if key not in groups:
                groups[key] = []
                indices_map[key] = []
            groups[key].append((i, row))
            indices_map[key].append(i)
        for key in sorted(groups.keys()):
            rows = [r[1] for r in groups[key]]
            df = MockDF(rows)
            df.index = MockIndex(indices_map[key])
            yield key, df

class MockDF:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.index = MockIndex(list(range(len(self.rows))))
        if self.rows and isinstance(self.rows[0], dict):
            self.columns = list(self.rows[0].keys()) if self.rows else []
        else:
            self.columns = []
    
    def groupby(self, col):
        return MockGroupBy(self.rows, col)
    
    def sort_values(self, by, ascending=True):
        # Sort by column value
        sorted_with_idx = sorted(enumerate(self.rows), key=lambda x: x[1].get(by, ''))
        sorted_rows = [r[1] for r in sorted_with_idx]
        new_df = MockDF(sorted_rows)
        new_df.index = MockIndex([self.index.values[i] for i, _ in sorted_with_idx])
        return new_df
    
    def copy(self):
        new_df = MockDF([r.copy() for r in self.rows])
        new_df.index = MockIndex(self.index.values.copy())
        return new_df
    
    def __len__(self):
        return len(self.rows)
    
    def __contains__(self, col):
        return col in self.columns
    
    def __iter__(self):
        return iter(self.rows)
    
    def __getitem__(self, col):
        """Return a MockSeries for column access or slice-like behavior."""
        if isinstance(col, str):
            values = [r.get(col) for r in self.rows]
            return MockSeries(dict(enumerate(values)))
        return self
    
    def __setitem__(self, col, values):
        """Set a column with new values."""
        if not self.columns or col not in self.columns:
            self.columns.append(col)
        for i, row in enumerate(self.rows):
            if isinstance(values, (list, np.ndarray)):
                row[col] = values[i]
            else:
                row[col] = values
    
    def iterrows(self):
        for i, r in enumerate(self.rows):
            yield i, MockSeries(r)

class MockIndex:
    def __init__(self, values):
        self.values = values
    
    def tolist(self):
        return self.values

class MockSeries:
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, key):
        return self.data.get(key)
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def apply(self, func):
        """Apply a function to each value in the series."""
        result = [func(v) for v in self.data.values()]
        return MockSeries(dict(enumerate(result)))

# Install pandas stub FIRST, before importing anything else
if 'pandas' not in sys.modules:
    pd_stub = types.ModuleType('pandas')
    pd_stub.__version__ = '1.5.0'
    pd_stub.DataFrame = MockDF
    pd_stub.read_csv = lambda path, **kw: MockDF()
    pd_stub.to_datetime = lambda x: x
    pd_stub.isna = lambda x: False
    sys.modules['pandas'] = pd_stub

from GNN import load_timeseries_file, ADNIDataset, ADNISubjectDataset



def _write_npy(tmpdir, name, arr):
    p = os.path.join(tmpdir, name)
    np.save(p, arr)
    return p


def _write_npz(tmpdir, name, arr, key='timeseries'):
    p = os.path.join(tmpdir, name)
    np.savez(p, **{key: arr})
    return p


def _write_csv(tmpdir, name, arr):
    p = os.path.join(tmpdir, name)
    np.savetxt(p, arr, delimiter=',')
    return p


def _write_mat(tmpdir, name, arr, key='timeseries'):
    p = os.path.join(tmpdir, name)
    scipy.io.savemat(p, {key: arr})
    return p


def test_load_timeseries_file_formats():
    with tempfile.TemporaryDirectory() as td:
        arr = np.random.randn(8, 5).astype(np.float32)
        npy = _write_npy(td, 'a.npy', arr)
        npz = _write_npz(td, 'b.npz', arr)
        csvf = _write_csv(td, 'c.csv', arr)
        mat = _write_mat(td, 'd.mat', arr)

        for p in [npy, npz, csvf, mat]:
            got = load_timeseries_file(p)
            assert isinstance(got, np.ndarray)
            assert got.shape == arr.shape


def test_subject_grouping_and_sorting():
    with tempfile.TemporaryDirectory() as td:
        # create small timeseries files
        arr1 = np.random.randn(4, 6).astype(np.float32)
        arr2 = np.random.randn(3, 6).astype(np.float32)
        arr3 = np.random.randn(5, 6).astype(np.float32)

        f1 = _write_npy(td, 's1_ses1.npy', arr1)
        f2 = _write_npy(td, 's1_ses2.npy', arr2)
        f3 = _write_npy(td, 's2_ses1.npy', arr3)

        # Manually create dataframe with the CSV data (no session_date to avoid sorting issues in stub)
        df_data = [
            {'subject_id': 'sub-001', 'session_id': '2020-01-01', 'diagnosis': 'CN', 'timeseries_path': os.path.basename(f1)},
            {'subject_id': 'sub-001', 'session_id': '2021-01-01', 'diagnosis': 'MCI', 'timeseries_path': os.path.basename(f2)},
            {'subject_id': 'sub-002', 'session_id': '2019-06-01', 'diagnosis': 'AD', 'timeseries_path': os.path.basename(f3)},
        ]
        mock_df = MockDF(df_data)
        
        # Manually patch read_csv with mock data
        manifest = os.path.join(td, 'manifest.csv')
        import pandas as pd
        original_read_csv = pd.read_csv
        pd.read_csv = lambda path, **kw: mock_df
        try:
            ds_sub = ADNISubjectDataset(manifest, adni_root=td, min_sessions=2, sort_by_date=False)

            assert len(ds_sub) == 1, f"Expected 1 subject with >=2 sessions, got {len(ds_sub)}"
            item = ds_sub[0]
            assert item['subject_id'] == 'sub-001'
            # Sessions sorted by session_id (string sort)
            expected_sessions = ['2020-01-01', '2021-01-01']
            assert item['session_ids'] == expected_sessions, f"Expected {expected_sessions}, got {item['session_ids']}"
            assert len(item['sessions']) == 2
        finally:
            pd.read_csv = original_read_csv


def test_collate_subject_batch_and_padding():
    with tempfile.TemporaryDirectory() as td:
        arr1 = np.random.randn(6, 4).astype(np.float32)
        arr2 = np.random.randn(4, 4).astype(np.float32)
        arr3 = np.random.randn(5, 4).astype(np.float32)

        f1 = _write_npy(td, 's1_ses1.npy', arr1)
        f2 = _write_npy(td, 's1_ses2.npy', arr2)
        f3 = _write_npy(td, 's2_ses1.npy', arr3)

        manifest = os.path.join(td, 'manifest2.csv')
        with open(manifest, 'w', newline='') as mf:
            w = csv.writer(mf)
            w.writerow(['subject_id', 'session_id', 'diagnosis', 'timeseries_path'])
            w.writerow(['sub-001', 's1', 'CN', os.path.basename(f1)])
            w.writerow(['sub-001', 's2', 'CN', os.path.basename(f2)])
            w.writerow(['sub-002', 's1', 'AD', os.path.basename(f3)])

        # Manually patch read_csv with mock data
        df_data = [
            {'subject_id': 'sub-001', 'session_id': 's1', 'diagnosis': 'CN', 'timeseries_path': os.path.basename(f1)},
            {'subject_id': 'sub-001', 'session_id': 's2', 'diagnosis': 'CN', 'timeseries_path': os.path.basename(f2)},
            {'subject_id': 'sub-002', 'session_id': 's1', 'diagnosis': 'AD', 'timeseries_path': os.path.basename(f3)},
        ]
        mock_df = MockDF(df_data)
        
        import pandas as pd
        original_read_csv = pd.read_csv
        pd.read_csv = lambda path, **kw: mock_df
        try:
            ds_sub = ADNISubjectDataset(manifest, adni_root=td, min_sessions=1, sort_by_date=False)
            assert len(ds_sub) == 2
            items = [ds_sub[i] for i in range(len(ds_sub))]
            batch = ds_sub.collate_subject_batch(items)
            # shapes: (B, S, T, N, F)
            assert 'node_features' in batch
            nf = batch['node_features']
            assert nf.ndim == 5
            B, S, T, N, F = nf.shape
            assert B == 2
            assert S == 2  # max sessions among subjects
            assert batch['mask'].shape == (B, S, T)
            # ensure mask has True for actual timesteps
            assert batch['mask'][0, 0, :6].all()  # first subject first session has T=6
            assert batch['mask'][0, 1, :4].all()  # first subject second session has T=4
            assert batch['mask'][1, 0, :5].all()  # second subject first session has T=5
        finally:
            pd.read_csv = original_read_csv
