"""Generate synthetic ADNI-like longitudinal fMRI ROI timeseries and a manifest CSV.

This script produces per-session .npy timeseries files (shape T x N) and a manifest
CSV with columns: subject_id, session_id, diagnosis, session_date, timeseries_path.

Usage (CLI):
    python scripts/generate_synthetic_adni.py --outdir ./synthetic_adni --n_subjects 2000

Defaults are conservative to avoid huge disk usage; adjust CLI args to create an
"extremely large" dataset (e.g., n_subjects=10000) if you have space/time.

The generator uses `simulate_subject_sequence` from `GNN.py` to create a plausible
longitudinal adjacency progression and then simulates BOLD-like ROI timeseries
matching each session's adjacency via a low-frequency AR(1) process with spatial
covariance derived from the adjacency matrix.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import csv
import datetime
import math
import random

# Import local utilities from GNN.py
from GNN import simulate_subject_sequence


def _canonical_hrf(tr: float = 2.0, duration: float = 32.0, peak1: float = 6.0, peak2: float = 16.0, ratio: float = 0.35) -> np.ndarray:
    """Return a parameterized double-gamma HRF sampled at `tr` seconds.

    Params:
      - peak1, peak2: locations of the positive and undershoot peaks (seconds)
      - ratio: amplitude ratio of undershoot to main peak
    """
    try:
        from scipy.stats import gamma
    except Exception:
        t = np.arange(0, duration, tr)
        h = np.exp(-((t - peak1) ** 2) / (2.0 * 3.0 ** 2)) - ratio * np.exp(-((t - peak2) ** 2) / (2.0 * 7.0 ** 2))
        return (h / (h.sum() + 1e-12)).astype(np.float32)

    t = np.arange(0, duration, tr)
    p1 = gamma.pdf(t, peak1, scale=1.0)
    p2 = gamma.pdf(t, peak2, scale=1.0)
    h = p1 - ratio * p2
    h = h / (h.sum() + 1e-12)
    return h.astype(np.float32)


def timeseries_from_adjacency(
    adj: np.ndarray,
    length: int = 120,
    ar: float = 0.3,
    lowpass_smooth: float = 3.0,
    rng: np.random.RandomState = None,
    tr: float = 2.0,
    tr_jitter: float = 0.0,
    apply_hrf: bool = False,
    hrf_peak1: float = 6.0,
    hrf_peak2: float = 16.0,
    hrf_ratio: float = 0.35,
    parcel_mean: float = 1.0,
    parcel_std: float = 0.05,
    parcel_dist: str = 'normal',
    noise_model: str = 'gaussian',
    student_df: float = 4.0,
    ar_order: int = 1,
    ar_decay: float = 0.6
) -> np.ndarray:
    """Simulate T x N BOLD-like ROI timeseries given an adjacency matrix.

    New features:
      - TR and optional HRF convolution to produce BOLD-like signals
      - per-parcel amplitude scaling (parcel sizes)
      - noise model: gaussian (default) or studentt (heavy-tailed)

    Args:
        adj: (N,N) adjacency matrix with non-negative values in [0,1]
        length: number of timepoints (T)
        ar: AR(1) coefficient (0 < ar < 1)
        lowpass_smooth: kernel size for simple moving average smoothing
        rng: np.random.RandomState
        tr: repetition time (seconds)
        apply_hrf: whether to convolve neural-like signal with canonical HRF
        parcel_mean, parcel_std: control per-region amplitude scaling (simulates parcel sizes / SNR)
        noise_model: 'gaussian' or 'studentt'
        student_df: degrees of freedom for Student's t innovations

    Returns:
        timeseries: (T, N) array
    """
    if rng is None:
        rng = np.random.RandomState(int((adj.sum() * 1e6) % (2**31 - 1)))

    N = adj.shape[0]
    # Build a covariance matrix: ensure PSD by adding jitter on diagonal
    cov = adj.copy().astype(float)
    # scale covariance to have diagonal ~1
    diag = np.clip(cov.diagonal(), 0.0, None)
    if diag.max() <= 0:
        cov = cov + np.eye(N) * 0.01
    # ensure SPD by adding small value to diagonal
    cov = cov + np.eye(N) * 1e-3

    # Normalize covariance to correlation-like scale
    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1.0
    corr = cov / np.outer(std, std)
    corr = np.nan_to_num(corr)
    # add small identity to ensure positive definiteness
    corr = corr + np.eye(N) * 1e-3

    # Cholesky (with jitter fallback)
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # Add jitter until PSD
        jitter = 1e-4
        max_tries = 10
        for _ in range(max_tries):
            try:
                L = np.linalg.cholesky(corr + np.eye(N) * jitter)
                break
            except np.linalg.LinAlgError:
                jitter *= 10
        else:
            # fallback to identity
            L = np.linalg.cholesky(np.eye(N))

    # parcel amplitude scaling (simulate parcel sizes / ROI SNR differences)
    if parcel_dist == 'normal':
        parcel_scale = rng.randn(N) * parcel_std + parcel_mean
    elif parcel_dist == 'lognormal':
        parcel_scale = np.exp(rng.randn(N) * parcel_std) * parcel_mean
    elif parcel_dist == 'gamma':
        shape = max(0.1, (parcel_mean / parcel_std) ** 2)
        scale = (parcel_std ** 2) / parcel_mean
        parcel_scale = rng.gamma(shape, scale, size=N)
    else:
        parcel_scale = rng.randn(N) * parcel_std + parcel_mean
    parcel_scale = np.clip(parcel_scale, 0.05, 4.0)

    # generate neural-like signal using AR(p)
    neural = np.zeros((length, N), dtype=np.float32)
    # history buffer of last p states
    history = np.zeros((max(1, ar_order), N), dtype=np.float32)
    # AR coefficients with geometric decay so higher lags contribute less
    ar_coefs = np.array([ar * (ar_decay ** k) for k in range(ar_order)], dtype=np.float32)
    # normalize to keep stable dynamics
    ar_coefs = ar_coefs / (ar_coefs.sum() + 1e-6) * ar

    for t in range(length):
        if noise_model == 'studentt':
            innov = L @ rng.standard_t(student_df, size=N)
        else:
            innov = L @ rng.randn(N)
        val = 0.0
        for k in range(ar_order):
            val += ar_coefs[k] * history[-1 - k]
        x = val + 0.25 * innov
        neural[t] = x
        # shift history
        if ar_order > 1:
            history = np.roll(history, -1, axis=0)
            history[-1] = x
        else:
            history[-1] = x

    # apply parcel scaling
    neural = neural * parcel_scale.reshape(1, -1)

    # optional temporal lowpass smoothing
    k = max(1, int(lowpass_smooth))
    if k > 1:
        kernel = np.ones(k) / k
        from scipy.signal import convolve
        for i in range(N):
            neural[:, i] = convolve(neural[:, i], kernel, mode='same')

    # optionally convolve with HRF (allow slight TR jitter)
    if apply_hrf:
        # allow small TR jitter per session
        tr_eff = float(max(0.2, tr + rng.randn() * tr_jitter))
        h = _canonical_hrf(tr=tr_eff, peak1=hrf_peak1, peak2=hrf_peak2, ratio=hrf_ratio)
        from scipy.signal import fftconvolve
        bold = np.zeros_like(neural)
        for i in range(N):
            bold[:, i] = fftconvolve(neural[:, i], h, mode='full')[:length]
    else:
        bold = neural

    # add small measurement noise (Gaussian)
    bold = bold + rng.randn(*bold.shape) * 0.01

    # normalize each ROI timeseries to have zero-mean, unit variance
    bold = (bold - bold.mean(axis=0, keepdims=True))
    sstd = bold.std(axis=0, keepdims=True)
    sstd[sstd == 0] = 1.0
    bold = bold / sstd

    return bold.astype(np.float32)


def diagnosis_from_progression(final_progression: float) -> str:
    """Map final progression (0..1) to clinical category label."""
    if final_progression < 0.25:
        return 'CN'
    if final_progression < 0.6:
        return 'MCI'
    return 'AD'


def generate_dataset(
    outdir: str,
    n_subjects: int = 2000,
    num_regions: int = 90,
    sessions_per_subject: int = 6,
    ts_length: int = 120,
    seed: int = 0,
    tr: float = 2.0,
    apply_hrf: bool = False,
    parcel_mean: float = 1.0,
    parcel_std: float = 0.05,
    noise_model: str = 'gaussian',
    student_df: float = 4.0,
    compress: bool = False,
    save_metadata: bool = True,
    progress_interval: int = 100,
    start_subject: int = 0
):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    timeseries_dir = out / 'timeseries'
    timeseries_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out / 'manifest.csv'
    rng = np.random.RandomState(seed)

    fieldnames = ['subject_id', 'session_id', 'diagnosis', 'session_date', 'timeseries_path']
    if save_metadata:
        fieldnames += ['mmse', 'age', 'sex', 'apoe4', 'csf_amyloid', 'csf_tau']

    with open(manifest_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for s in range(start_subject, n_subjects):
            subj_id = f'SUBJ_{s:06d}'
            # if subject already has all session files present, skip (resume support)
            already_exists = True
            for t_idx in range(sessions_per_subject):
                ext = '.npz' if compress else '.npy'
                existing = timeseries_dir / f'{subj_id}_ses{t_idx:02d}{ext}'
                if not existing.exists():
                    already_exists = False
                    break
            if already_exists:
                if (s + 1) % progress_interval == 0:
                    print(f'Skipping {s+1} — already generated')
                continue
            # each subject: simulate a sequence of adjacencies & node features
            sub_seed = rng.randint(0, 2**31 - 1)
            degenerate = rng.choice(num_regions, max(1, num_regions // 12), replace=False).tolist()
            node_feats, adjs, times, meta = simulate_subject_sequence(
                num_regions=num_regions,
                num_timepoints=sessions_per_subject,
                seed=int(sub_seed),
                degenerate_regions=degenerate,
                noise_level=0.03,
                disease_heterogeneity=(rng.rand() < 0.4),
                cascade_hops=rng.randint(1, 4)
            )

            # choose an initial date
            start_date = datetime.date(2005 + rng.randint(0, 10), rng.randint(1, 12), rng.randint(1, 28))

            # derive subject-level cognitive score baseline (for metadata)
            final_adj = adjs[-1]
            deg_mean_strength = final_adj[degenerate, :].mean()
            cognitive_score = 30.0 - 12.0 * (1.0 - deg_mean_strength) + rng.randn() * 0.8
            cognitive_score = float(np.clip(cognitive_score, 0.0, 30.0))

            # demographic & genetic metadata
            age = float(np.clip(rng.randn() * 7 + 72, 55, 95))
            sex = 'M' if rng.rand() < 0.48 else 'F'
            # APOE4 allele count: 0/1/2 with rough population frequencies
            p0, p1, p2 = 0.70, 0.25, 0.05
            ap = rng.rand()
            if ap < p2:
                apoe = 2
            elif ap < p2 + p1:
                apoe = 1
            else:
                apoe = 0
            # CSF biomarkers correlated with disease progression (higher tau with disease, lower Aβ with disease-like map)
            final_prog = float(meta['disease_progression'][-1])
            csf_amyloid = float(np.clip(1.2 - 0.6 * final_prog + rng.randn() * 0.05, 0.2, 1.8))
            csf_tau = float(np.clip(0.8 + 0.9 * final_prog + rng.randn() * 0.05, 0.2, 2.5))

            # APOE4 increases risk / worsens cognitive score and biomarkers moderately
            if apoe == 1:
                cognitive_score -= 1.2
                csf_amyloid -= 0.05
                csf_tau += 0.08
            elif apoe == 2:
                cognitive_score -= 2.4
                csf_amyloid -= 0.1
                csf_tau += 0.16
            cognitive_score = float(np.clip(cognitive_score, 0.0, 30.0))

            for t_idx in range(sessions_per_subject):
                sess_id = f'{subj_id}_SES_{t_idx:02d}'
                sess_date = start_date + datetime.timedelta(days=int(365 * t_idx * 0.5))

                # derive timeseries from adjacency at this session
                adj = adjs[t_idx]
                # fractionally mix with global noise to get variety
                adj_mixed = 0.92 * adj + 0.08 * rng.rand(*adj.shape) * adj.mean()
                ts = timeseries_from_adjacency(
                    adj_mixed,
                    length=ts_length,
                    ar=0.28 + 0.1 * rng.rand(),
                    lowpass_smooth=3 + rng.randint(0, 3),
                    rng=rng,
                    tr=args.tr,
                    tr_jitter=args.tr_jitter,
                    apply_hrf=apply_hrf,
                    hrf_peak1=args.hrf_peak1,
                    hrf_peak2=args.hrf_peak2,
                    hrf_ratio=args.hrf_ratio,
                    parcel_mean=args.parcel_mean,
                    parcel_std=args.parcel_std,
                    parcel_dist=args.parcel_dist,
                    noise_model=args.noise_model,
                    student_df=args.student_df,
                    ar_order=args.ar_order,
                    ar_decay=args.ar_decay
                )

                # file path
                ext = '.npz' if compress else '.npy'
                fname = timeseries_dir / f'{subj_id}_ses{t_idx:02d}{ext}'
                if compress:
                    np.savez_compressed(str(fname), timeseries=ts.astype(np.float32))
                else:
                    np.save(str(fname), ts.astype(np.float32))

                # diagnosis based on final progression severity (we use progression at last timepoint)
                final_prog = float(meta['disease_progression'][-1])
                diagnosis = diagnosis_from_progression(final_prog)

                row = {
                    'subject_id': subj_id,
                    'session_id': sess_id,
                    'diagnosis': diagnosis,
                    'session_date': sess_date.isoformat(),
                    'timeseries_path': str(fname)
                }
                if save_metadata:
                    row['mmse'] = float(np.clip(cognitive_score + rng.randn() * 1.2, 0.0, 30.0))
                    row['age'] = age
                    row['sex'] = sex
                    row['apoe4'] = int(apoe)
                    row['csf_amyloid'] = csf_amyloid
                    row['csf_tau'] = csf_tau

                writer.writerow(row)

            if (s + 1) % progress_interval == 0:
                print(f'Generated {s+1}/{n_subjects} subjects...')

    print(f'Wrote manifest: {manifest_path}')
    print(f'Wrote timeseries files to {timeseries_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='synthetic_adni', help='Output directory for manifest and timeseries')
    parser.add_argument('--n_subjects', type=int, default=2000, help='Number of subjects to simulate (increase for extremely large datasets)')
    parser.add_argument('--num_regions', type=int, default=90, help='Number of ROIs (nodes)')
    parser.add_argument('--sessions_per_subject', type=int, default=6, help='Number of longitudinal sessions per subject')
    parser.add_argument('--ts_length', type=int, default=120, help='Number of timepoints per session')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # Realism options
    parser.add_argument('--tr', type=float, default=2.0, help='Repetition time (TR) in seconds')
    parser.add_argument('--tr_jitter', type=float, default=0.0, help='Stddev (s) of TR jitter to sample per-session (small)')
    parser.add_argument('--apply_hrf', action='store_true', help='Convolve neural-like signal with canonical HRF (produces BOLD-like signals)')
    parser.add_argument('--hrf_peak1', type=float, default=6.0, help='HRF primary peak (seconds)')
    parser.add_argument('--hrf_peak2', type=float, default=16.0, help='HRF undershoot peak (seconds)')
    parser.add_argument('--hrf_ratio', type=float, default=0.35, help='HRF undershoot ratio')
    parser.add_argument('--parcel_mean', type=float, default=1.0, help='Mean parcel amplitude (simulates parcel size/SNR)')
    parser.add_argument('--parcel_std', type=float, default=0.05, help='Stddev for parcel amplitude scaling')
    parser.add_argument('--parcel_dist', type=str, choices=['normal','lognormal','gamma'], default='normal', help='Distribution to sample parcel amplitudes')
    parser.add_argument('--noise_model', type=str, choices=['gaussian', 'studentt'], default='gaussian', help='Noise model for innovations')
    parser.add_argument('--student_df', type=float, default=4.0, help='Degrees of freedom for Student-t noise (if selected)')
    parser.add_argument('--ar_order', type=int, default=1, help='Order for AR(p) temporal model')
    parser.add_argument('--ar_decay', type=float, default=0.6, help='Geometric decay for higher AR lags')
    parser.add_argument('--compress', action='store_true', help='Save timeseries as compressed .npz files instead of .npy')
    parser.add_argument('--no_metadata', action='store_true', help='Do not save additional metadata (mmse/age/etc) in manifest')
    parser.add_argument('--progress_interval', type=int, default=100, help='How frequently to print progress during generation')
    parser.add_argument('--start_subject', type=int, default=0, help='Subject index to start generation from (for resuming)')

    args = parser.parse_args()

    generate_dataset(
        args.outdir,
        n_subjects=args.n_subjects,
        num_regions=args.num_regions,
        sessions_per_subject=args.sessions_per_subject,
        ts_length=args.ts_length,
        seed=args.seed,
        tr=args.tr,
        apply_hrf=args.apply_hrf,
        parcel_mean=args.parcel_mean,
        parcel_std=args.parcel_std,
        noise_model=args.noise_model,
        student_df=args.student_df,
        compress=args.compress,
        save_metadata=not args.no_metadata,
        progress_interval=args.progress_interval
    )
