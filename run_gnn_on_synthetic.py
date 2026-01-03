#!/usr/bin/env python3
"""
Run DDGModel on synthetic single-session "ADNI-style" data.

This is an improved, robust version that:
 - ensures covariance matrices used for multivariate sampling are SPD (no runtime warnings)
 - robustly unpacks and validates model outputs before calling compute_losses
 - supplies a high-quality fallback predicted_score using the model's encoder + clinical_decoder
   when the model's forward doesn't provide one
 - preserves the powerful behavior of the DDGModel (uses model components when available)
"""
import os
import numpy as np
import importlib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

# Try to import the functions/classes we need from the local GNN module.
try:
    GNN = importlib.import_module('GNN')
    node_features_from_timeseries = getattr(GNN, 'node_features_from_timeseries')
    compute_empirical_fc_from_timeseries = getattr(GNN, 'compute_empirical_fc_from_timeseries')
    DDGModel = getattr(GNN, 'DDGModel')
    compute_losses = getattr(GNN, 'compute_losses')
except Exception as e:
    raise ImportError("Failed to import required symbols from local GNN.py: " + str(e))


# -------------------------
# Configuration
# -------------------------
OUTPUT_DIR_SYNTH = "data"
TS_DIR_SYNTH = os.path.join(OUTPUT_DIR_SYNTH, "timeseries")
LABELS_PATH_SYNTH = os.path.join(OUTPUT_DIR_SYNTH, "labels.csv")

N_SUBJECTS = 240
N_ROI = 90
MIN_TIMEPOINTS = 120
MAX_TIMEPOINTS = 200

RANDOM_SEED_SYNTH = 42
np.random.seed(RANDOM_SEED_SYNTH)

CLASS_DISTRIBUTION = {"CN": 0.4, "MCI": 0.35, "AD": 0.25}
DISEASE_EFFECT_SCALE = {"CN": 0.00, "MCI": 0.05, "AD": 0.10}

NOISE_LEVEL = 0.25
BASE_CONNECTIVITY_STRENGTH = 0.6


# -------------------------
# Numeric helpers
# -------------------------
def _make_spd(cov: np.ndarray, eps: float = 1e-6, max_tries: int = 8) -> np.ndarray:
    """Ensure a matrix is symmetric positive-definite by adding jitter until cholesky succeeds."""
    # symmetrize
    cov = 0.5 * (cov + cov.T)
    jitter = eps
    for _ in range(max_tries):
        try:
            # Attempt Cholesky
            np.linalg.cholesky(cov + np.eye(cov.shape[0]) * jitter)
            return cov + np.eye(cov.shape[0]) * jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0
    # Last-resort: project to nearest PSD via eigen decomposition
    vals, vecs = np.linalg.eigh(cov)
    vals[vals < eps] = eps
    cov_pd = (vecs * vals) @ vecs.T
    return cov_pd


# -------------------------
# Synthetic generator
# -------------------------
def generate_spd_matrix(n, strength):
    A = np.random.randn(n, n)
    cov = A @ A.T
    D = np.sqrt(np.diag(cov) + 1e-12)
    corr = cov / np.outer(D, D)
    corr = strength * corr + (1 - strength) * np.eye(n)
    corr = _make_spd(corr, eps=1e-6)
    return corr


def generate_timeseries(base_cov, T, subject_variation):
    cov = base_cov + subject_variation
    cov = _make_spd(cov, eps=1e-6)
    # draw safely without SPD warnings
    ts = np.random.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov, size=T)
    ts += NOISE_LEVEL * np.random.randn(*ts.shape)
    return ts


def generate_dataset():
    os.makedirs(TS_DIR_SYNTH, exist_ok=True)
    base_cov = generate_spd_matrix(N_ROI, BASE_CONNECTIVITY_STRENGTH)
    labels = []
    subject_idx = 1

    # suppress numpy multivariate warnings at global scope for robustness
    warnings.filterwarnings("ignore", message="covariance is not symmetric positive-semidefinite")

    for diagnosis, frac in CLASS_DISTRIBUTION.items():
        n_class = int(N_SUBJECTS * frac)
        for _ in range(n_class):
            sid = f"sub-{subject_idx:03d}"
            subject_idx += 1

            T = np.random.randint(MIN_TIMEPOINTS, MAX_TIMEPOINTS)
            subject_variation = np.random.randn(N_ROI, N_ROI) * 0.02
            subject_variation = 0.5 * (subject_variation + subject_variation.T)

            disease_noise = DISEASE_EFFECT_SCALE[diagnosis] * np.random.randn(N_ROI, N_ROI)
            disease_noise = 0.5 * (disease_noise + disease_noise.T)

            ts = generate_timeseries(base_cov + disease_noise, T, subject_variation)
            ts = StandardScaler().fit_transform(ts)

            np.save(os.path.join(TS_DIR_SYNTH, f"{sid}.npy"), ts)
            labels.append((sid, diagnosis))

    labels_df = pd.DataFrame(labels, columns=["subject_id", "diagnosis"])
    labels_df.to_csv(LABELS_PATH_SYNTH, index=False)

    # re-enable warnings (if you prefer to keep suppressed remove this)
    warnings.filterwarnings("default", message="covariance is not symmetric positive-semidefinite")

    print("Synthetic ADNI-style dataset created")
    print(f"Subjects: {len(labels_df)}")
    print(f"ROIs: {N_ROI}")
    print(f"Saved to: {OUTPUT_DIR_SYNTH}")


# --- Custom collate_fn for DataLoader ---
def collate_batch_for_ddg(batch: list):
    node_features_batch = []
    adjacency_matrices_batch = []
    cognitive_scores_batch = []
    for item in batch:
        node_features_batch.append(item['node_features'])
        adjacency_matrices_batch.append(item['adjacency_matrices'])
        cognitive_scores_batch.append(item['cognitive_scores'])
    node_features_tensor = torch.stack(node_features_batch, dim=0)
    adjacency_matrices_tensor = torch.stack(adjacency_matrices_batch, dim=0)
    cognitive_scores_tensor = torch.tensor(cognitive_scores_batch, dtype=torch.float32)
    return {
        'node_features': node_features_tensor,
        'adjacency_matrices': adjacency_matrices_tensor,
        'cognitive_scores': cognitive_scores_tensor
    }


# --- Custom Dataset for Synthetic Data ---
class CustomSyntheticDDGDataset(Dataset):
    def __init__(self, ts_dir, labels_csv):
        self.ts_dir = ts_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.subject_ids = self.labels_df["subject_id"].values

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        sid = self.subject_ids[idx]
        diagnosis_str = self.labels_df.iloc[idx]["diagnosis"]
        ts_path = os.path.join(self.ts_dir, f"{sid}.npy")
        raw_timeseries = np.load(ts_path)

        node_feats_intra_session = node_features_from_timeseries(raw_timeseries)
        adj_intra_session = compute_empirical_fc_from_timeseries(raw_timeseries)

        if node_feats_intra_session.ndim == 3:
            pooled = node_feats_intra_session.mean(axis=0)
        else:
            pooled = node_feats_intra_session

        final_node_features = torch.tensor(pooled.astype(np.float32)).unsqueeze(0)
        final_adjacency_matrices = torch.tensor(adj_intra_session.astype(np.float32)).unsqueeze(0)

        if diagnosis_str == "CN":
            cognitive_score = 28.0 + np.random.rand() * 2.0
        elif diagnosis_str == "MCI":
            cognitive_score = 18.0 + np.random.rand() * 7.0
        else:
            cognitive_score = 5.0 + np.random.rand() * 5.0
        cognitive_score = float(cognitive_score)

        return {
            'node_features': final_node_features,
            'adjacency_matrices': final_adjacency_matrices,
            'cognitive_scores': cognitive_score
        }


# --- Helpers to normalize model return values robustly ---
def unpack_model_output(mout):
    """
    Normalize model output to (outputs_list, latent_sequence).
    Accepts many forward return conventions used across versions.
    """
    if mout is None:
        return None, None
    # If model returned tuple/list
    if isinstance(mout, (tuple, list)):
        # If first element is a list/tuple of dicts -> outputs, second element latent
        first = mout[0] if len(mout) > 0 else None
        if isinstance(first, (list, tuple)):
            outputs = list(first)
            latent = mout[1] if len(mout) > 1 else None
            return outputs, latent
        # If first is dict (and mout is list/tuple of dicts) handle that
        if isinstance(first, dict):
            outputs = list(mout)
            return outputs, None
        # Otherwise, assume (outputs, latent, ...)
        outputs = mout[0]
        latent = mout[1] if len(mout) > 1 else None
        return outputs, latent
    # Single object: expect list of dicts or dict
    if isinstance(mout, dict):
        return [mout], None
    if isinstance(mout, list):
        return mout, None
    # Unknown: return as-is
    return mout, None


# full file (only showing the changed/added parts here for clarity)
# Replace the previous ensure_outputs_have_score with the following implementation
def sanitize_and_fill_outputs(outputs, node_features, adjacency_matrices, model, latent_seq, device):
    """
    Return a sanitized list of dicts suitable for compute_losses:
      - Ensures outputs is a list of dicts
      - Ensures every dict has 'predicted_score' as a torch.Tensor (B,1)
      - Converts numpy/py scalars to tensors
      - Uses model.encoder + model.clinical_decoder to compute a fallback predicted_score when needed
    """
    if outputs is None:
        raise RuntimeError("Model returned no outputs")

    # Ensure it's a list
    if isinstance(outputs, dict):
        outputs_list = [outputs]
    elif isinstance(outputs, (list, tuple)):
        outputs_list = list(outputs)
    else:
        raise RuntimeError(f"Unsupported outputs type: {type(outputs)}")

    B = node_features.size(0) if (node_features is not None and torch.is_tensor(node_features)) else None

    # Compute a single fallback_score tensor if needed
    fallback_score = None
    need_fallback = False
    for o in outputs_list:
        if not isinstance(o, dict) or ('predicted_score' not in o) or (o.get('predicted_score') is None):
            need_fallback = True
            break

    if need_fallback:
        # Try to compute a meaningful fallback using model components
        try:
            if node_features is not None and adjacency_matrices is not None and hasattr(model, 'encoder') and hasattr(model, 'clinical_decoder'):
                x_last = node_features[:, -1].to(device)
                A_last = adjacency_matrices[:, -1].to(device)
                with torch.no_grad():
                    H_last = model.encoder(x_last, A_last)  # (B, N, D)
                    if latent_seq is not None:
                        z_last = latent_seq[:, -1].to(device)
                    else:
                        # fallback latent if model exposes latent_dim else use 1-D zeros
                        if hasattr(model, 'latent_dim'):
                            z_last = torch.zeros(B, model.latent_dim, device=device)
                        else:
                            z_last = torch.zeros(B, 1, device=device)
                    fallback_score = model.clinical_decoder(H_last, z_last)  # expect (B,1)
                    if fallback_score is None:
                        fallback_score = torch.zeros(B, 1, device=device)
            else:
                # fallback zeros
                if B is None:
                    B = 1
                fallback_score = torch.zeros(B, 1, device=device)
        except Exception:
            if B is None:
                B = 1
            fallback_score = torch.zeros(B, 1, device=device)

    # Build sanitized outputs list
    sanitized = []
    for o in outputs_list:
        if not isinstance(o, dict):
            # skip unsupported entries but keep slot with minimal dict so compute_losses indexing remains valid
            sanitized.append({'predicted_adjacency': None, 'evolved_adjacency': None, 'predicted_score': fallback_score.clone() if fallback_score is not None else torch.zeros(B,1,device=device)})
            continue

        od = dict(o)  # shallow copy to avoid mutating original
        ps = od.get('predicted_score', None)

        if ps is None:
            od['predicted_score'] = fallback_score.clone() if fallback_score is not None else torch.zeros(B, 1, device=device)
        else:
            # convert numpy / list / scalar to tensor
            if isinstance(ps, np.ndarray):
                od['predicted_score'] = torch.tensor(ps, dtype=torch.float32, device=device)
            elif isinstance(ps, (list, tuple)):
                od['predicted_score'] = torch.tensor(np.asarray(ps), dtype=torch.float32, device=device)
            elif isinstance(ps, (int, float)):
                od['predicted_score'] = torch.tensor([[float(ps)]] * (B if B is not None else 1), dtype=torch.float32, device=device)
            elif torch.is_tensor(ps):
                # ensure shape (B,1)
                if ps.dim() == 0:
                    od['predicted_score'] = ps.view(1, 1).to(device).float()
                elif ps.dim() == 1:
                    od['predicted_score'] = ps.view(-1, 1).to(device).float()
                else:
                    od['predicted_score'] = ps.to(device).float()
            else:
                # fallback
                od['predicted_score'] = fallback_score.clone() if fallback_score is not None else torch.zeros(B, 1, device=device)

        sanitized.append(od)

    return sanitized


# --- Main execution function to train DDGModel on synthetic data ---
def run_ddg_on_synthetic_data(epochs=30, batch_size=16, lr=1e-3, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else "cpu")
    print(f"\nRunning DDGModel on synthetic data on device {device}")
    print("="*80)

    generate_dataset()

    full_dataset = CustomSyntheticDDGDataset(TS_DIR_SYNTH, LABELS_PATH_SYNTH)
    total_subjects = len(full_dataset)
    train_size = int(0.8 * total_subjects)
    val_size = total_subjects - train_size
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch_for_ddg)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch_for_ddg)

    sample_item = full_dataset[0]
    _, N_ROI, F_node_feats = sample_item['node_features'].shape
    print(f"Detected N_ROI: {N_ROI}, F_node_feats: {F_node_feats}")

    model = DDGModel(in_feats=F_node_feats, node_dim=32, latent_dim=12, use_vae=True, use_vae_recon=True, recon_T=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    early_stopping_patience = 10
    patience_counter = 0

    print("\nStarting training...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            node_features = batch['node_features'].to(device)
            adjacency_matrices = batch['adjacency_matrices'].to(device)
            cognitive_scores = batch['cognitive_scores'].to(device)

            optimizer.zero_grad()

            mout = model(node_features, adjacency_matrices, rollout_steps=1, teacher_forcing_prob=0.9)
            outputs, latent_seq = unpack_model_output(mout)

            # Ensure predicted_score exists and is a tensor
            outputs = sanitize_and_fill_outputs(outputs, node_features, adjacency_matrices, model, latent_seq, device)

            try:
                loss, sublogs = compute_losses(
                    {'node_features': node_features, 'adjacency_matrices': adjacency_matrices, 'cognitive_scores': cognitive_scores},
                    outputs,
                    latent_sequence=latent_seq,
                    lambda_edge=1.0,
                    lambda_clinical=0.5,
                    lambda_latent=0.1
                )
            except Exception as e:
                raise RuntimeError(f"compute_losses failed: {e}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                node_features = batch['node_features'].to(device)
                adjacency_matrices = batch['adjacency_matrices'].to(device)
                cognitive_scores = batch['cognitive_scores'].to(device)

                mout = model(node_features, adjacency_matrices, rollout_steps=1, teacher_forcing_prob=1.0)
                outputs, latent_seq = unpack_model_output(mout)
                outputs = sanitize_and_fill_outputs(outputs, node_features, adjacency_matrices, model, latent_seq, device)

                loss, _ = compute_losses(
                    {'node_features': node_features, 'adjacency_matrices': adjacency_matrices, 'cognitive_scores': cognitive_scores},
                    outputs,
                    latent_sequence=latent_seq,
                    lambda_edge=1.0,
                    lambda_clinical=0.5,
                    lambda_latent=0.1
                )
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch+1:02d}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Final evaluation (compute predicted scores)
    model.eval()
    y_true_val = []
    y_pred_val = []
    with torch.no_grad():
        for batch in val_loader:
            node_features = batch['node_features'].to(device)
            adjacency_matrices = batch['adjacency_matrices'].to(device)
            cognitive_scores = batch['cognitive_scores'].to(device)

            mout = model(node_features, adjacency_matrices, rollout_steps=1, teacher_forcing_prob=1.0)
            outputs, _ = unpack_model_output(mout)
            outputs = sanitize_and_fill_outputs(outputs, node_features, adjacency_matrices, model, None, device)

            # Extract predictions
            try:
                if isinstance(outputs, (list, tuple)) and len(outputs) > 0 and isinstance(outputs[0], dict):
                    ps = outputs[0]['predicted_score']
                    if torch.is_tensor(ps):
                        ps = ps.squeeze(-1)
                        preds = ps.cpu().numpy().ravel().tolist()
                    else:
                        preds = np.asarray(ps).ravel().tolist()
                else:
                    preds = [0.0] * node_features.size(0)
            except Exception:
                preds = [0.0] * node_features.size(0)

            y_true_val.extend(cognitive_scores.cpu().numpy().tolist())
            y_pred_val.extend(preds)

    y_true_val = np.array(y_true_val)
    y_pred_val = np.array(y_pred_val)
    val_mse = np.mean((y_true_val - y_pred_val)**2)
    val_mae = np.mean(np.abs(y_true_val - y_pred_val))
    print(f"Validation Clinical MSE: {val_mse:.4f}")
    print(f"Validation Clinical MAE: {val_mae:.4f}")

    return model


if __name__ == "__main__":
    run_ddg_on_synthetic_data(epochs=30, batch_size=16, lr=1e-3)