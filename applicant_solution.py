import json
import gdown

import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers

url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing "
downloaded_file = "challenge.mat"
gdown.download(url, downloaded_file, quiet=False, fuzzy=True)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def your_canceller(tx_n, rx):
    N_samples, n_tx = tx_n.shape
    N_samples2, n_rx = rx.shape
    
    predicted_interference = np.zeros_like(rx)
    
    for c in range(n_rx):
        feat_list = []
        
        # linear
        for i in range(n_tx):
            feat_list.append(tx_n[:, i].real)
            feat_list.append(tx_n[:, i].imag)
        
        # |tx|^2
        for i in range(n_tx):
            feat_list.append(np.abs(tx_n[:, i]) ** 2)
        
        # cross products
        for i in range(n_tx):
            for j in range(i+1, n_tx):
                cross = tx_n[:, i] * np.conj(tx_n[:, j])
                feat_list.append(cross.real)
                feat_list.append(cross.imag)
                cross2 = tx_n[:, i] * tx_n[:, j]
                feat_list.append(cross2.real)
                feat_list.append(cross2.imag)
        
        # squared
        for i in range(n_tx):
            sq = tx_n[:, i] ** 2
            feat_list.append(sq.real)
            feat_list.append(sq.imag)
        
        X = np.column_stack(feat_list)
        y = rx[:, c]
        
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coeffs
        predicted_interference[:, c] = pred
    
    rx_after = rx - predicted_interference
    
    # external interference - spatially coherent = rank 1?
    leftover = rx_after.copy()
    U, S, Vh = np.linalg.svd(leftover, full_matrices=False)
    
    # not sure about this formula but it runs
    rank1 = np.outer(U[:, 0] * S[0], Vh[0, :])
    
    # 1.0 was worse
    rx_hat = rx_after - 0.7 * rank1
    
    # rx_hat = rx_hat + 0.1 * predicted_interference  # tried, didnt help
    
    return rx_hat


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
