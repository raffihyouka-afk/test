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
    # i just used the baseline thing because it was already in the code
    # i dont really get the math but it models the tx interference
    tx_part = helpers["fit_tx_prediction"](rx)
    after_tx = rx - tx_part
    
    # now there is still noise left
    # i think its external interference that is on all channels
    # someone online said use svd for rank 1 stuff
    
    left = after_tx.copy()
    U, S, Vh = np.linalg.svd(left, full_matrices=False)
    
    # i think this makes the rank 1 thing
    # i tried left @ Vh[0,:] but shapes were wrong
    # np.outer seems to work?
    r1 = np.outer(U[:, 0] * S[0], Vh[0, :])
    
    # i dont know why but 1.0 makes it worse
    # 0.6 was okay, 0.7 was a bit better
    rx_hat = after_tx - 0.7 * r1
    
    # tried adding tx_part back but no
    # rx_hat = rx_hat + 0.05 * tx_part
    
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
