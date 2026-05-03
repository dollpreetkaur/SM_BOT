import numpy as np

def compute_temporal_features(timestamps):
    if len(timestamps) < 2:
        return [0, 0]

    timestamps = sorted(timestamps)

    gaps = [
        (timestamps[i] - timestamps[i-1]).total_seconds()
        for i in range(1, len(timestamps))
    ]

    avg_gap = np.mean(gaps)
    std_gap = np.std(gaps)

    return [avg_gap, std_gap] 