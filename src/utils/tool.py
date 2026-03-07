import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def batchify(data, batch_size, shuffle=False):
    """
    Batch generator for list data.
    """
    n_samples = len(data)
    indices = np.arange(n_samples)
    if shuffle:  # Shuffle at the start of epoch
        np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        batch_data = [data[idx] for idx in batch_idx]
        yield batch_data


def wav_normalization(wav: np.array, mx: float=None) -> np.array:
    denom = max(abs(wav))
    if denom == 0 or np.isnan(denom):
        raise ValueError
    return wav / denom if mx is None else np.clip(wav / denom * mx, -1.0, 1.0)


def plot_length_distribution(output_path, task, tname: str, n_repeat: int=1):
    lengths = []
    for example in tqdm(task):
        duration = len(example["audio_input"]) / 16000.0 * n_repeat
        lengths.append(duration)

    plt.hist(lengths, bins=50)
    plt.title(f"Audio Duration Distribution ({tname})")
    plt.xlabel("Duration")
    plt.ylabel("Frequency")
    plt.grid(True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
