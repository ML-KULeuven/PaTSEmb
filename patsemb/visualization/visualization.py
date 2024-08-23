
import numpy as np
import matplotlib.pyplot as plt


def plot_embedding(time_series: np.ndarray, embedding: np.ndarray) -> plt.Figure:
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(20, 5), sharex='all')
    ax1.plot(time_series)
    ax1.set_title('Time series')
    ax2.imshow(embedding, aspect='auto', cmap='Grays')
    ax2.set_title('Embedding')
    ax2.set_xticks([])
    ax2.set_xlabel('Time')
    ax2.set_yticks([])
    ax2.set_ylabel('Patterns')
    return fig
