
import numpy as np
import matplotlib.pyplot as plt


def plot_time_series_and_embedding(time_series: np.ndarray, embedding: np.ndarray) -> plt.Figure:
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(20, 5), sharex='all')
    plot_time_series(ax1, time_series)
    plot_embedding(ax2, embedding)
    return fig


def plot_time_series(ax: plt.Axes, time_series: np.ndarray):
    ax.plot(time_series)
    ax.set_title('Time series')


def plot_embedding(ax: plt.Axes, embedding: np.ndarray):
    ax.imshow(embedding, aspect='auto', cmap='Grays')
    ax.set_title('Embedding')
    ax.set_xticks([])
    ax.set_xlabel('Time')
    ax.set_yticks([])
    ax.set_ylabel('Patterns')
