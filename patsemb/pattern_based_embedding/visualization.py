
import numpy as np
import matplotlib.pyplot as plt


def plot_time_series_and_embedding(time_series: np.ndarray, embedding: np.ndarray, *, time_stamps=None, **kwargs) -> plt.Figure:
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(20, 5), sharex='all')
    plot_time_series(ax1, time_series, time_stamps=time_stamps)
    plot_embedding(ax2, embedding, **kwargs)
    ax2.set_xlabel('Time')
    return fig


def plot_time_series(ax: plt.Axes, time_series: np.ndarray, *, time_stamps=None) -> None:
    if time_stamps is None:
        ax.plot(time_series)
        ax.set_xticks([])
    else:
        ax.plot(time_stamps, time_series)
    ax.set_title('Time series')


def plot_embedding(ax: plt.Axes, embedding: np.ndarray, *, show_pattern_ids: bool = True) -> None:
    ax.imshow(embedding, aspect='auto', cmap='gist_gray_r')
    ax.set_title('Embedding')
    if show_pattern_ids:
        ax.set_yticks(ticks=range(embedding.shape[0]), labels=range(embedding.shape[0]))
    else:
        ax.set_yticks([])
    ax.set_ylabel('Patterns')
