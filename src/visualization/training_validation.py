import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve


def save_roc_curve_plot(model, x, y, path):
    plt.figure(figsize=(8, 8))
    plot_roc_curve(model, x, y)
    plt.savefig(path)
