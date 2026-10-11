"""Regenerate the two additional figures as native vector PDFs from saved data.

This script does not refit models or reconstruct the seven historical PNGs.
Run: python code/generate_figures.py
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "build" / "matplotlib-cache"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_vector(fig, path):
    """Keep paths and text as vectors, including colorbar meshes."""
    for artist in fig.findobj():
        if hasattr(artist, "set_rasterized"):
            artist.set_rasterized(False)
    fig.savefig(path, format="pdf", metadata={
        "Creator": "Kernel quasi-potential: code/generate_figures.py",
        "CreationDate": None, "ModDate": None,
    })
    plt.close(fig)


def make_figures(root=ROOT):
    root = Path(root)
    output = root / "figures"
    output.mkdir(exist_ok=True)
    data = json.loads((root / "results" / "experiments.json").read_text())
    records = data["runs"]
    arrays = np.load(root / "results" / "fitted_models.npz", allow_pickle=False)
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "text.usetex": False,
    })
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for i, name in enumerate(["maier_stein", "double_well"]):
        xx, yy = arrays[f"{name}_x"], arrays[f"{name}_y"]
        truth = .5 * (xx * xx - 1) ** 2
        truth += (xx * xx + 1) * yy * yy if name == "maier_stein" else yy * yy
        limit = max(truth.max(), *(arrays[f"{name}_{m}_grid_value"].max()
                                   for m in ["direct", "koopman"]))
        for j, method in enumerate(["truth", "direct", "koopman"]):
            value = truth if method == "truth" else arrays[f"{name}_{method}_grid_value"]
            contour = axes[i, j].contourf(xx, yy, value,
                                         levels=np.linspace(0, limit, 16), cmap="viridis")
            axes[i, j].set_title(f'{name.replace("_", " ").title()} | {method.title()}')
            axes[i, j].set_xlabel("$x_1$")
            axes[i, j].set_ylabel("$x_2$")
            bar = fig.colorbar(contour, ax=axes[i, j], shrink=.8)
            if bar.solids is not None:
                bar.solids.set_rasterized(False)
    save_vector(fig, output / "comparison_2d.pdf")
    arrays.close()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    names = ["maier_stein", "double_well", "multiplicative", "rotational_8d"]
    for j, metric in enumerate(["rel_l2", "hj_rms"]):
        for method in ["riccati", "direct", "koopman"]:
            means = []
            for name in names:
                selected = [row[metric] for row in records
                            if row["problem"] == name and row["method"] == method]
                if len(selected) != 3:
                    raise ValueError(f"Expected three seeds: {name}, {method}")
                means.append(np.mean(selected))
            axes[j].plot(range(4), np.maximum(means, 1e-15), "o-", label=method.title())
        axes[j].set_yscale("log")
        axes[j].set_xticks(range(4), ["MS 2D", "DW 2D", "Mult. 3D", "Rot. 8D"])
        axes[j].set_title("Relative potential error" if j == 0
                          else "Held-out HJ residual (RMS)")
        axes[j].legend()
        axes[j].grid(alpha=.2)
    save_vector(fig, output / "ablation.pdf")
    print("Generated native vector PDFs: figures/ablation.pdf, figures/comparison_2d.pdf")
    print("Generate the seven recovered-code counterparts with code/plot_recovered_runs.py.")


if __name__ == "__main__":
    make_figures()
