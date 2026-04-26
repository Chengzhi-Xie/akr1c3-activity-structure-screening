# plot_pca.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

plt.rcParams.update({
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,

    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.direction":    "in",
    "ytick.direction":    "in",

    "xtick.top":          False,
    "ytick.right":        False,

    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",

    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.facecolor":     "white",
    "figure.facecolor":   "white",
    "axes.grid":          False,
})

COLORS = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "purple": "#CC79A7",
    "red":    "#D55E00",
}

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
TOTAL_TIME_NS        = 100.0
PROJECTION_FILE      = "pca_projections.dat"
SMOOTH_SIGMA         = 2.5    # Gaussian smoothing (increase → smoother basins)
ENERGY_CAP           = 5.0    # kcal/mol — cap ΔG to focus color range on basins
NBINS                = 100    # histogram bins per axis
TEMPERATURE          = 300.0  # K — match your simulation temperature


# ── Helpers ────────────────────────────────────────────────────────────────
def load_projections(fname):
    """Load cpptraj projection file → (frames, PC1, PC2, PC3)."""
    frames, pc1, pc2, pc3 = [], [], [], []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                frames.append(float(parts[0]))
                pc1.append(float(parts[1]))
                pc2.append(float(parts[2]))
                pc3.append(float(parts[3]))
    return np.array(frames), np.array(pc1), np.array(pc2), np.array(pc3)


def to_ns(frames):
    """Scale frame indices → 0 to TOTAL_TIME_NS."""
    if len(frames) < 2:
        return frames
    return (frames - frames[0]) / (frames[-1] - frames[0]) * TOTAL_TIME_NS


# ── 1. PC1 vs PC2 Scatter (coloured by time) ─────────────────────────────
def plot_pc1_vs_pc2():
    frames, pc1, pc2, pc3 = load_projections(PROJECTION_FILE)
    time_ns = to_ns(frames)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    sc = ax.scatter(pc1, pc2, c=time_ns, cmap="viridis", s=4, alpha=0.6,
                    edgecolors="none", rasterized=True)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Time (ns)")

    ax.set_xlabel("PC1 (Å)")
    ax.set_ylabel("PC2 (Å)")
    ax.set_title("PCA — PC1 vs PC2 (coloured by time)", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("pca_pc1_pc2.png")
    plt.close()
    print("Saved pca_pc1_pc2.png")


# ── 2. PC1 vs PC3 Scatter (coloured by time) ─────────────────────────────
def plot_pc1_vs_pc3():
    frames, pc1, pc2, pc3 = load_projections(PROJECTION_FILE)
    time_ns = to_ns(frames)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    sc = ax.scatter(pc1, pc3, c=time_ns, cmap="viridis", s=4, alpha=0.6,
                    edgecolors="none", rasterized=True)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Time (ns)")

    ax.set_xlabel("PC1 (Å)")
    ax.set_ylabel("PC3 (Å)")
    ax.set_title("PCA — PC1 vs PC3 (coloured by time)", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("pca_pc1_pc3.png")
    plt.close()
    print("Saved pca_pc1_pc3.png")


# ── 3. Free Energy Landscape (PC1 vs PC2) ────────────────────────────────
def plot_free_energy():
    frames, pc1, pc2, pc3 = load_projections(PROJECTION_FILE)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # 2D histogram → Gaussian smooth → free energy
    H, xedges, yedges = np.histogram2d(pc1, pc2, bins=NBINS, density=True)
    H_smooth = gaussian_filter(H, sigma=SMOOTH_SIGMA)

    kB = 0.001987204  # kcal/(mol·K)
    kT = kB * TEMPERATURE

    # Compute ΔG; unvisited bins → NaN
    H_safe = np.where(H_smooth > 0, H_smooth, np.nan)
    G = -kT * np.log(H_safe / np.nanmax(H_safe))

    # Cap energy and fill unvisited bins at the cap
    G_capped = np.clip(G, 0, ENERGY_CAP)
    G_filled = np.where(np.isnan(G_capped), ENERGY_CAP, G_capped)

    # Heatmap — blue = low ΔG (basins), red = high ΔG / unsampled
    im = ax.pcolormesh(xedges, yedges, G_filled.T,
                       cmap="jet", vmin=0, vmax=ENERGY_CAP,
                       shading="flat", rasterized=True)

    # Contour lines to delineate basins
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xcenters, ycenters)
    contour_levels = np.arange(0.5, ENERGY_CAP, 0.5)
    ax.contour(X, Y, G_filled.T, levels=contour_levels,
               colors="k", linewidths=0.4, alpha=0.5)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("ΔG (kcal mol$^{-1}$)")

    ax.set_xlabel("PC1 (Å)")
    ax.set_ylabel("PC2 (Å)")
    ax.set_title("Free Energy Landscape — PC1 vs PC2", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("pca_free_energy.png")
    plt.close()
    print("Saved pca_free_energy.png")


# ── Run all ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_pc1_vs_pc2()
    plot_pc1_vs_pc3()
    plot_free_energy()
    print("\nAll PCA plots saved!")
