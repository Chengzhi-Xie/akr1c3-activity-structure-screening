# plot_cpptraj_analysis.py
import matplotlib.pyplot as plt
import numpy as np

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
# CONFIGURATION — adjust to match your simulation
# ══════════════════════════════════════════════════════════════════════════
TOTAL_TIME_NS = 100.0   # total simulation length in ns

# ── Helpers ────────────────────────────────────────────────────────────────
def load(fname):
    """Load CPPTRAJ .dat file → (col0, col1)."""
    x, y = [], []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                x.append(float(parts[0]))
                y.append(float(parts[1]))
    return np.array(x), np.array(y)


def load_multi(fname, ncols=3):
    """Load CPPTRAJ .dat with multiple value columns."""
    data = [[] for _ in range(ncols)]
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            parts = line.split()
            if len(parts) >= ncols:
                for i in range(ncols):
                    data[i].append(float(parts[i]))
    return [np.array(col) for col in data]


def to_ns(frames):
    """Scale frame indices so that the axis spans 0 → TOTAL_TIME_NS."""
    if len(frames) < 2:
        return frames
    return (frames - frames[0]) / (frames[-1] - frames[0]) * TOTAL_TIME_NS


# ── 1. RMSD — Protein & Ligand ────────────────────────────────────────────
def plot_rmsd():
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    frames_p, rmsd_p = load("rmsd_protein.dat")
    frames_l, rmsd_l = load("rmsd_ligand.dat")
    time_p = to_ns(frames_p)
    time_l = to_ns(frames_l)

    ax.plot(time_p, rmsd_p, color=COLORS["blue"], linewidth=0.8, alpha=0.9,
            label=f"Protein backbone (mean {np.mean(rmsd_p):.2f} Å)")
    ax.plot(time_l, rmsd_l, color=COLORS["orange"], linewidth=0.8, alpha=0.9,
            label=f"Ligand heavy atoms (mean {np.mean(rmsd_l):.2f} Å)")

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("RMSD — Protein & Ligand", pad=8)
    ax.set_xlim(0, TOTAL_TIME_NS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_off()
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("rmsd_plot.png")
    plt.close()
    print("Saved rmsd_plot.png")


# ── 2. RMSF — Per Residue ─────────────────────────────────────────────────
def plot_rmsf():
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    residues, rmsf = load("rmsf.dat")

    ax.plot(residues, rmsf, color=COLORS["green"], linewidth=0.8, alpha=0.9,
            label="Backbone RMSF")
    ax.fill_between(residues, 0, rmsf, color=COLORS["green"], alpha=0.15)

    mean_val = np.mean(rmsf)
    ax.axhline(mean_val, color=COLORS["red"], linewidth=0.8,
               linestyle="--", alpha=0.7, label=f"Mean ({mean_val:.2f} Å)")

    ax.set_xlabel("Residue Number")
    ax.set_ylabel("RMSF (Å)")
    ax.set_title("Per-Residue RMSF (Backbone Flexibility)", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_off()
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("rmsf_plot.png")
    plt.close()
    print("Saved rmsf_plot.png")


# ── 3. Radius of Gyration ─────────────────────────────────────────────────
def plot_rog():
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    frames, rog = load("rog.dat")
    time_ns = to_ns(frames)

    ax.plot(time_ns, rog, color=COLORS["orange"], linewidth=0.8, alpha=0.9,
            label="R$_g$")

    mean_val = np.mean(rog)
    ax.axhline(mean_val, color=COLORS["red"], linewidth=0.8,
               linestyle="--", alpha=0.7, label=f"Mean ({mean_val:.2f} Å)")

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Radius of Gyration (Å)")
    ax.set_title("Radius of Gyration (Compactness)", pad=8)
    ax.set_xlim(0, TOTAL_TIME_NS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_off()
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("rog_plot.png")
    plt.close()
    print("Saved rog_plot.png")


# ── 4. SASA ────────────────────────────────────────────────────────────────
def plot_sasa():
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    frames, sasa = load("surf.dat")
    time_ns = to_ns(frames)

    ax.plot(time_ns, sasa, color=COLORS["purple"], linewidth=0.8, alpha=0.9,
            label="SASA")

    mean_val = np.mean(sasa)
    ax.axhline(mean_val, color=COLORS["red"], linewidth=0.8,
               linestyle="--", alpha=0.7, label=f"Mean ({mean_val:.1f} Å²)")

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("SASA (Å$^{2}$)")
    ax.set_title("Solvent Accessible Surface Area", pad=8)
    ax.set_xlim(0, TOTAL_TIME_NS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_off()
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("sasa_plot.png")
    plt.close()
    print("Saved sasa_plot.png")


# ── 5. Hydrogen Bonds (Protein–Ligand) ────────────────────────────────────
def plot_hbonds():
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    frames_p2l, nhb_p2l = load("nhb_prot2lig.dat")
    frames_l2p, nhb_l2p = load("nhb_lig2prot.dat")
    time_p2l = to_ns(frames_p2l)
    time_l2p = to_ns(frames_l2p)

    total_hb = nhb_p2l + nhb_l2p

    ax.plot(time_p2l, nhb_p2l, color=COLORS["blue"], linewidth=0.8,
            alpha=0.9, label="Protein → Ligand")
    ax.plot(time_l2p, nhb_l2p, color=COLORS["orange"], linewidth=0.8,
            alpha=0.9, label="Ligand → Protein")
    ax.plot(time_p2l, total_hb, color=COLORS["green"], linewidth=0.8,
            alpha=0.9, label="Total")

    mean_val = np.mean(total_hb)
    ax.axhline(mean_val, color=COLORS["red"], linewidth=0.8,
               linestyle="--", alpha=0.7, label=f"Mean ({mean_val:.1f})")

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Number of H-bonds")
    ax.set_title("Protein–Ligand Hydrogen Bonds", pad=8)
    ax.set_xlim(0, TOTAL_TIME_NS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_off()
    ax.legend(loc="upper right")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig("hbonds_plot.png")
    plt.close()
    print("Saved hbonds_plot.png")


# ── 6. LIE – Electrostatic & vdW Interactions ─────────────────────────────
def plot_lie():
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True)

    frames, eelec, evdw = load_multi("lie.dat", ncols=3)
    time_ns = to_ns(frames)

    # Electrostatic
    ax = axes[0]
    ax.plot(time_ns, eelec, color=COLORS["blue"], linewidth=0.8, alpha=0.9,
            label="Electrostatic")
    mean_e = np.mean(eelec)
    ax.axhline(mean_e, color=COLORS["red"], linewidth=0.8,
               linestyle="--", alpha=0.7, label=f"Mean ({mean_e:.1f})")
    ax.set_ylabel("E$_{elec}$ (kcal mol$^{-1}$)")
    ax.set_title("Protein–Lignad Interaction Energies (LIE)", pad=8)
    ax.set_xlim(0, TOTAL_TIME_NS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_off()
    ax.legend(loc="upper right")

    # van der Waals
    ax = axes[1]
    ax.plot(time_ns, evdw, color=COLORS["orange"], linewidth=0.8, alpha=0.9,
            label="van der Waals")
    mean_v = np.mean(evdw)
    ax.axhline(mean_v, color=COLORS["red"], linewidth=0.8,
               linestyle="--", alpha=0.7, label=f"Mean ({mean_v:.1f})")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("E$_{vdW}$ (kcal mol$^{-1}$)")
    ax.set_xlim(0, TOTAL_TIME_NS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_off()
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("lie_plot.png")
    plt.close()
    print("Saved lie_plot.png")


# ── Run all ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_rmsd()
    plot_rmsf()
    plot_rog()
    plot_sasa()
    plot_hbonds()
    plot_lie()
    print("\nAll plots saved!")
