# plot_md_analysis.py
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

# ── Helper ─────────────────────────────────────────────────────────────────
def load(fname):
    times, vals = [], []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                times.append(float(parts[0]))
                vals.append(float(parts[1]))
    return np.array(times) / 1000, np.array(vals)   # ps → ns, values


# ── 1. Energy ──────────────────────────────────────────────────────────────
def plot_energy():
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    files = {
        "E$_{tot}$": ("summary.ETOT",  COLORS["blue"]),
        "E$_{pot}$": ("summary.EPTOT", COLORS["orange"]),
        "E$_{kin}$": ("summary.EKTOT", COLORS["green"]),
    }

    for label, (fname, color) in files.items():
        t, e = load(fname)
        ax.plot(t, e, label=label, color=color, linewidth=0.8, alpha=0.9)

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Energy (kcal mol$^{-1}$)")
    ax.set_title("MD Simulation Energy Components", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_off()
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("energy_plot.png")
    plt.close()
    print("Saved energy_plot.png")


# ── 2. Temperature ─────────────────────────────────────────────────────────
def plot_temperature(target_temp=300):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    t, temp = load("summary.TEMP")
    ax.plot(t, temp, color=COLORS["blue"], linewidth=0.8, alpha=0.9,
            label="Temperature")
    ax.axhline(target_temp, color=COLORS["red"], linewidth=0.8,
               linestyle="--", alpha=0.7, label=f"Target ({target_temp} K)")

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("MD Simulation Temperature", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_off()
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("temperature_plot.png")
    plt.close()
    print("Saved temperature_plot.png")


# ── 3. Density ─────────────────────────────────────────────────────────────
def plot_density():
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    t, dens = load("summary.DENSITY")
    ax.plot(t, dens, color=COLORS["green"], linewidth=0.8, alpha=0.9,
            label="Density")

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Density (g cm$^{-3}$)")
    ax.set_title("MD Simulation Density", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_off()
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("density_plot.png")
    plt.close()
    print("Saved density_plot.png")


# ── 4. Volume ──────────────────────────────────────────────────────────────
def plot_volume():
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    t, vol = load("summary.VOLUME")
    ax.plot(t, vol, color=COLORS["orange"], linewidth=0.8, alpha=0.9,
            label="Volume")

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Volume (Å$^{3}$)")
    ax.set_title("MD Simulation Volume", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_off()
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("volume_plot.png")
    plt.close()
    print("Saved volume_plot.png")


# ── 5. Pressure ────────────────────────────────────────────────────────────
def plot_pressure():
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    t, pres = load("summary.PRES")
    window      = max(1, len(pres) // 50)
    pres_smooth = np.convolve(pres, np.ones(window) / window, mode="same")

    ax.plot(t, pres,        color=COLORS["purple"], linewidth=0.5,
            alpha=0.35, label="Pressure (raw)")
    ax.plot(t, pres_smooth, color=COLORS["purple"], linewidth=1.2,
            alpha=0.95, label=f"Running avg (n={window})")
    ax.axhline(1.0, color=COLORS["red"], linewidth=0.8, linestyle="--",
               alpha=0.7, label="1 atm")

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Pressure (bar)")
    ax.set_title("MD Simulation Pressure", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_off()
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("pressure_plot.png")
    plt.close()
    print("Saved pressure_plot.png")


# ── Run all ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_energy()
    plot_temperature(target_temp=300)   # ← change to match your temp0
    plot_density()
    plot_volume()
    plot_pressure()
