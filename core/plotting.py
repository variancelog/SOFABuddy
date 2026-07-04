"""
Shared plotting helpers for SOFA Buddy.

These generate matplotlib figures in-process (e.g. from the averaging post-step
running inside the worker thread). The Agg backend is used to avoid touching the
Qt main thread.
"""


def plot_diffuse_field(freqs, val_l, val_r, out_path, tilt,
                       title="Diffuse Field HRTF (Averaged)",
                       label_l="Left Ear (avg)",
                       label_r="Right Ear (avg)"):
    """
    Render a diffuse-field HRTF graph with Left (solid) and Right (dashed) channels
    and save it to *out_path*.

    Styling is intentionally identical to the per-SOFA 'df' graph produced by
    core/scripts/generate_dfhrtf.py so averaged and per-SOFA graphs look consistent.
    """
    import matplotlib
    matplotlib.use("Agg")          # must precede pyplot import; safe in worker thread
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.semilogx(freqs, val_l, label=label_l, linewidth=2, alpha=0.8)
    plt.semilogx(freqs, val_r, label=label_r, linewidth=2, alpha=0.8, linestyle='--')
    plt.title(f"{title}\nSpectral Tilt: {tilt} dB/oct")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.ylim(-25, 25)
    plt.xlim(20, 20000)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.savefig(out_path, dpi=150)
    plt.close()
